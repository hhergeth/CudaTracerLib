#pragma once

#include <boost/filesystem.hpp>

#include "Primitives.h"
#include "Utils.h"
#include "PropertyParser.h"

#include <Base/Buffer.h>
#include <SceneTypes/PhaseFunction.h>
#include <SceneTypes/BSDF.h>
#include <SceneTypes/Light.h>
#include <SceneTypes/Sensor.h>
#include <SceneTypes/Texture.h>
#include <SceneTypes/Volumes.h>
#include <Engine/MaterialLib.h>
#include <Engine/Mesh.h>
#include <Engine/Material.h>
#include <SceneTypes/Node.h>
#include <Math/MathFunc.h>
#include <Math/half.h>
#include <Base/CudaRandom.h>

#define START_PARSE(TYPE, PARSERTYPE) \
	if (S.ref_storage.is_ref(node)) \
		return *S.ref_storage.get<TYPE>(node); \
	auto T = node.get_attribute("type"); \
	std::string err_type = #TYPE; \
	auto P = PARSERTYPE();
#define CALL_PARSE(NAME) \
	if (T == #NAME) { \
		auto C = P.NAME(node, S); \
		S.ref_storage.add_helper(node, C); \
		C.As()->Update(); \
		return C; }
#define CALL_PARSE_NO_UPDATE(NAME) \
	if (T == #NAME) \
		return S.ref_storage.add_helper(node, P.NAME(node, S));
#define END_PARSE() \
	throw std::runtime_error("invalid " + err_type + " type : " + T);

namespace CudaTracerLib {

inline StreamReference<Node> load_virtual(const std::string name, ParserState& S, const std::string& data)
{
	auto str = MemInputStream((const unsigned char*)data.c_str(), data.size(), false);
	auto obj = S.scene.CreateNode(name + ".obj", str);
	//these virtual meshes are generated without names
	S.scene.getMaterials(obj)(0)->Name = decltype(S.scene.getMaterials(obj)(0)->Name)("default");
	S.scene.getMaterials(obj)(0)->NodeLightIndex = -1;
	return obj;
}

inline Spectrum parseColor(const XMLNode& node, ParserState& S)
{
	if (node.name() == "srgb")
		return parseRGB(node, S, true);
	else if (node.name() == "rgb")
		return parseRGB(node, S, false);
	else if (node.name() == "spectrum")
		return parseSpectrum(node, S);
	else if (node.name() == "float")
		return S.def_storage.as_float(node, "value");
	else throw std::runtime_error("invalid spectrum type : " + node.name());
}

inline Spectrum tryParseColor(const XMLNode& node, ParserState& S, const std::string& prop_name, const Spectrum& def_value)
{
	if (node.has_property(prop_name))
		return parseColor(node.get_property(prop_name), S);
	else return def_value;
}

class TextureParser
{
public:
	Texture bitmap(const XMLNode& node, ParserState& S)
	{
		auto filepath = S.def_storage.prop_string(node, "filename");
		float uoff = S.def_storage.prop_float(node, "uoff", 0.0f);
		float voff = S.def_storage.prop_float(node, "voff", 0.0f);
		float uscale = S.def_storage.prop_float(node, "uscale", 1.0f);
		float vscale = S.def_storage.prop_float(node, "vscale", 1.0f);

		return CreateAggregate<Texture>(ImageTexture(TextureMapping2D(uscale, vscale, uoff, voff), S.map_asset_filepath(filepath), 1.0f));
	}

	Texture checkerboard(const XMLNode& node, ParserState& S)
	{
		Spectrum color0 = tryParseColor(node, S, "color0", 0.4f);
		Spectrum color1 = tryParseColor(node, S, "color1", 0.2f);
		float uoff = S.def_storage.prop_float(node, "uoff", 0.0f);
		float voff = S.def_storage.prop_float(node, "voff", 0.0f);
		float uscale = S.def_storage.prop_float(node, "uscale", 1.0f);
		float vscale = S.def_storage.prop_float(node, "vscale", 1.0f);

		return CreateAggregate<Texture>(CheckerboardTexture(color0, color1, TextureMapping2D(uscale, vscale, uoff, voff)));
	}

	Texture wireframe(const XMLNode& node, ParserState& S)
	{
		Spectrum interiorColor = tryParseColor(node, S, "interiorColor", 0.5f);
		Spectrum edgeColor = tryParseColor(node, S, "edgeColor", 0.1f);

		return CreateAggregate<Texture>(WireframeTexture(0.1f, interiorColor, edgeColor));
	}

	static Texture parse(const XMLNode& node, ParserState& S)
	{
		START_PARSE(Texture, TextureParser);
		CALL_PARSE(bitmap);
		CALL_PARSE(checkerboard);
		CALL_PARSE(wireframe);
		END_PARSE();
	}
};

inline Texture parseTexturOrColor(const XMLNode& node, ParserState& S)
{
	if (node.name() == "texture")
		return TextureParser::parse(node, S);
	else if (node.name() == "ref")
		return *S.ref_storage.get<Texture>(node);
	else return CreateAggregate<Texture>(ConstantTexture(parseColor(node, S)));
}

inline Texture tryParseTexturOrColor(const XMLNode& node, ParserState& S, const std::string& prop_name, const Spectrum& def_value)
{
	if (node.has_property(prop_name))
		return parseTexturOrColor(node.get_property(prop_name), S);
	else return CreateAggregate<Texture>(ConstantTexture(def_value));
}

class PhaseFunctionParser
{
public:
	PhaseFunction isotropic(const XMLNode& node, ParserState& S)
	{
		return CreateAggregate<PhaseFunction>(IsotropicPhaseFunction());
	}

	PhaseFunction hg(const XMLNode& node, ParserState& S)
	{
		float g = S.def_storage.as_float(node.get_property("g"), "value");
		return CreateAggregate<PhaseFunction>(HGPhaseFunction(g));
	}

	PhaseFunction rayleigh(const XMLNode& node, ParserState& S)
	{
		return CreateAggregate<PhaseFunction>(RayleighPhaseFunction());
	}

	PhaseFunction kkay(const XMLNode& node, ParserState& S)
	{
		float ks = S.def_storage.prop_float(node, "ks", 0.4f);
		float kd = S.def_storage.prop_float(node, "kd", 0.2f);
		float e = S.def_storage.prop_float(node, "e", 4.0f);

		return CreateAggregate<PhaseFunction>(KajiyaKayPhaseFunction(ks, kd, e));
	}

	static PhaseFunction parse(const XMLNode& node, ParserState& S)
	{
		START_PARSE(PhaseFunction, PhaseFunctionParser);
		CALL_PARSE(isotropic);
		CALL_PARSE(hg);
		CALL_PARSE(rayleigh);
		CALL_PARSE(kkay);
		END_PARSE();
	}
};

class MediumParser
{
	float4x4 toWorld;
public:
	VolumeRegion homogeneous(const XMLNode& node, ParserState& S)
	{
		PhaseFunction f = CreateAggregate<PhaseFunction>(IsotropicPhaseFunction());
		if (node.has_child_node("phase"))
			f = PhaseFunctionParser::parse(node.get_child_node("phase"), S);

		Spectrum sigma_a = 0.0, sigma_s = 0.0;
		if (node.has_property("material"))
		{
			auto mat_name = S.def_storage.prop_string(node, "material");
			if (!MaterialLibrary::hasMat(mat_name))
				throw std::runtime_error("invalid mat name : " + mat_name);
			sigma_a = MaterialLibrary::getSigmaA(mat_name);
			sigma_s = MaterialLibrary::getSigmaS(mat_name);
		}
		else
		{
			if (node.has_property("sigmaT") || node.has_property("albedo"))
			{
				Spectrum sigma_t = node.has_property("sigmaT") ? parseColor(node.get_property("sigmaT"), S) : Spectrum(1.0f);
				Spectrum albedo = node.has_property("albedo") ? parseColor(node.get_property("albedo"), S) : Spectrum(1.0f);
				sigma_s = albedo * sigma_t;
				sigma_a = sigma_t - sigma_s;
			}
			else if (node.has_property("sigmaA") || node.has_property("sigmaS"))
			{
				sigma_a = node.has_property("sigmaA") ? parseColor(node.get_property("sigmaA"), S) : Spectrum(0.0f);
				sigma_s = node.has_property("sigmaS") ? parseColor(node.get_property("sigmaS"), S) : Spectrum(0.0f);
			}
			else sigma_a = sigma_s = 0.0f;// throw std::runtime_error("invalid homogenous medium init");
		}

		float scale = S.def_storage.prop_float(node, "scale", 1.0f);

		auto R = HomogeneousVolumeDensity(f, toWorld, sigma_a * scale, sigma_s * scale, 0.0f);
		R.Update();
		return CreateAggregate<VolumeRegion>(R);
	}

	VolumeRegion heterogeneous(const XMLNode& node, ParserState& S)
	{
		PhaseFunction f = CreateAggregate<PhaseFunction>(IsotropicPhaseFunction());
		if (node.has_child_node("phase"))
			f = PhaseFunctionParser::parse(node.get_child_node("phase"), S);
		float scale = S.def_storage.prop_float(node, "scale", 1.0f);

		struct VolData
		{
			bool is_constant = true;

			Spectrum const_value = 0.0f;

			Vec3u cell_dims;
			std::vector<float> data;
			AABB box;

			Vec3u dims() const
			{
				return is_constant ? Vec3u(1) : cell_dims;
			}

			//pos is [0,1]^3
			float eval(const Vec3f& pos) const
			{
				if(is_constant)
					return const_value.getLuminance();

				int x = int(pos.x * cell_dims.x), y = int(pos.y * cell_dims.y), z = int(pos.z * cell_dims.z);
				return data[idx(x,y,z)];
			}

			size_t idx(int xpos, int ypos, int zpos, int chan = 0, int num_channels = 1) const
			{
				return ((zpos*cell_dims.y + ypos)*cell_dims.x + xpos)*num_channels + chan;
			}
		};

		boost::optional<float4x4> volume_to_world;
		auto parse_volume = [&](const XMLNode& vol_node)
		{
			VolData dat;
			if(vol_node.get_attribute("type") == "constvolume")
			{
				dat.is_constant = true;
				dat.const_value = parseColor(vol_node.get_property("value"), S);
				if(vol_node.has_property("toWorld"))
				{
					if(volume_to_world)
						throw std::runtime_error("only one volume to world matrix allowed!");
					volume_to_world = parseMatrix(vol_node.get_property("toWorld"), S);
				}
			}
			else if(vol_node.get_attribute("type") == "gridvolume")
			{
				dat.is_constant = false;
				auto filename = S.def_storage.prop_string(vol_node, "filename");
				filename = S.map_asset_filepath(filename);
				boost::optional<std::tuple<Vec3f, Vec3f>> optional_aabb;
				if(vol_node.has_property("toWorld") || vol_node.has_property("min"))
				{
					if(volume_to_world)
						throw std::runtime_error("only one volume to world matrix allowed!");
					if(vol_node.has_property("toWorld"))
						volume_to_world = parseMatrix(vol_node.get_property("toWorld"), S);
					if(vol_node.has_property("min"))//max must also exist
					{
						auto min_vol = parseVector(vol_node.get_property("min"), S);
						auto max_vol = parseVector(vol_node.get_property("max"), S);
						optional_aabb = std::make_tuple(min_vol, max_vol);
					}
				}

				std::ifstream ser_str(filename, std::ios::binary);
				enum EVolumeType {
					EFloat32 = 1,
					EFloat16 = 2,
					EUInt8 = 3,
					EQuantizedDirections = 4
				};
				uint8_t header[4];
				ser_str.read((char*)header, 4);
				if(header[0] != 'V' || header[1] != 'O' || header[2] != 'L' || header[3] != 3)
					throw std::runtime_error("expected VOL3 header");
				EVolumeType data_type;
				ser_str.read((char*)&data_type, sizeof(data_type));
				ser_str.read((char*)&dat.cell_dims, sizeof(dat.cell_dims));
				uint32_t num_channels;
				ser_str.read((char*)&num_channels, sizeof(num_channels));
				ser_str.read((char*)&dat.box, sizeof(dat.box));

				size_t N = dat.cell_dims.x * dat.cell_dims.y * dat.cell_dims.z;
				dat.data.resize(N);
				size_t val_size = data_type == EVolumeType::EFloat32 ? sizeof(float) : (data_type == EVolumeType::EFloat16 ? sizeof(half) : sizeof(uint8_t));
				std::vector<uint8_t> buffer(N * num_channels * val_size);
				ser_str.read((char*)buffer.data(), buffer.size());

				for(size_t i = 0; i < N; i++)
				{
					float sum = 0;
					for(uint32_t channel = 0; channel < num_channels; channel++)
					{
						uint8_t* ptr = &buffer[(i * num_channels + channel) * val_size];
						if(data_type == EVolumeType::EFloat32)
						{
							sum += *(float*)ptr;
						}
						else if(data_type == EVolumeType::EFloat16)
						{
							sum += ((half*)ptr)->ToFloat();
						}
						else if(data_type == EVolumeType::EUInt8)
						{
							sum += (*ptr) / 255.0f;
						}
						else throw std::runtime_error("unsopported volume data type : " + std::to_string((int)data_type));
					}

					dat.data[i] = sum / num_channels;
				}

				if(optional_aabb)
					dat.box = AABB(std::get<0>(optional_aabb.get()), std::get<1>(optional_aabb.get()));

				auto vtow = volume_to_world ? volume_to_world.get() : float4x4::Identity();
				if(distance(dat.box.minV, Vec3f(0)) > 1e-3f || distance(dat.box.maxV, Vec3f(1)) > 1e-3f)
					volume_to_world = vtow % float4x4::Translate(dat.box.minV) % float4x4::Scale(dat.box.Size());
			}
			else throw std::runtime_error("invalid volume type : " + vol_node.get_attribute("type"));
			return dat;
		};

		VolData density_data;
		VolData albedo_data;

		if(node.has_property("density"))
			density_data = parse_volume(node.get_property("density"));
		if(node.has_property("albedo"))
			albedo_data = parse_volume(node.get_property("albedo"));

		auto vol_to_world = volume_to_world ? volume_to_world.get() : parseMatrix_Id(S);
		vol_to_world = toWorld % vol_to_world;// the order here is unclear
		if(density_data.is_constant && albedo_data.is_constant)
		{
			Spectrum sigma_s = albedo_data.const_value * density_data.const_value * scale;
			Spectrum sigma_a = density_data.const_value  * scale - sigma_s;
			return CreateAggregate<VolumeRegion>(HomogeneousVolumeDensity(f, vol_to_world, sigma_a, sigma_s, 0.0f));
		}
		else
		{
			auto max_dims = max(density_data.dims(), albedo_data.dims());
			auto G = VolumeGrid(f, vol_to_world, S.scene.getTempBuffer(), max_dims, max_dims, Vec3u(1));
			G.sigAMax = G.sigSMax = 1.0f;

			auto scaleF = Vec3f((float)max_dims.x, (float)max_dims.y, (float)max_dims.z);
			for(unsigned int x = 0; x < max_dims.x; x++)
				for(unsigned int y = 0; y < max_dims.y; y++)
					for(unsigned int z = 0; z < max_dims.z; z++)
					{
						auto pos = Vec3f((float)x, (float)y, (float)z) / scaleF;
						float density = density_data.eval(pos) * scale;
						float albedo = albedo_data.eval(pos);
						G.gridS.value(x,y,z) = albedo * density;
						G.gridA.value(x, y, z) = density * (1.0f - albedo);
					}

			G.Update();
			return CreateAggregate<VolumeRegion>(G);
		}
	}

	static VolumeRegion parse(const XMLNode& node, ParserState& S, const float4x4& toWorld = float4x4::Identity())
	{
		START_PARSE(VolumeRegion, MediumParser);
		P.toWorld = toWorld;
		CALL_PARSE_NO_UPDATE(homogeneous);
		CALL_PARSE_NO_UPDATE(heterogeneous);
		END_PARSE();
	}
};

class SensorParser
{
	void parseGeneric(Sensor& C, const XMLNode& node, ParserState& S)
	{
		int width = 768, height = 576;
		if (node.has_child_node("film"))
		{
			width = S.def_storage.prop_int(node.get_child_node("film"), "width", width);
			height = S.def_storage.prop_int(node.get_child_node("film"), "height", height);
		}
		S.set_film_resolution(width, height);
		C.SetFilmData(width, height);

		auto* camera = C.As();
		camera->SetNearFarDepth(S.def_storage.prop_float(node, "nearClip", 1e-2f), S.def_storage.prop_float(node, "farClip", 10000.0f));
		camera->SetApperture(0.0);

		auto set_diagonal_fov = [&](float diag_fov)
		{
			float aspect = width / float(height);
			float diagonal = 2 * std::tan(0.5f * diag_fov);
			float width = diagonal / std::sqrt(1.0f + 1.0f / (aspect*aspect));
			camera->SetFov(2 * std::atan(width*0.5f));
		};

		if (node.has_property("focalLength"))
		{
			float focal_length = S.def_storage.prop_float(node, "focalLength");
			float c = std::sqrt((float)(36 * 36 + 24 * 24));
			float dfov = 2 * std::atan(c / (2 * focal_length));
			set_diagonal_fov(dfov);
		}

		if (node.has_property("fov"))
		{
			float fov_val = S.def_storage.prop_float(node, "fov");
			std::string fov_axis = S.def_storage.prop_string(node, "fovAxis", std::string("x"));

			auto set_vertical_fov = [&](float fov)
			{
				camera->SetFov(fov * height / (float)width);
			};

			if (fov_axis == "x")
				camera->SetFov(fov_val);
			else if (fov_axis == "y")
				set_vertical_fov(fov_val);
			else if (fov_axis == "diagonal")
				set_diagonal_fov(fov_val);
			else if (fov_axis == "smaller")
			{
				if (width < height)
					camera->SetFov(fov_val);
				else set_vertical_fov(fov_val);
			}
			else if (fov_axis == "larger")
			{
				if (width < height)
					set_vertical_fov(fov_val);
				else camera->SetFov(fov_val);
			}
			else throw std::runtime_error("invalid fov axis type : " + fov_axis);
		}

		if (node.has_property("toWorld"))
		{
			auto T = parseMatrix(node.get_property("toWorld"), S);
			//C.SetToWorld(T.Translation(), T.Translation() + T.Forward(), Vec3f(0,0,1));
			C.SetToWorld(T.Translation(), T.Forward());
		}
		C.As()->Update();
	}
public:
	Sensor perspective(const XMLNode& node, ParserState& S)
	{
		auto C = CreateAggregate<Sensor>(PerspectiveSensor());
		parseGeneric(C, node, S);

		return C;
	}

	Sensor thinlens(const XMLNode& node, ParserState& S)
	{
		auto C = CreateAggregate<Sensor>(ThinLensSensor());
		parseGeneric(C, node, S);

		float fdist = S.def_storage.prop_float(node, "focusDistance", 0.0f);
		C.As<ThinLensSensor>()->SetFocalDistance(fdist);
		if (node.has_property("apertureRadius"))
			C.As<ThinLensSensor>()->SetApperture(S.def_storage.prop_float(node, "apertureRadius"));

		return C;
	}

	Sensor orthographic(const XMLNode& node, ParserState& S)
	{
		auto C = CreateAggregate<Sensor>(OrthographicSensor());
		parseGeneric(C, node, S);
		return C;
	}

	Sensor telecentric(const XMLNode& node, ParserState& S)
	{
		auto C = CreateAggregate<Sensor>(TelecentricSensor());
		parseGeneric(C, node, S);
		return C;
	}

	static Sensor parse(const XMLNode& node, ParserState& S)
	{
		START_PARSE(Sensor, SensorParser);
		CALL_PARSE(perspective);
		CALL_PARSE(thinlens);
		CALL_PARSE(orthographic);
		CALL_PARSE(telecentric);
		END_PARSE();
	}
};

class LightParser
{
	StreamReference<Light> add_to_scene(const Light& L, ParserState& S)
	{
		return S.scene.CreateLight(L);
	}

	StreamReference<Light> parseSun(const XMLNode& node, ParserState& S)
	{
		Vec3f dir;
		if(node.has_property("sunDirection"))
		{
			dir = parseVector(node.get_property("sunDirection"), S);
		}
		else
		{
			int year = S.def_storage.prop_int(node, "year", 2010);
			int month = S.def_storage.prop_int(node, "month", 7);
			int day = S.def_storage.prop_int(node, "day", 10);
			float hour = S.def_storage.prop_float(node, "hour", 15.0f);
			float minute = S.def_storage.prop_float(node, "minute", 0.0f);
			float second = S.def_storage.prop_float(node, "second", 0.0f);
			float latitude = S.def_storage.prop_float(node, "latitude", 35.6894f);
			float longitude = S.def_storage.prop_float(node, "longitude", 139.6917f);
			float timezone = (float)S.def_storage.prop_int(node, "timezone", 9);

			//all code from Mitsuba which is based on "Computing the Solar Vector" by Manuel Blanco-Muriel,
			//Diego C. Alarcon-Padilla, Teodoro Lopez-Moratalla, and Martin Lara-Coira in "Solar energy", vol 27, number 5, 2001 by Pergamon Press
			// Main variables
			double elapsedJulianDays, decHours;
			double eclipticLongitude, eclipticObliquity;
			double rightAscension, declination;
			double elevation, azimuth;

			// Auxiliary variables
			double dY;
			double dX;

			{
				// Calculate time of the day in UT decimal hours
				decHours = hour - timezone +
					(minute + second / 60.0 ) / 60.0;

				// Calculate current Julian Day
				int liAux1 = (month-14) / 12;
				int liAux2 = (1461*(year + 4800 + liAux1)) / 4
					+ (367 * (month - 2 - 12 * liAux1)) / 12
					- (3 * ((year + 4900 + liAux1) / 100)) / 4
					+ day - 32075;
				double dJulianDate = (double) liAux2 - 0.5 + decHours / 24.0;

				// Calculate difference between current Julian Day and JD 2451545.0
				elapsedJulianDays = dJulianDate - 2451545.0;
			}

			 /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
			   ecliptic in radians but without limiting the angle to be less than 2*Pi
			   (i.e., the result may be greater than 2*Pi) */
			{
				double omega = 2.1429 - 0.0010394594 * elapsedJulianDays;
				double meanLongitude = 4.8950630 + 0.017202791698 * elapsedJulianDays; // Radians
				double anomaly = 6.2400600 + 0.0172019699 * elapsedJulianDays;

				eclipticLongitude = meanLongitude + 0.03341607 * std::sin(anomaly)
					+ 0.00034894 * std::sin(2*anomaly) - 0.0001134
					- 0.0000203 * std::sin(omega);

				eclipticObliquity = 0.4090928 - 6.2140e-9 * elapsedJulianDays
					+ 0.0000396 * std::cos(omega);
			}

			/* Calculate celestial coordinates ( right ascension and declination ) in radians
			   but without limiting the angle to be less than 2*Pi (i.e., the result may be
			   greater than 2*Pi) */
			{
				double sinEclipticLongitude = std::sin(eclipticLongitude);
				dY = std::cos(eclipticObliquity) * sinEclipticLongitude;
				dX = std::cos(eclipticLongitude);
				rightAscension = std::atan2(dY, dX);
				if (rightAscension < 0.0)
					rightAscension += 2*PI;
				declination = std::asin(std::sin(eclipticObliquity) * sinEclipticLongitude);
			}

			// Calculate local coordinates (azimuth and zenith angle) in degrees
			{
				double greenwichMeanSiderealTime = 6.6974243242
					+ 0.0657098283 * elapsedJulianDays + decHours;

				auto degToRad = [](double f){return f * PI / 180.0f; };

				double localMeanSiderealTime = degToRad(((greenwichMeanSiderealTime * 15
					+ longitude)));

				double latitudeInRadians = degToRad(latitude);
				double cosLatitude = std::cos(latitudeInRadians);
				double sinLatitude = std::sin(latitudeInRadians);

				double hourAngle = localMeanSiderealTime - rightAscension;
				double cosHourAngle = std::cos(hourAngle);

				elevation = std::acos(cosLatitude * cosHourAngle
					* std::cos(declination) + std::sin(declination) * sinLatitude);

				dY = -std::sin(hourAngle);
				dX = std::tan(declination) * cosLatitude - sinLatitude * cosHourAngle;

				azimuth = std::atan2(dY, dX);
				if (azimuth < 0.0)
					azimuth += 2*PI;

				// Parallax Correction
				const float EARTH_MEAN_RADIUS = 6371.01f;
				const float ASTRONOMICAL_UNIT = 149597890.0f;
				elevation += (EARTH_MEAN_RADIUS / ASTRONOMICAL_UNIT) * std::sin(elevation);
			}

			float sinTheta, cosTheta, sinPhi, cosPhi;

			sincos((float)elevation, &sinTheta, &cosTheta);
			sincos((float)azimuth, &sinPhi, &cosPhi);

			dir = Vec3f(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		}

		float scale = S.def_storage.prop_float(node, "scale", 1.0f);
		float sunRadiusScale = S.def_storage.prop_float(node, "sunRadiusScale", 1.0f);

		if(node.has_property("toWorld"))
			dir = parseMatrix(node.get_property("toWorld"), S).TransformDirection(dir);

		//return add_to_scene(CreateAggregate<Light>(DistantLight(Spectrum(scale), dir.normalized(), sunRadiusScale)), S);
		auto scene_box = S.scene.getSceneBox();
		float scene_rad = scene_box.Size().length();
		CudaRNG rng((int)time(0));
		int N_lights = 8;
		const float EARTH_MEAN_RADIUS = 6371.01f;
		float rel_radius = scene_rad / EARTH_MEAN_RADIUS * sunRadiusScale * 10;
		StreamReference<Light> L;
		for (int i = 0; i < N_lights; i++)
		{
			auto p = scene_box.Center() - dir.normalized() * scene_rad / 2;
			p += Frame(dir.normalized()).toWorld(Vec3f(2 * rng.randomFloat() - 1, 2 * rng.randomFloat() - 1, 0)) * rel_radius;
			L = add_to_scene(CreateAggregate<Light>(SpotLight(p, scene_box.Center(), Spectrum(scale * scene_rad * scene_rad / N_lights), 90, 90)), S);
		}
		return L;
	}
public:
	StreamReference<Light> point(const XMLNode& node, ParserState& S)
	{
		auto emission = tryParseColor(node, S, "intensity", 1.0);

		Vec3f pos;
		if (node.has_property("position"))
			pos = parseVector(node.get_property("position"), S);
		else if(node.has_property("toWorld"))
			pos = parseMatrix(node.get_property("toWorld"), S).Translation();
		else throw std::runtime_error("no position specified for point light");

		return add_to_scene(CreateAggregate<Light>(PointLight(pos, emission)), S);
	}

	StreamReference<Light> spot(const XMLNode& node, ParserState& S)
	{
		auto emission = tryParseColor(node, S, "intensity", 1.0);
		float cutoffAngle = S.def_storage.prop_float(node, "cutoffAngle", 20);
		float beamWidth = S.def_storage.prop_float(node, "beamWidth", 20 * 0.75f);
		auto T = parseMatrix(node.get_property("toWorld"), S);

		return add_to_scene(CreateAggregate<Light>(SpotLight(T.Translation(), T.Translation() + T.Forward(), emission, cutoffAngle, beamWidth)), S);
	}

	StreamReference<Light> directional(const XMLNode& node, ParserState& S)
	{
		auto irradiance = tryParseColor(node, S, "irradiance", 1.0);

		Vec3f dir;
		if (node.has_property("direction"))
			dir = parseVector(node.get_property("direction"), S);
		else if (node.has_property("toWorld"))
			dir = parseMatrix(node.get_property("toWorld"), S).Forward();
		else throw std::runtime_error("no direction specified for directional light");

		return add_to_scene(CreateAggregate<Light>(DistantLight(irradiance, dir.normalized(), 1)), S);
	}

	StreamReference<Light> envmap(const XMLNode& node, ParserState& S)
	{
		float scale = S.def_storage.prop_float(node, "scale", 1.0f);
		auto T = node.has_property("toWorld") ? parseMatrix(node.get_property("toWorld"), S) : parseMatrix_Id(S);
		auto filename = S.def_storage.prop_string(node, "filename");

		auto Lref = S.scene.setEnvironementMap(scale, S.map_asset_filepath(filename));
		Lref->As<InfiniteLight>()->m_worldTransform = *(NormalizedT<OrthogonalAffineMap>*)&T;
		return Lref;
	}

	StreamReference<Light> sunsky(const XMLNode& node, ParserState& S)
	{
		return parseSun(node, S);
	}

	StreamReference<Light> sun(const XMLNode& node, ParserState& S)
	{
		return parseSun(node, S);
	}

	StreamReference<Light> sky(const XMLNode& node, ParserState& S)
	{
		throw std::runtime_error("NOT YET IMPLEMENTED");
	}

	StreamReference<Light> constant(const XMLNode& node, ParserState& S)
	{
		auto radiance = tryParseColor(node, S, "radiance", 1.0);

		//3 choices : 1. white environment map, 2. lots of directional lights, 3. giant sphere

		/*float r = 1;
		StreamReference<Light> L;
		float N = 8;
		for(float i = 0; i < N; i++)
			for (float j = 0; j < N; j++)
			{
				auto dir = Warp::squareToUniformSphere(Vec2f(i, j) / N);
				L = add_to_scene(CreateAggregate<Light>(DistantLight(radiance, dir, r)), S);
			}
		return L;*/

		auto obj = load_virtual("sphere", S, get_sphere_obj_text());
		float rad = S.scene.getSceneBox().Size().length();
		S.scene.SetNodeTransform(float4x4::Scale(Vec3f(-rad)), obj);
		if (!obj->m_uInstanciatedMaterial)//modifying the material -> instanciate material
			S.scene.instanciateNodeMaterials(obj);
		radiance = radiance / (4*PI*rad*rad);
		return S.scene.CreateLight(obj, S.scene.getMaterials(obj)(0)->Name, radiance);
	}

	static StreamReference<Light> parse(const XMLNode& node, ParserState& S)
	{
		START_PARSE(StreamReference<Light>, LightParser);
		CALL_PARSE_NO_UPDATE(point);
		CALL_PARSE_NO_UPDATE(spot);
		CALL_PARSE_NO_UPDATE(directional);
		CALL_PARSE_NO_UPDATE(envmap);
		CALL_PARSE_NO_UPDATE(sunsky);
		CALL_PARSE_NO_UPDATE(sun);
		CALL_PARSE_NO_UPDATE(sky);
		CALL_PARSE_NO_UPDATE(constant);
		END_PARSE();
	}
};

class BsdfParser
{
	struct BsdfData
	{
		BSDFALL bsdf;
		boost::optional<Texture> heightmap;
		boost::optional<Texture> alphamap;
		bool two_sided;
	};

	template<typename T> BsdfData create(const T& bsdf, std::vector<BsdfData>* others = 0)
	{
		BsdfData data;
		data.bsdf = CreateAggregate<BSDFALL>(bsdf);
		data.two_sided = false;

		if (others)
		{
			for (auto child_bsdf : *others)
			{
				data.two_sided |= child_bsdf.two_sided;
				if (!data.heightmap && child_bsdf.heightmap)
					data.heightmap = child_bsdf.heightmap.get();
				if (!data.alphamap && child_bsdf.alphamap)
					data.alphamap = child_bsdf.alphamap.get();
			}
		}

		return data;
	}

	void parseGenericRough(const XMLNode& node, ParserState& S, Texture& alphaU, Texture& alphaV, MicrofacetDistribution::EType& dist)
	{
		alphaU = CreateAggregate<Texture>(ConstantTexture(0.1f));
		alphaV = CreateAggregate<Texture>(ConstantTexture(0.1f));

		if (node.has_property("alpha"))
			alphaU = alphaV = parseTexturOrColor(node.get_property("alpha"), S);
		if (node.has_property("alphaU"))
			alphaU = parseTexturOrColor(node.get_property("alphaU"), S);
		if (node.has_property("alphaV"))
			alphaV = parseTexturOrColor(node.get_property("alphaV"), S);

		auto dist_s = S.def_storage.prop_string(node, "distribution", std::string("beckmann"));
		dist = dist_s == "beckmann" ? MicrofacetDistribution::EBeckmann : ("ggx" ? MicrofacetDistribution::EGGX : ("phong" ? MicrofacetDistribution::EPhong : (MicrofacetDistribution::EBeckmann)));
	}

	void parseGenericIOR(const XMLNode& node, ParserState& S, Texture& refl, Texture& trans, float& ior)
	{
		refl = CreateAggregate<Texture>(ConstantTexture(1.0f));
		trans = CreateAggregate<Texture>(ConstantTexture(1.0f));
		float intIOR = S.ior_lib.get("bk7"), extIOR = S.ior_lib.get("air");

		refl = tryParseTexturOrColor(node, S, "specularReflectance", 1.0f);
		trans = tryParseTexturOrColor(node, S, "specularTransmittance", 1.0f);

		if (node.has_property("intIOR"))
			intIOR = S.ior_lib.get(node.get_property("intIOR"), S.def_storage);
		if (node.has_property("extIOR"))
			extIOR = S.ior_lib.get(node.get_property("extIOR"), S.def_storage);

		ior = intIOR / extIOR;
	}

	void parseGenericEta(const XMLNode& node, ParserState& S, Spectrum& eta, Spectrum& k, Texture& refl)
	{
		eta = 0.0f;
		k = 0.0f;
		refl = CreateAggregate<Texture>(ConstantTexture(1.0f));
		float extEta = S.ior_lib.get("air");

		if (node.has_property("material") && S.def_storage.prop_string(node, "material") != "none")
		{
			auto load_sdp = [&](const std::string& file_path)
			{
				std::ifstream t(file_path);
				t.seekg(0, std::ios::end);
				std::string data = "";
				data.reserve(t.tellg());
				t.seekg(0, std::ios::beg);

				data.assign(std::istreambuf_iterator<char>(t),
							std::istreambuf_iterator<char>());

				std::vector<float> wls, vals;
				std::vector<std::string> lines = split_string(data, {"\r", "\n"});
				for (auto& line : lines)
				{
					if (line.size() == 0 || line[0] == '#')
						continue;
					auto parts = split_string_array(line);
					wls.push_back(std::stof(parts[0]));
					vals.push_back(std::stof(parts[1]));
				}
				Spectrum C;
				C.fromContinuousSpectrum(&wls[0], &vals[0], (int)wls.size());
				return C;
			};

			auto mat_name = S.def_storage.prop_string(node, "material");
			eta = load_sdp(S.map_spd_filepath(mat_name + ".eta.spd"));
			k = load_sdp(S.map_spd_filepath(mat_name + ".k.spd"));
		}

		eta = tryParseColor(node, S, "eta", eta);
		k = tryParseColor(node, S, "k", k);

		extEta = S.def_storage.prop_float(node, "extEta", extEta);

		eta /= extEta;
		k /= extEta;
	}

	void parseGenericCoating(const XMLNode& node, ParserState& S, float& thickness, Texture& sigmaA)
	{
		thickness = S.def_storage.prop_float(node, "thickness", 1.0f);
		sigmaA = tryParseTexturOrColor(node, S, "sigmaA", 0.0f);
	}

	void parseGenericPlastic(const XMLNode& node, ParserState& S, Texture& specRefl, Texture& diffRefl, bool& nonlinear)
	{
		nonlinear = S.def_storage.prop_bool(node, "nonlinear", false);
		specRefl = tryParseTexturOrColor(node, S, "specularReflectance", 1.0f);
		diffRefl = tryParseTexturOrColor(node, S, "diffuseReflectance", 0.5f);
	}

	std::vector<BsdfData> parseAllNested(const XMLNode& node, ParserState& S)
	{
		std::vector<BsdfData> child_bsdfs;
		node.iterate_child_nodes([&](const XMLNode& child)
		{
			if (child.name() == "ref" || child.name() == "bsdf")
				child_bsdfs.push_back(parse(child, S, m_recursive_depth + 1));
		});
		return child_bsdfs;
	}
	int m_recursive_depth;
	//used by bsdfs such as heightmap which are not represented as bsdfs
	void dec_depth()
	{
		m_recursive_depth--;
	}
	bool is_max_depth()
	{
		return m_recursive_depth == 1;
	}
	BSDFFirst convert_bsdf(const BsdfData& bsdf)
	{
		BSDFFirst f;
		f.setTypeToken(bsdf.bsdf.getTypeToken());
		memcpy(f.As(), bsdf.bsdf.As(), f.DATA_SIZE);
		return f;
	}
public:
	BsdfData diffuse(const XMLNode& node, ParserState& S)
	{
		auto refl = tryParseTexturOrColor(node, S, "reflectance", 0.5f);
		return create(CudaTracerLib::diffuse(refl));
	}

	BsdfData roughdiffuse(const XMLNode& node, ParserState& S)
	{
		auto refl = tryParseTexturOrColor(node, S, "reflectance", 0.5f);
		auto alpha = tryParseTexturOrColor(node, S, "reflectance", 0.2f);
		return create(CudaTracerLib::roughdiffuse(refl, alpha));
	}

	BsdfData dielectric(const XMLNode& node, ParserState& S)
	{
		Texture refl, trans;
		float ior;
		parseGenericIOR(node, S, refl, trans, ior);
		return create(CudaTracerLib::dielectric(ior, refl, trans));
	}

	BsdfData thindielectric(const XMLNode& node, ParserState& S)
	{
		Texture refl, trans;
		float ior;
		parseGenericIOR(node, S, refl, trans, ior);
		return create(CudaTracerLib::thindielectric(ior, refl, trans));
	}

	BsdfData roughdielectric(const XMLNode& node, ParserState& S)
	{
		Texture refl, trans, alphaU, alphaV;
		float ior;
		MicrofacetDistribution::EType dist;
		parseGenericIOR(node, S, refl, trans, ior);
		parseGenericRough(node, S, alphaU, alphaV, dist);
		return create(CudaTracerLib::roughdielectric(dist, ior, alphaU, alphaV, refl, trans));
	}

	BsdfData conductor(const XMLNode& node, ParserState& S)
	{
		Texture refl;
		Spectrum eta, k;
		parseGenericEta(node, S, eta, k, refl);
		return create(CudaTracerLib::conductor(eta, k, refl));
	}

	BsdfData roughconductor(const XMLNode& node, ParserState& S)
	{
		Texture refl, alphaU, alphaV;
		Spectrum eta, k;
		MicrofacetDistribution::EType dist;
		parseGenericEta(node, S, eta, k, refl);
		parseGenericRough(node, S, alphaU, alphaV, dist);
		return create(CudaTracerLib::roughconductor(dist, eta, k, alphaU, alphaV, refl));
	}

	BsdfData plastic(const XMLNode& node, ParserState& S)
	{
		Texture _1, _2, diff, spec;
		float ior;
		bool nonlinear;
		parseGenericIOR(node, S, _1, _2, ior);
		parseGenericPlastic(node, S, spec, diff, nonlinear);
		return create(CudaTracerLib::plastic(ior, diff, spec, nonlinear));
	}

	BsdfData roughplastic(const XMLNode& node, ParserState& S)
	{
		Texture _1, _2, diff, spec, alphaU, alphaV;
		float ior;
		bool nonlinear;
		MicrofacetDistribution::EType dist;
		parseGenericIOR(node, S, _1, _2, ior);
		parseGenericPlastic(node, S, spec, diff, nonlinear);
		parseGenericRough(node, S, alphaU, alphaV, dist);
		return create(CudaTracerLib::plastic(ior, diff, spec, nonlinear));
		//return create(CudaTracerLib::ward(ward::EModelVariant::EWard, diff, CreateAggregate<Texture>(ConstantTexture(0.2f)), alphaU, alphaV));
		//return create(CudaTracerLib::roughplastic(dist, ior, alphaU, diff, spec, nonlinear));
	}

	BsdfData coating(const XMLNode& node, ParserState& S)
	{
		auto nested = parseAllNested(node, S);
		if (nested.size() != 1)
			throw std::runtime_error("expected 1 nested bsdf in coating!");
		if (is_max_depth())
			return nested[0];

		Texture refl, trans, sigmaA;
		float ior, thickness;
		parseGenericIOR(node, S, refl, trans, ior);
		parseGenericCoating(node, S, thickness, sigmaA);

		return create(CudaTracerLib::coating(convert_bsdf(nested[0]), ior, thickness, sigmaA, refl), &nested);
	}

	BsdfData roughcoating(const XMLNode& node, ParserState& S)
	{
		auto nested = parseAllNested(node, S);
		if (nested.size() != 1)
			throw std::runtime_error("expected 1 nested bsdf in coating!");
		if (is_max_depth())
			return nested[0];

		Texture refl, trans, sigmaA, alphaU, alphaV;
		float ior, thickness;
		MicrofacetDistribution::EType dist;
		parseGenericIOR(node, S, refl, trans, ior);
		parseGenericCoating(node, S, thickness, sigmaA);
		parseGenericRough(node, S, alphaU, alphaV, dist);

		return create(CudaTracerLib::roughcoating(dist, convert_bsdf(nested[0]), ior, thickness, sigmaA, alphaU, refl), &nested);
	}

	BsdfData bumpmap(const XMLNode& node, ParserState& S)
	{
		dec_depth();
		auto nested = parseAllNested(node, S);
		if (nested.size() != 1)
			throw std::runtime_error("expected 1 nested bsdf in bumpmap!");

		auto child = nested[0];
		child.heightmap = tryParseTexturOrColor(node, S, "texture", 0.0f);
		return child;
	}

	BsdfData phong(const XMLNode& node, ParserState& S)
	{
		auto exponent = tryParseTexturOrColor(node, S, "exponent", 30.0f);
		auto specularReflectance = tryParseTexturOrColor(node, S, "specularReflectance", 0.2f);
		auto diffuseReflectance = tryParseTexturOrColor(node, S, "diffuseReflectance", 0.5f);

		return create(CudaTracerLib::phong(diffuseReflectance, specularReflectance, exponent));
	}

	BsdfData ward(const XMLNode& node, ParserState& S)
	{
		auto specularReflectance = tryParseTexturOrColor(node, S, "specularReflectance", 0.2f);
		auto diffuseReflectance = tryParseTexturOrColor(node, S, "diffuseReflectance", 0.5f);
		auto alphaU = tryParseTexturOrColor(node, S, "alphaU", 0.1f);
		auto alphaV = tryParseTexturOrColor(node, S, "alphaV", 0.1f);
		auto dist_s = S.def_storage.prop_string(node, "variant", std::string("balanced"));
		auto dist = dist_s == "ward" ? ward::EModelVariant::EWard : (dist_s == "ward-duer" ? ward::EModelVariant::EWardDuer : (ward::EModelVariant::EBalanced));

		return create(CudaTracerLib::ward(dist, diffuseReflectance, specularReflectance, alphaU, alphaV));
	}

	BsdfData mixturebsdf(const XMLNode& node, ParserState& S)
	{
		auto nested = parseAllNested(node, S);
		if (nested.size() != 2)
			throw std::runtime_error("expected 2 nested bsdf in mixturebsdf (LIMITATION)!");
		if (is_max_depth())
			return nested[0];

		auto weights_s = S.def_storage.prop_string(node, "weights");
		auto weights = split_string_array(weights_s);
		if (weights.size() != 2)
			throw std::runtime_error("not able to get 2 weights from weights property");

		float w1 = std::stof(weights[0]), w2 = std::stof(weights[1]);
		float alpha = w1 / (w1 + w2);
		return create(CudaTracerLib::blend(convert_bsdf(nested[0]), convert_bsdf(nested[1]), CreateAggregate<Texture>(ConstantTexture(alpha))), &nested);
	}

	BsdfData blendbsdf(const XMLNode& node, ParserState& S)
	{
		auto nested = parseAllNested(node, S);
		if (nested.size() != 2)
			throw std::runtime_error("expected 2 nested bsdf in blendbsdf!");
		if (is_max_depth())
			return nested[0];

		auto alpha = tryParseTexturOrColor(node, S, "weight", 0.5f);
		return create(CudaTracerLib::blend(convert_bsdf(nested[0]), convert_bsdf(nested[1]), alpha), &nested);
	}

	BsdfData mask(const XMLNode& node, ParserState& S)
	{
		dec_depth();
		auto nested = parseAllNested(node, S);
		if (nested.size() != 1)
			throw std::runtime_error("expected 1 nested bsdf in mask!");

		auto child = nested[0];
		child.alphamap = tryParseTexturOrColor(node, S, "opacity", 0.0f);

		if (child.bsdf.Is<CudaTracerLib::diffuse>())
		{
			auto r = create(CudaTracerLib::diffuse(child.alphamap.get()), &nested);
			r.bsdf.As<CudaTracerLib::diffuse>()->setModus(EDiffuseTransmission);
			return r;
		}
		else return child;
	}

	BsdfData difftrans(const XMLNode& node, ParserState& S)
	{
		auto refl = tryParseTexturOrColor(node, S, "reflectance", 0.5f);
		auto b = CudaTracerLib::diffuse(refl);
		b.setModus(EDiffuseTransmission);
		return create(b);
	}

	BsdfData twosided(const XMLNode& node, ParserState& S)
	{
		dec_depth();
		auto nested = parseAllNested(node, S);
		if (nested.size() != 1)
			throw std::runtime_error("expected 1 nested bsdf in twosided!");

		nested[0].two_sided = true;
		return nested[0];
	}

	static BsdfData parse(const XMLNode& node, ParserState& S, int recursive_depth)
	{
		START_PARSE(BsdfData, BsdfParser);
		P.m_recursive_depth = recursive_depth;
		CALL_PARSE_NO_UPDATE(diffuse);
		CALL_PARSE_NO_UPDATE(roughdiffuse);
		CALL_PARSE_NO_UPDATE(dielectric);
		CALL_PARSE_NO_UPDATE(thindielectric);
		CALL_PARSE_NO_UPDATE(roughdielectric);
		CALL_PARSE_NO_UPDATE(conductor);
		CALL_PARSE_NO_UPDATE(roughconductor);
		CALL_PARSE_NO_UPDATE(plastic);
		CALL_PARSE_NO_UPDATE(roughplastic);
		CALL_PARSE_NO_UPDATE(coating);
		CALL_PARSE_NO_UPDATE(roughcoating);
		CALL_PARSE_NO_UPDATE(bumpmap);
		CALL_PARSE_NO_UPDATE(phong);
		CALL_PARSE_NO_UPDATE(ward);
		CALL_PARSE_NO_UPDATE(mixturebsdf);
		CALL_PARSE_NO_UPDATE(blendbsdf);
		CALL_PARSE_NO_UPDATE(mask);
		CALL_PARSE_NO_UPDATE(difftrans);
		CALL_PARSE_NO_UPDATE(twosided);
		END_PARSE();
	}

	//either parses a new bsdf or retrieves a stored one
	//then applies it to the first material
	static void apply_bsdf(const XMLNode& node, ParserState& S, StreamReference<Node> obj)
	{
		auto data = parse(node, S, 0);
		if (!obj->m_uInstanciatedMaterial)
			S.scene.instanciateNodeMaterials(obj);
		auto mat = S.scene.getMaterials(obj);
		mat->bsdf = data.bsdf;
		mat->bsdf.As()->m_enableTwoSided = data.two_sided;
		if (data.heightmap)
			mat->SetHeightMap(data.heightmap.get());
		if (data.alphamap)
			mat->SetAlphaMap(data.alphamap.get(), AlphaBlendState::AlphaMap_Luminance);
		mat.Invalidate();
	}
};

class ShapeParser
{
	StreamReference<Node> parseFiles(const std::string& name, ParserState& S)
	{
		auto filename = S.map_asset_filepath(name);
		auto token = S.get_scene_name() + "/" + name;

		auto xmsh_path = S.scene.getFileManager()->getCompiledMeshPath(token);
		auto xmsh_folder = boost::filesystem::path(xmsh_path).parent_path();

		if(!boost::filesystem::exists(xmsh_folder))
			boost::filesystem::create_directories(xmsh_folder);

		auto* str = OpenFile(filename);
		auto obj = S.scene.CreateNode(token, *str);
		delete str;
		return obj;
	}

	void parseGeneric(StreamReference<Node> obj, const XMLNode& node, ParserState& S, const float4x4& local = float4x4::Identity(), bool is_in_coord = false)
	{
		auto T = node.has_property("toWorld") ? parseMatrix(node.get_property("toWorld"), S) : (is_in_coord ? float4x4::Identity() : parseMatrix_Id(S));
		S.scene.SetNodeTransform(T % local, obj);

		if (node.has_child_node("emitter"))
		{
			auto em_node = node.get_child_node("emitter");
			if (em_node.get_attribute("type") != "area")
				throw std::runtime_error("only supports area light sources on meshes, not : " + em_node.get_attribute("type"));
			if (!obj->m_uInstanciatedMaterial)//modifying the material -> instanciate material
				S.scene.instanciateNodeMaterials(obj);
			auto emission = parseColor(em_node.get_property("radiance"), S);
			S.scene.CreateLight(obj, S.scene.getMaterials(obj)(0)->Name, emission);
		}

		bool has_bsdf = false;

		if(node.has_child_node("bsdf"))
		{
			BsdfParser::apply_bsdf(node.get_child_node("bsdf"), S, obj);
			has_bsdf = true;
		}

		node.iterate_child_nodes([&](const XMLNode& child_node)
		{
			if(child_node.name() != "ref")
				return;

			if(!child_node.has_attribute("name") || child_node.get_attribute("name") == "bsdf")
			{
				BsdfParser::apply_bsdf(child_node, S, obj);
				has_bsdf = true;
			}
		});

		//auto local_aabb = S.scene.getMesh(obj)->m_sLocalBox;
		//auto obj_mat = float4x4::Translate(-Vec3f(1e10f / 2))  % float4x4::Scale(Vec3f(1e10f));//S.scene.GetNodeTransform(obj) % float4x4::Translate(local_aabb.minV) % float4x4::Scale(max(local_aabb.Size(), Vec3f(10,10,10)));
		auto scene_box = S.scene.getSceneBox();
		auto obj_mat = float4x4::Translate(scene_box.minV) % float4x4::Scale(scene_box.Size());

		bool delete_node = false;
		auto create_bssrdf = [&](const std::string bssrdf_xml_node)
		{
			if (node.has_property(bssrdf_xml_node))
			{
				if (has_bsdf)
				{
					if (!obj->m_uInstanciatedMaterial)
						S.scene.instanciateNodeMaterials(obj);
					auto mat = S.scene.getMaterials(obj)(0);
					mat->usedBssrdf = true;
					mat->bssrdf = MediumParser::parse(node.get_property(bssrdf_xml_node), S, obj_mat);

					mat.Invalidate();
				}
				else
				{
					S.scene.CreateVolume(MediumParser::parse(node.get_property(bssrdf_xml_node), S, obj_mat));
					delete_node = true;
				}
			}
		};

		if(S.create_exterior_bssrdf)
			create_bssrdf("exterior");
		if(S.create_interior_bssrdf)
			create_bssrdf("interior");
		if (delete_node)
			S.nodes_to_remove.push_back(obj);
	}
public:
	class ShapegroupData
	{
		std::vector<std::tuple<float4x4, StreamReference<Node>>> nodes;
		int num_instanciations = 0;
	public:
		void add(StreamReference<Node> obj, ParserState& S)
		{
			nodes.push_back(std::make_tuple(S.scene.GetNodeTransform(obj), obj));
		}

		void instanciate(const float4x4& m, ParserState& S)
		{
			for(auto& el : nodes)
			{
				auto mat = m % parseMatrix_Id(S).inverse() % std::get<0>(el);
				if (num_instanciations == 0)
					S.scene.SetNodeTransform(mat, std::get<1>(el));
				else assign(std::get<1>(el), mat, S);
			}

			num_instanciations++;
		}

		static void assign(StreamReference<Node> node_src, const float4x4& m, ParserState& S)
		{
			auto m_ref = S.scene.getMesh(node_src);
			std::string mesh_path = m_ref->m_uPath.c_str();
			auto compiled_folder = S.scene.getFileManager()->getCompiledMeshPath("");
			mesh_path = compiled_folder + "/" + mesh_path;

			auto node_tar = S.scene.CreateNode(mesh_path);
			S.scene.SetNodeTransform(m, node_tar);
			auto mat_tar = S.scene.getMaterials(node_tar), mat_src = S.scene.getMaterials(node_src);
			mat_tar->bsdf = mat_src->bsdf;
			mat_tar->AlphaMap = mat_src->AlphaMap;
			mat_tar->HeightMap = mat_src->HeightMap;
		}
	};

	struct ShapeParseResult
	{
		int type;// 1 => node, 2 => group, 3 => nothing
		StreamReference<Node> obj;
		ShapegroupData group;

		explicit ShapeParseResult()
			: type(3)
		{

		}

		ShapeParseResult(StreamReference<Node> obj)
			: obj(obj), type(1)
		{
		}

		ShapeParseResult(const ShapegroupData& group)
			: group(group), type(2)
		{
		}

		bool is_node() const { return type == 1; }
		bool is_group() const { return type == 2; }
	};

	ShapeParseResult serialized(const XMLNode& node, ParserState& S);

	ShapeParseResult obj(const XMLNode& node, ParserState& S)
	{
		auto name = S.def_storage.prop_string(node, "filename");

		auto obj = parseFiles(name, S);
		parseGeneric(obj, node, S);
		return obj;
	}

	ShapeParseResult ply(const XMLNode& node, ParserState& S)
	{
		auto name = S.def_storage.prop_string(node, "filename");

		auto obj = parseFiles(name, S);
		parseGeneric(obj, node, S);
		return obj;
	}

	ShapeParseResult rectangle(const XMLNode& node, ParserState& S)
	{
		bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);

		auto obj = load_virtual("cube", S, get_cube_obj_text());
		float s = flipNormals ? -2.0f : 2.0f;
		parseGeneric(obj, node, S, float4x4::Scale(Vec3f(s,s,s*1e-3f)) % float4x4::Translate(Vec3f(-0.5f)));
		return obj;
	}

	ShapeParseResult sphere(const XMLNode& node, ParserState& S)
	{
		bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);
		float radius = S.def_storage.prop_float(node, "radius", 1.0f);
		Vec3f pos = node.has_property("center") ? parseVector(node.get_property("center"), S) : Vec3f(0.0f);

		auto obj = load_virtual("sphere", S, get_sphere_obj_text());
		float r = (flipNormals ? -1.0f : 1.0f) * radius;
		parseGeneric(obj, node, S, float4x4::Translate(pos) % float4x4::Scale(Vec3f(r)), true);
		return obj;
	}

	ShapeParseResult cube(const XMLNode& node, ParserState& S)
	{
		bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);

		auto obj = load_virtual("cube", S, get_cube_obj_text());
		parseGeneric(obj, node, S, float4x4::Scale(Vec3f(flipNormals ? -2.0f : 2.0f)) % float4x4::Translate(Vec3f(-0.5f)));
		return obj;
	}

	ShapeParseResult cylinder(const XMLNode& node, ParserState& S)
	{
		bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);
		float radius = S.def_storage.prop_float(node, "radius", 1.0f);
		Vec3f p0 = node.has_property("p0") ? parseVector(node.get_property("p0"), S) : Vec3f(0, 0, 0);
		Vec3f p1 = node.has_property("p1") ? parseVector(node.get_property("p1"), S) : Vec3f(0, 0, 1);

		auto obj = load_virtual("cylinder", S, get_cylinder_obj_text());
		float r = (flipNormals ? -1.0f : 1.0f) * radius;
		parseGeneric(obj, node, S, float4x4::Translate(p0) % Frame((p1-p0).normalized()).ToWorldMatrix() % float4x4::Scale(Vec3f(r,r,(p1-p0).length()/2)), true);
		return obj;
	}

	ShapeParseResult disk(const XMLNode& node, ParserState& S)
	{
		bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);

		auto obj = load_virtual("disk", S, get_disk_obj_text());
		parseGeneric(obj, node, S, float4x4::Scale(Vec3f(flipNormals ? -1.0f : 1.0f)));
		return obj;
	}

	ShapeParseResult shapegroup(const XMLNode& node, ParserState& S)
	{
		ShapegroupData data;
		node.iterate_child_nodes([&](const XMLNode& child_node)
		{
			if(child_node.name() == "shape")
			{
				auto obj = ShapeParser::parse(child_node, S);
				if(!obj.is_node())
					throw std::runtime_error("invalid xml parsed, expected node");
				data.add(obj.obj, S);
			}
			else throw std::runtime_error("invalid node in shapegroup : " + child_node.name());
		});
		return data;
	}

	ShapeParseResult instance(const XMLNode& node, ParserState& S)
	{
		auto T = node.has_property("toWorld") ? parseMatrix(node.get_property("toWorld"), S) : parseMatrix_Id(S);
		auto ref_s = S.def_storage.as_string(node.get_child_node("ref"), "id");
		ShapeParseResult* ref_group = S.ref_storage.get<ShapeParseResult>(ref_s);
		if(ref_group->is_group())
			ref_group->group.instanciate(T, S);
		else if(ref_group->is_node())
		{
			std::cout << "node : " << ref_s << std::endl;
			ShapegroupData::assign(ref_group->obj, T, S);
		}
		else throw std::runtime_error("invalid ref type : " + std::to_string(ref_group->type));
		return ShapeParseResult();
	}

	ShapeParseResult hair(const XMLNode& node, ParserState& S)
	{
		std::cout << "hair model is not implemented" << std::endl;
		return ShapeParseResult();
	}

	static ShapeParseResult parse(const XMLNode& node, ParserState& S)
	{
		START_PARSE(ShapeParseResult, ShapeParser);
		CALL_PARSE_NO_UPDATE(serialized);
		CALL_PARSE_NO_UPDATE(obj);
		CALL_PARSE_NO_UPDATE(ply);
		CALL_PARSE_NO_UPDATE(rectangle);
		CALL_PARSE_NO_UPDATE(sphere);
		CALL_PARSE_NO_UPDATE(cube);
		CALL_PARSE_NO_UPDATE(cylinder);
		CALL_PARSE_NO_UPDATE(disk);
		CALL_PARSE_NO_UPDATE(shapegroup);
		CALL_PARSE_NO_UPDATE(instance);
		CALL_PARSE_NO_UPDATE(hair);
		END_PARSE();
	}
};

}