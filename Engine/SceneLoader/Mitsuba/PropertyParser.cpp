#include <StdAfx.h>
#include "PropertyParser.h"

namespace CudaTracerLib {

Vec3f parseVector(const XMLNode& node, ParserState& S)
{
	if (S.ref_storage.is_ref(node))
		return *S.ref_storage.get<Vec3f>(node);
	Vec3f v = Vec3f(S.def_storage.as_float(node, "x"), S.def_storage.as_float(node, "y"), S.def_storage.as_float(node, "z"));
	v = S.id_matrix.TransformDirection(v);
	S.ref_storage.add_if_ref(node, new Vec3f(v));
	return v;
}

float4x4 parseMatrix(const XMLNode& node, ParserState& S, bool apply_id_rot)
{
	auto translate = [&](const XMLNode& N)
	{
		return float4x4::Translate(S.def_storage.as_float(N, "x", 0.0f), S.def_storage.as_float(N, "y", 0.0f), S.def_storage.as_float(N, "z", 0.0f));
	};
	auto rotate = [&](const XMLNode& N)
	{
		auto dir = NormalizedT<Vec3f>(S.def_storage.as_float(N, "x", 0.0f), S.def_storage.as_float(N, "y", 0.0f), S.def_storage.as_float(N, "z", 0.0f));
		return float4x4::RotationAxis(dir, S.def_storage.as_float(N, "angle"));
	};
	auto scale = [&](const XMLNode& N)
	{
		if (N.has_attribute("value"))
			return float4x4::Scale(Vec3f(S.def_storage.as_float(N, "value")));
		return float4x4::Scale(S.def_storage.as_float(N, "x", 1.0f), S.def_storage.as_float(N, "y", 1.0f), S.def_storage.as_float(N, "z", 1.0f));
	};
	auto matrix = [&](const XMLNode& N)
	{
		std::vector<std::string> strs = split_string_array(S.def_storage.as_string(N, "value"));
		float4x4 M;
		for (int i = 0; i < 16; i++)
			M(i / 4, i % 4) = std::stof(strs[i]);
		return M;
	};
	auto lookat = [&](const XMLNode& N)
	{
		auto conv = [](const std::string s)
		{
			std::vector<std::string> strs = split_string_array(s);
			if (strs.size() != 3)
				throw std::runtime_error("invalid vector");
			return Vec3f(std::stof(strs[0]), std::stof(strs[1]), std::stof(strs[2]));
		};
		auto origin = conv(S.def_storage.as_string(N, "origin"));
		auto target = conv(S.def_storage.as_string(N, "target"));
		auto up = conv(S.def_storage.as_string(N, "up"));
		return float4x4::lookAt(origin, target, up);
	};

	if (S.ref_storage.is_ref(node))
		return *S.ref_storage.get<float4x4>(node);

	float4x4 T = float4x4::Identity();
	node.iterate_child_nodes([&](const XMLNode& t_node)
	{
		float4x4 I = float4x4::Identity();
		auto N = t_node.name();
		if (N == "translate")
			I = translate(t_node);
		else if (N == "rotate")
			I = rotate(t_node);
		else if (N == "scale")
			I = scale(t_node);
		else if (N == "matrix")
			I = matrix(t_node);
		else if (N == "lookat")
			I = lookat(t_node);
		else throw std::runtime_error("invalid matrix operation : " + N);

		T = I % T;
	});
	if(apply_id_rot)
		T = parseMatrix_Id(S) % T;//in case of rotated coordinate system
	S.ref_storage.add_if_ref(node, new float4x4(T));
	return T;
}

Spectrum parseRGB(const XMLNode& node, ParserState& S, bool srgb)
{
	if (S.ref_storage.is_ref(node))
		return *S.ref_storage.get<Spectrum>(node);

	auto intent = node.has_attribute("intent") ? S.def_storage.as_string(node, "intent") : "reflectance";
	auto intent_e = intent == std::string("reflectance") ? Spectrum::EConversionIntent::EReflectance : Spectrum::EConversionIntent::EIlluminant;
	auto s = S.def_storage.as_string(node, "value");

	Spectrum C;
	if (srgb && s[0] == '#')
	{
		auto hexValue = std::stoi(s.substr(1, s.size() - 1));
		C = Spectrum(((hexValue >> 16) & 0xFF) / 255.0f, ((hexValue >> 8) & 0xFF) / 255.0f, ((hexValue) & 0xFF) / 255.0f);
	}
	else
	{
		std::vector<std::string> strs = split_string_array(s);
        if (strs.size() == 1)
            strs = { strs[0] , strs[0] , strs[0] };
		if (srgb)
			C.fromSRGB(std::stof(strs[0]), std::stof(strs[1]), std::stof(strs[2]));
		else C.fromLinearRGB(std::stof(strs[0]), std::stof(strs[1]), std::stof(strs[2]), intent_e);
	}

	S.ref_storage.add_if_ref(node, new Spectrum(C));
	return C;
}

Spectrum parseSpectrum(const XMLNode& node, ParserState& S)
{
	if (S.ref_storage.is_ref(node))
		return *S.ref_storage.get<Spectrum>(node);
	Spectrum C = Spectrum(0.0);

	auto intent = node.has_attribute("intent") ? S.def_storage.as_string(node, "intent") : "reflectance";
	auto intent_e = intent == "reflectence" ? Spectrum::EConversionIntent::EReflectance : Spectrum::EConversionIntent::EIlluminant;
	std::string data;
	if (node.has_attribute("filename"))
	{
		auto file_path = S.map_spd_filepath(S.def_storage.as_string(node, "filename"));
		std::ifstream t(file_path);
		t.seekg(0, std::ios::end);
		data.reserve(t.tellg());
		t.seekg(0, std::ios::beg);

		data.assign(std::istreambuf_iterator<char>(t),
			std::istreambuf_iterator<char>());
	}
	else data = S.def_storage.as_string(node, "value");

	std::vector<std::string> strs = split_string_array(data);
	if (data.find_first_of(":") != size_t(-1))
	{
		std::vector<float> wls, vals;
		for (std::string p : strs)
		{
			auto parts = split_string(p, { ":" });
			wls.push_back(std::stof(parts[0]));
			vals.push_back(std::stof(parts[1]));
		}
		C.fromContinuousSpectrum(&wls[0], &vals[0], (int)wls.size());
	}
	else if (strs.size() == 3)
		C.fromLinearRGB(std::stof(strs[0]), std::stof(strs[1]), std::stof(strs[2]), intent_e);
	else if (node.has_attribute("filename") && S.def_storage.as_string(node, "filename").find(".spd"))
	{
		std::vector<float> wls, vals;
		std::vector<std::string> lines = split_string(data, { "\r", "\n" });
		for (auto& line : lines)
		{
			if (line.size() == 0 || line[0] == '#')
				continue;
			auto parts = split_string_array(line);
			wls.push_back(std::stof(parts[0]));
			vals.push_back(std::stof(parts[1]));
		}
		C.fromContinuousSpectrum(&wls[0], &vals[0], (int)wls.size());
	}
	else
	{
		try
		{
			C = Spectrum(std::stof(data));
		}
		catch (...)
		{
			throw std::runtime_error("invalid spectrum data");
		}
	}

	S.ref_storage.add_if_ref(node, new Spectrum(C));
	return C;
}

}