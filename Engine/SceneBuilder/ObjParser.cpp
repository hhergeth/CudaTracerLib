#include <StdAfx.h>
#include "..\e_Mesh.h"
#include "TangentSpaceHelper.h"
#include <Engine\SceneBuilder\Importer.h>
#include <boost\algorithm\string.hpp>
#include <boost\filesystem.hpp>

bool parseSpace(const char*& ptr)
{
	while (*ptr == ' ' || *ptr == '\t')
		ptr++;
	return true;
}

bool parseChar(const char*& ptr, char chr)
{
	if (*ptr != chr)
		return false;
	ptr++;
	return true;
}

bool parseLiteral(const char*& ptr, const char* str)
{
	const char* tmp = ptr;

	while (*str && *tmp == *str)
	{
		tmp++;
		str++;
	}
	if (*str)
		return false;

	ptr = tmp;
	return true;
}

bool parseInt(const char*& ptr, int& value)
{
	const char* tmp = ptr;
	int v = 0;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
	if (*tmp < '0' || *tmp > '9')
		return false;
	while (*tmp >= '0' && *tmp <= '9')
		v = v * 10 + *tmp++ - '0';

	value = (neg) ? -v : v;
	ptr = tmp;
	return true;
}

bool parseInt(const char*& ptr, long long& value)
{
	const char* tmp = ptr;
	long long v = 0;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
	if (*tmp < '0' || *tmp > '9')
		return false;
	while (*tmp >= '0' && *tmp <= '9')
		v = v * 10 + *tmp++ - '0';

	value = (neg) ? -v : v;
	ptr = tmp;
	return true;
}

bool parseHex(const char*& ptr, unsigned int& value)
{
	const char* tmp = ptr;
	unsigned int v = 0;
	for (;;)
	{
		if (*tmp >= '0' && *tmp <= '9')         v = v * 16 + *tmp++ - '0';
		else if (*tmp >= 'A' && *tmp <= 'F')    v = v * 16 + *tmp++ - 'A' + 10;
		else if (*tmp >= 'a' && *tmp <= 'f')    v = v * 16 + *tmp++ - 'a' + 10;
		else                                    break;
	}

	if (tmp == ptr)
		return false;

	value = v;
	ptr = tmp;
	return true;
}

bool parseFloat(const char*& ptr, float& value)
{
#define bitsToFloat(x) (*(float*)&x)
	const char* tmp = ptr;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));

	float v = 0.0f;
	int numDigits = 0;
	while (*tmp >= '0' && *tmp <= '9')
	{
		v = v * 10.0f + (float)(*tmp++ - '0');
		numDigits++;
	}
	if (parseChar(tmp, '.'))
	{
		float scale = 1.0f;
		while (*tmp >= '0' && *tmp <= '9')
		{
			scale *= 0.1f;
			v += scale * (float)(*tmp++ - '0');
			numDigits++;
		}
	}
	if (!numDigits)
		return false;

	ptr = tmp;
	if (*ptr == '#')
	{
		unsigned int v = 0;
		if (parseLiteral(ptr, "#INF"))
			v = 0x7F800000;
		else if (parseLiteral(ptr, "#SNAN"))
			v = 0xFF800001;
		else if (parseLiteral(ptr, "#QNAN"))
			v = 0xFFC00001;
		else if (parseLiteral(ptr, "#IND"))
			v = 0xFFC00000;
		if (v)
		{
			v |= neg << 31;
			value = *(float*)&v;
			return true;
		}
		else return false;
	}

	int e = 0;
	if ((parseChar(tmp, 'e') || parseChar(tmp, 'E')) && parseInt(tmp, e))
	{
		ptr = tmp;
		if (e)
			v *= pow(10.0f, (float)e);
	}
	value = (neg) ? -v : v;
	return true;
#undef bitsToFloat
}

enum TextureType
{
	TextureType_Diffuse = 0,    // Diffuse color map.
	TextureType_Alpha,          // Alpha map (green = opacity).
	TextureType_Displacement,   // Displacement map (green = height).
	TextureType_Normal,         // Tangent-space normal map.
	TextureType_Environment,    // Environment map (spherical coordinates).

	TextureType_Max
};

struct Material
{
	std::string		Name;
	int				IlluminationModel;
	Vec4f          diffuse;
	Vec3f          specular;
	Vec3f			emission;
	float           glossiness;
	float           displacementCoef; // height = texture/255 * coef + bias
	float           displacementBias;
	float			IndexOfRefraction;
	Vec3f			Tf;
	std::string     textures[TextureType_Max];
	//int				SubMesh;

	Material(void)
	{
		diffuse = Vec4f(0.75f, 0.75f, 0.75f, 1.0f);
		specular = Vec3f(0.5f);
		glossiness = 32.0f;
		displacementCoef = 1.0f;
		displacementBias = 0.0f;
		emission = Vec3f(0, 0, 0);
	}
};

struct TextureSpec
{
	std::string	              texture;
	float                     base;
	float                     gain;
};

struct MatHash
{
	std::vector<Material> vec;
	std::map<std::string, int> map;

	void add(const std::string& name, const Material& mat)
	{
		map[name] = (int)vec.size();
		vec.push_back(mat);
	}

	bool contains(const std::string& name)
	{
		return map.find(name) != map.end();
	}

	int searchi(const std::string& name)
	{
		std::map<std::string, int>::iterator it = map.find(name);
		return it == map.end() ? -1 : it->second;
	}
};

bool parseFloats(const char*& ptr, float* values, int num)
{
	const char* tmp = ptr;
	for (int i = 0; i < num; i++)
	{
		if (i)
			parseSpace(tmp);
		if (!parseFloat(tmp, values[i]))
			return false;
	}
	ptr = tmp;
	return true;
}

bool parseTexture(const char*& ptr, TextureSpec& value, const std::string& dirName)
{
	// Initialize result.

	std::string name;
	value.texture = "";
	value.base = 0.0f;
	value.gain = 1.0f;

	// Parse options.

	while (*ptr)
	{
		parseSpace(ptr);
		if ((parseLiteral(ptr, "-blendu ") || parseLiteral(ptr, "-blendv ") || parseLiteral(ptr, "-cc ") || parseLiteral(ptr, "-math::clamp ")) && parseSpace(ptr))
		{
			if (!parseLiteral(ptr, "on") && !parseLiteral(ptr, "off"))
				return false;
		}
		else if (parseLiteral(ptr, "-mm ") && parseSpace(ptr))
		{
			if (!parseFloat(ptr, value.base) || !parseSpace(ptr) || !parseFloat(ptr, value.gain))
				return false;
		}
		else if ((parseLiteral(ptr, "-o ") || parseLiteral(ptr, "-s ") || parseLiteral(ptr, "-t ")) && parseSpace(ptr))
		{
			float tmp[2];
			if (!parseFloats(ptr, tmp, 2))
				return false;
			parseSpace(ptr);
			parseFloat(ptr, tmp[0]);
		}
		else if ((parseLiteral(ptr, "-texres ") || parseLiteral(ptr, "-bm ")) && parseSpace(ptr))
		{
			float tmp;
			if (!parseFloat(ptr, tmp))
				return false;
		}
		else if (parseLiteral(ptr, "-type ") && parseSpace(ptr))
		{
			if (!parseLiteral(ptr, "sphere") &&
				!parseLiteral(ptr, "cube_top") && !parseLiteral(ptr, "cube_bottom") &&
				!parseLiteral(ptr, "cube_front") && !parseLiteral(ptr, "cube_back") &&
				!parseLiteral(ptr, "cube_left") && !parseLiteral(ptr, "cube_right"))
			{
				return false;
			}
		}
		else
		{
			if (*ptr == '-' || name.size())
				return false;
			while (*ptr && (*ptr != '-' || !boost::algorithm::ends_with(name, " ")))
				name += *ptr++;
		}
	}

	// Process file name.

	while (boost::algorithm::starts_with(name, "/"))
		name = name.substr(1);
	while (boost::algorithm::ends_with(name, " "))
		name = name.substr(0, name.size() - 1);

	// Zero-length file name => ignore.

	if (!name.size())
		return true;

	// Import texture.

	value.texture = dirName + '/' + name;

	return true;
}

struct ImportState
{
	std::vector<Vec3f>						positions;
	std::vector<Vec3f>						normals;
	std::vector<Vec2f>						texCoords;

	MatHash materialHash;
};

void loadMtl(ImportState& s, IInStream& mtlIn, const std::string& dirName)
{
	char ptrLast[256];
	Material* mat = NULL;
	std::string lineS;
	while (mtlIn.getline(lineS))
	{
		boost::algorithm::trim(lineS);
		const char* ptr = lineS.c_str();
		parseSpace(ptr);
		bool valid = false;

		if (!*ptr || parseLiteral(ptr, "#"))
		{
			valid = true;
		}
		else if (parseLiteral(ptr, "newmtl ") && parseSpace(ptr) && *ptr) // material name
		{
			if (mat != 0)
				s.materialHash.add(std::string(ptrLast), *mat);
			if (!s.materialHash.contains(std::string(ptr)))
			{
				mat = new Material();
				Platform::SetMemory(ptrLast, sizeof(ptrLast));
				memcpy(ptrLast, ptr, strlen(ptr));
				mat->Name = std::string(ptrLast);
			}
			valid = true;
		}
		else if (parseLiteral(ptr, "Ka ") && parseSpace(ptr)) // ambient color
		{
			float tmp[3];
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, tmp, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Kd ") && parseSpace(ptr)) // diffuse color
		{
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, (float*)&mat->diffuse, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ks ") && parseSpace(ptr)) // specular color
		{
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, (float*)&mat->specular, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "d ") && parseSpace(ptr)) // alpha
		{
			if (parseFloat(ptr, mat->diffuse.w) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ns ") && parseSpace(ptr)) // glossiness
		{
			if (parseFloat(ptr, mat->glossiness) && parseSpace(ptr) && !*ptr)
				valid = true;
			if (mat->glossiness <= 0.0f)
			{
				mat->glossiness = 1.0f;
				mat->specular = Vec3f(0);
			}
		}
		else if (parseLiteral(ptr, "map_Kd ")) // diffuse texture
		{
			TextureSpec tex;
			mat->textures[TextureType_Diffuse] = std::string(ptr);
			valid = parseTexture(ptr, tex, dirName);
		}
		else if (parseLiteral(ptr, "Ke "))
		{
			if (parseFloats(ptr, (float*)&mat->emission, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Tf "))
		{
			if (parseFloats(ptr, (float*)&mat->Tf, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ni ") && parseSpace(ptr)) // alpha
		{
			if (parseFloat(ptr, mat->IndexOfRefraction) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "illum ") && parseSpace(ptr)) // alpha
		{
			if (parseInt(ptr, mat->IlluminationModel) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "map_d ") || parseLiteral(ptr, "map_D ") || parseLiteral(ptr, "map_opacity ")) // alpha texture
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->textures[TextureType_Alpha] = tex.texture;
		}
		else if (parseLiteral(ptr, "disp ")) // displacement map
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->displacementCoef = tex.gain;
			mat->displacementBias = tex.base * tex.gain;
			mat->textures[TextureType_Displacement] = tex.texture;
		}
		else if (parseLiteral(ptr, "bump ") || parseLiteral(ptr, "map_bump ") || parseLiteral(ptr, "map_Bump ")) // bump map
		{
			TextureSpec tex;
			mat->displacementCoef = tex.gain;
			mat->displacementBias = tex.base * tex.gain;
			mat->textures[TextureType_Displacement] = std::string(ptr);
			valid = parseTexture(ptr, tex, dirName);
		}
		else if (parseLiteral(ptr, "refl ")) // environment map
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->textures[TextureType_Environment] = tex.texture;
		}
		else if (
			parseLiteral(ptr, "vp ") ||             // parameter space vertex
			parseLiteral(ptr, "Kf ") ||             // transmission color
			parseLiteral(ptr, "illum ") ||          // illumination model
			parseLiteral(ptr, "d -halo ") ||        // orientation-dependent alpha
			parseLiteral(ptr, "sharpness ") ||      // reflection sharpness
			parseLiteral(ptr, "Ni ") ||             // index of refraction
			parseLiteral(ptr, "map_Ks ") ||         // specular texture
			parseLiteral(ptr, "map_kS ") ||         // ???
			parseLiteral(ptr, "map_kA ") ||         // ???
			parseLiteral(ptr, "map_Ns ") ||         // glossiness texture
			parseLiteral(ptr, "map_aat ") ||        // texture antialiasing
			parseLiteral(ptr, "decal ") ||          // blended texture
			parseLiteral(ptr, "Km ") ||             // ???
			parseLiteral(ptr, "Tr ") ||             // ???
			parseLiteral(ptr, "Ke ") ||             // ???
			parseLiteral(ptr, "pointgroup ") ||     // ???
			parseLiteral(ptr, "pointdensity ") ||   // ???
			parseLiteral(ptr, "smooth") ||          // ???
			parseLiteral(ptr, "R "))                // ???
		{
			valid = true;
		}
	}
	if (mat != 0)
		s.materialHash.add(std::string(ptrLast), *mat);
}

struct SubMesh
{
	int indexStart;
	int mat;
	SubMesh(unsigned int start)
		: indexStart(start)
	{
	}
};

static e_Texture CreateTexture(const char* p, const Spectrum& col)
{
	if (p && *p)
		return CreateTexture(p);
	else return CreateTexture(col);
}

void compileobj(IInStream& in, OutputStream& a_Out)
{
	std::string dirName = boost::filesystem::path(in.getFilePath()).parent_path().string();
	ImportState* state = new ImportState();
	ImportState& s = *state;//do not put it on the stack it is kinda big
	int submesh = -1;
	int defaultSubmesh = -1;

	std::vector<Vec3f> positions;
	std::vector<Vec3f> normals;
	std::vector<Vec2f> texCoords;
	std::vector<unsigned int> indices;
	std::vector<SubMesh> subMeshes;
	AABB box = AABB::Identity();

	std::string line;
	while (in.getline(line))
	{
		boost::algorithm::trim(line);
		const char* ptr = line.c_str();
		parseSpace(ptr);
		bool valid = false;

		if (!*ptr || parseLiteral(ptr, "#"))
		{
			valid = true;
		}
		else if (parseLiteral(ptr, "v ") && parseSpace(ptr)) // position vertex
		{
			Vec3f v;
			if (parseFloats(ptr, (float*)&v, 3) && parseSpace(ptr) && !*ptr)
			{
				s.positions.push_back(v);
				valid = true;
			}
		}
		else if (parseLiteral(ptr, "vn ") && parseSpace(ptr)) // position vertex
		{
			Vec3f v;
			if (parseFloats(ptr, (float*)&v, 3) && parseSpace(ptr) && !*ptr)
			{
				s.normals.push_back(v);
				valid = true;
			}
		}
		else if (parseLiteral(ptr, "vt ") && parseSpace(ptr)) // texture vertex
		{
			Vec2f v;
			if (parseFloats(ptr, (float*)&v, 2) && parseSpace(ptr))
			{
				float dummy;
				while (parseFloat(ptr, dummy) && parseSpace(ptr));

				if (!*ptr)
				{
					s.texCoords.push_back(Vec2f(v.x, 1.0f - v.y));
					valid = true;
				}
			}
		}
		else if (parseLiteral(ptr, "f ") && parseSpace(ptr)) // face
		{
			while (*ptr)
			{
				int ptn[3];
				if (!parseInt(ptr, ptn[0]))
					break;
				for (int i = 1; i < 4 && parseLiteral(ptr, "/"); i++)
				{
					int tmp = 0;
					parseInt(ptr, tmp);
					if (i < 3)
						ptn[i] = tmp;
				}
				parseSpace(ptr);

				int size[3] = { (int)s.positions.size(), (int)s.texCoords.size(), (int)s.normals.size() };
				for (int i = 0; i < 3; i++)
				{
					if (ptn[i] < 0)
						ptn[i] += size[i];
					else
						ptn[i]--;

					if (ptn[i] < 0 || ptn[i] >= size[i])
						ptn[i] = -1;
				}

				Vec3f p = (ptn[0] == -1) ? Vec3f(0.0f) : s.positions[ptn[0]];
				box.Enlarge(p);
				positions.push_back(p);
				texCoords.push_back((ptn[1] == -1) ? Vec2f(0.0f) : s.texCoords[ptn[1]]);
				normals.push_back((ptn[2] == -1) ? Vec3f(0.0f) : s.normals[ptn[2]]);
				indices.push_back(unsigned int(positions.size() - 1));
			}
		}
		else if (parseLiteral(ptr, "usemtl ") && parseSpace(ptr)) // material name
		{
			int mat = s.materialHash.searchi(std::string(ptr));
			if (submesh != -1)
			{
				submesh = -1;
			}
			if (mat != -1)
			{
				SubMesh q = SubMesh((unsigned int)indices.size());
				q.mat = mat;
				subMeshes.push_back(q);
			}
			valid = true;
		}
		else if (parseLiteral(ptr, "mtllib ") && parseSpace(ptr) && *ptr) // material library
		{
			if (dirName.size())
			{
				boost::algorithm::trim(std::string(ptr));
				std::string fileName = dirName + "/" + ptr;
				MemInputStream mtlIn(fileName.c_str());
				loadMtl(s, mtlIn, dirName);
				mtlIn.Close();
			}
			valid = true;
		}
		else if (
			parseLiteral(ptr, "vp ") ||         // parameter space vertex
			parseLiteral(ptr, "deg ") ||        // degree
			parseLiteral(ptr, "bmat ") ||       // basis matrix
			parseLiteral(ptr, "step ") ||       // step size
			parseLiteral(ptr, "cstype ") ||     // curve/surface type
			parseLiteral(ptr, "p ") ||          // point
			parseLiteral(ptr, "l ") ||          // line
			parseLiteral(ptr, "curv ") ||       // curve
			parseLiteral(ptr, "curv2 ") ||      // 2d curve
			parseLiteral(ptr, "surf ") ||       // surface
			parseLiteral(ptr, "parm ") ||       // curve/surface parameters
			parseLiteral(ptr, "trim ") ||       // curve/surface outer trimming loop
			parseLiteral(ptr, "hole ") ||       // curve/surface inner trimming loop
			parseLiteral(ptr, "scrv ") ||       // curve/surface special curve
			parseLiteral(ptr, "sp ") ||         // curve/surface special point
			parseLiteral(ptr, "end ") ||        // curve/surface end statement
			parseLiteral(ptr, "con ") ||        // surface connect
			parseLiteral(ptr, "g ") ||          // group name
			parseLiteral(ptr, "s ") ||          // smoothing group
			parseLiteral(ptr, "mg ") ||         // merging group
			parseLiteral(ptr, "o ") ||          // object name
			parseLiteral(ptr, "bevel ") ||      // bevel interpolation
			parseLiteral(ptr, "c_interp ") ||   // color interpolation
			parseLiteral(ptr, "d_interp ") ||   // dissolve interpolation
			parseLiteral(ptr, "lod ") ||        // level of detail
			parseLiteral(ptr, "shadow_obj ") || // shadow casting
			parseLiteral(ptr, "trace_obj ") ||  // ray tracing
			parseLiteral(ptr, "ctech ") ||      // curve approximation technique
			parseLiteral(ptr, "stech ") ||      // surface approximation technique
			parseLiteral(ptr, "g"))             // ???
		{
			valid = true;
		}
	}

	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	Platform::SetMemory(m_sLights, sizeof(m_sLights));
	int c = 0, lc = 0;
	std::vector<e_KernelMaterial> matData;
	matData.reserve(s.materialHash.vec.size());
	for (size_t i = 0; i < s.materialHash.vec.size(); i++)
	{
		Material M = s.materialHash.vec[i];
		e_KernelMaterial mat(M.Name.c_str());
		float f = 0.0f;
		if (M.IlluminationModel == 2)
		{
			diffuse d;
			d.m_reflectance = CreateTexture(M.textures[0].c_str(), Spectrum(M.diffuse.x, M.diffuse.y, M.diffuse.z));
			mat.bsdf.SetData(d);
		}
		else if (M.IlluminationModel == 5)
		{/*
		 dielectric d;
		 d.m_invEta = d.m_eta = 1;
		 d.m_specularReflectance = CreateTexture(0, Spectrum(M.specular));
		 d.m_specularTransmittance = CreateTexture(0, Spectrum(0.0f));
		 mat.bsdf.SetData(d);*/
			mat.bsdf.SetData(conductor(Spectrum(0.0f), Spectrum(1.0f)));
		}
		else if (M.IlluminationModel == 7)
		{
			dielectric d;
			d.m_eta = M.IndexOfRefraction;
			d.m_invEta = 1.0f / M.IndexOfRefraction;
			d.m_specularReflectance = CreateTexture(0, Spectrum(M.specular.x, M.specular.y, M.specular.z));
			d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf.x, M.Tf.y, M.Tf.z));
			mat.bsdf.SetData(d);
		}
		else if (M.IlluminationModel == 9)
		{
			dielectric d;
			d.m_eta = M.IndexOfRefraction;
			d.m_invEta = 1.0f / M.IndexOfRefraction;
			d.m_specularReflectance = CreateTexture(0, Spectrum(0.0f));
			d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf.x, M.Tf.y, M.Tf.z));
			mat.bsdf.SetData(d);
		}
		if (M.textures[TextureType_Displacement].size())
		{
			mat.SetHeightMap(M.textures[TextureType_Displacement].c_str());
		}
		/*
		if(0&&M.textures[3].c_str().size())
		{
		char* c = new char[M.textures[3].getID().getLength()+1];
		ZeroMemory(c, M.textures[3].getID().getLength()+1);
		memcpy(c, M.textures[3].getID().getPtr(), M.textures[3].getID().getLength());
		OutputDebugString(c);
		OutputDebugString(M.textures[3].getID().getPtr());
		mat.SetHeightMap(CreateTexture(c, make_float3(0)));
		mat.HeightScale = 0.5f;
		}

		#ifdef EXT_TRI
		if(M.textures[0].hasData())
		memcpy(m.m_cDiffusePath, M.textures[0].getID().getPtr(), M.textures[0].getID().getLength());
		if(M.textures[3].hasData())
		memcpy(m.m_cNormalPath, M.textures[3].getID().getPtr(), M.textures[3].getID().getLength());
		#endif
		*/
		/*		Material M = s.materialHash.data[matIndex].value;
		if(length(M.emission) != 0)
		{
		//duplicate material
		matData.push_back(matData[matIndex]);
		matIndex = matData.size() - 1;
		std::cout << "duplicating material : " << matData[matIndex].Name << "\n";

		Platform::SetMemory(m_sLights + lc, sizeof(e_MeshPartLight));
		m_sLights[lc].L = M.emission;
		strcpy(m_sLights[lc].MatName, M.Name.c_str());
		matData[matIndex].NodeLightIndex = lc++;
		}
		*/
		if (length(M.emission))
		{
			m_sLights[lc].L = Spectrum(M.emission.x, M.emission.y, M.emission.z);
			m_sLights[lc].MatName = M.Name;
			mat.NodeLightIndex = lc++;
		}
		matData.push_back(mat);
	}

	unsigned int m_numTriangles = (unsigned int)indices.size() / 3;
	unsigned int m_numVertices = (unsigned int)positions.size();
	e_TriangleData* triData = new e_TriangleData[m_numTriangles];
	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
#ifdef EXT_TRI
	Vec3f* v_Normals = new Vec3f[m_numVertices], *v_Tangents = new Vec3f[m_numVertices], *v_BiTangents = new Vec3f[m_numVertices];
	Platform::SetMemory(v_Normals, sizeof(Vec3f) * m_numVertices);
	Platform::SetMemory(v_Tangents, sizeof(Vec3f) * m_numVertices);
	Platform::SetMemory(v_BiTangents, sizeof(Vec3f) * m_numVertices);
	ComputeTangentSpace(&positions[0], &texCoords[0], &indices[0], m_numVertices, m_numTriangles, v_Normals, v_Tangents, v_BiTangents);
	if (m_numVertices == m_numTriangles * 3)
		v_Normals = &normals[0];
#endif

	for (unsigned int submesh = 0; submesh < subMeshes.size(); submesh++)
	{
		size_t matIndex = subMeshes[submesh].mat;

		unsigned int start = (unsigned int)subMeshes[submesh].indexStart, end = submesh < subMeshes.size() - 1 ? (unsigned int)subMeshes[submesh + 1].indexStart - 1 : (unsigned int)indices.size();
		for (size_t i = start; i < end; i += 3)
		{
			unsigned int* vi = &indices[i];
			for (size_t j = 0; j < 3; j++)
			{
				const int l = vi[j];
				p[j] = positions[l];
#ifdef EXT_TRI
				t[j] = texCoords[l];
				ta[j] = normalize(v_Tangents[l]);
				bi[j] = normalize(v_BiTangents[l]);
				n[j] = normalize(v_Normals[l]);
				//n[j] = v.n;
#endif
			}
			triData[c++] = e_TriangleData(p, (unsigned char)matIndex, t, n, ta, bi);
		}
	}

	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lc;
	a_Out << m_numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * m_numTriangles);
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	ConstructBVH(&positions[0], &indices[0], m_numVertices, m_numTriangles * 3, a_Out);
#ifdef EXT_TRI
	if (m_numVertices != m_numTriangles * 3)
		delete[] v_Normals;
	delete[] v_Tangents;
	delete[] v_BiTangents;
#endif
	delete[] triData;
}