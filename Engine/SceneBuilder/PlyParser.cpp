#include <StdAfx.h>
#include "..\e_Mesh.h"
#include "..\..\Base\FrameworkInterop.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <intrin.h>
#define TS_DEC_FRAMEWORK
#include <Base\TangentSpace.h>
#include <Engine\SceneBuilder\Importer.h>

typedef int format_type;
enum format {
	binary_little_endian_format = 0, 
	binary_big_endian_format = 1,
	ascii_format = 2
};

struct varReader
{
	//i dream unsigned^^
	enum type
	{
		u8 = 1,
		u16 = 2,
		u32 = 4,
		f32 = 5,
		f64 = 8,
		tinvalid,
	};
	type t;
	varReader()
	{
		t = tinvalid;
	}
	varReader(std::string type)
	{
		if(type[0] == 'u')
			type = type.substr(1, type.length() - 1);
		if(type == "char" || type == "int8")
			t = u8;
		else if(type == "short" || type == "int16")
			t = u16;
		else if(type == "int" || type == "int32")
			t = u32;
		else if(type == "float")
			t = f32;
		else if(type == "double")
			t = f64;
	}
	///in bytes
	unsigned int typeSize()
	{
		return t == f32 ? 4 : (unsigned int)t;
	}
	void read(std::string el, unsigned int* ures, float* fres)
	{
		switch (t)
		{
		case varReader::u8:
		case varReader::u16:
		case varReader::u32:
			*ures = std::stoi(el);
			break;
		case varReader::f32:
		case varReader::f64:
			*fres = (float)std::atof(el.c_str());
			break;
		default:
			throw 1;
		}
	}
	void read(void* buf, format_type format, unsigned int* ures, float* fres)
	{
		switch (t)
		{
		case varReader::u8:
			*ures = *(unsigned char*)buf;
			break;
		case varReader::u16:
			*ures = _byteswap_ushort(*(unsigned short*)buf);
			break;
		case varReader::u32:
			*ures = _byteswap_ulong(*(unsigned int*)buf);
			break;
		case varReader::f32:
			*fres = int_as_float_((_byteswap_ulong(*(unsigned int*)buf)));
			break;
		case varReader::f64:
			*fres = int_as_float_(_byteswap_uint64((unsigned int)*(double*)buf));
			break;
		default:
			throw 1;
		}
	}
};

struct stringHelper
{
	std::string var;
	int pos, next;
	stringHelper(){}
	stringHelper(std::string v)
	{
		var = v;
		pos = 0;
		next = -1;
	}
	bool hasNext()
	{
		pos = next + 1;
		next = var.find(' ', pos);
		return next != -1;
	}
	char* cString()
	{
		if(next != -1)
			var[next] = 0;
		return &var[pos];
	}
	char* nextC()
	{
		hasNext();
		return cString();
	}
};

void compileply(const char* a_InputFile, OutputStream& a_Out)
{
	format_type format;
	std::string line;
	int line_number_ = 0;
	InputStream istream(a_InputFile);
	//std::fstream istream(a_InputFile, std::ios::in);
	char magic[3];
	istream.Read(magic, 3);
	istream.Move(1);
	++line_number_;
	int vertexCount = -1, faceCount = -1;
	int hasUV = 0, vertexProp = 0, hasPos = 0;
	int posStart = -1, uvStart = -1, elementIndex = 0;
	varReader listCount, listElements;
	
	while (istream.getline(line))
	{
		++line_number_;
		std::istringstream stringstream(line);
		stringstream.unsetf(std::ios_base::skipws);

		stringstream >> std::ws;
		std::string keyword;
		stringstream >> keyword;
		if (keyword == "format")
		{
			std::string format_string, version;
			char space_format_format_string, space_format_string_version;
			stringstream >> space_format_format_string >> std::ws >> format_string >> space_format_string_version >> std::ws >> version >> std::ws;
			if (format_string == "ascii")
				format = ascii_format;
			else if (format_string == "binary_big_endian")
				format = binary_big_endian_format;
			else if (format_string == "binary_little_endian")
				format = binary_little_endian_format;
		}
		else if (keyword == "element")
		{
			std::string name;
			std::size_t count;
			char space_element_name, space_name_count;
			stringstream >> space_element_name >> std::ws >> name >> space_name_count >> std::ws >> count >> std::ws;
			vertexProp = false;
			if(name == "vertex")
			{
				elementIndex = 0;
				vertexCount = count;
				vertexProp = true;
			}
			else if(name == "face")
				faceCount = count;
		}
		else if (keyword == "property")
		{
			std::string type_or_list;
			char space_property_type_or_list;
			stringstream >> space_property_type_or_list >> std::ws >> type_or_list;
			if(vertexProp)
			{
				if (type_or_list != "list")
				{
					std::string name;
					std::string& type = type_or_list;
					char space_type_name;
					stringstream >> space_type_name >> std::ws >> name >> std::ws;
					if(name[0] >= 'x' && name[0] <= 'z' && name.length() == 1 && (type == "float" || type == "double"))
					{
						hasPos |= 1 << (name[0] - 'x');
						posStart = name[0] == 'x' ? elementIndex : posStart;
					}
					else if(name[0] == 'u' || name[0] == 'v' && name.length() == 1 && (type == "float" || type == "double"))
					{
						hasUV |= 1 << (name[0] - 'u');
						uvStart = name[0] == 'u' ? elementIndex : uvStart;
					}
					else if(name.find("material") != -1)
						throw 1;
				}
				else throw 1;
				elementIndex++;
			}
			else
			{
				std::string name;
				std::string size_type_string, scalar_type_string;
				char space_list_size_type, space_size_type_scalar_type, space_scalar_type_name;
				stringstream >> space_list_size_type >> std::ws >> size_type_string >> space_size_type_scalar_type >> std::ws >> scalar_type_string >> space_scalar_type_name >> std::ws >> name >> std::ws;
				if(name == "vertex_indices")
				{
					listCount = varReader(size_type_string);
					listElements = varReader(scalar_type_string);
				}
				else throw 1;
			}
		}
		else if(keyword == "end_header")
			break;
	}
	if(hasPos != 7)
		throw 1;

	FW::Mesh<FW::VertexPNT> M2;
	FW::VertexPNT* Vertices = (FW::VertexPNT*)malloc(vertexCount * sizeof(FW::VertexPNT));
	unsigned int* Indices = (unsigned int*)malloc(sizeof(unsigned int) * faceCount * 4);
	unsigned int indexCount = 0;
		
	varReader fReader("float");
	if (format == ascii_format)
	{
		//TODO This is slow cause of the use of native stream
		unsigned int* stack = (unsigned int*)alloca(MAX(elementIndex, 4) * sizeof(unsigned int));
		stringHelper hlp;
		for(int v = 0; v < vertexCount; v++)	
		{
			istream.getline(line);
			hlp = stringHelper(line);
			for(unsigned int i = 0; i < elementIndex; i++)
				fReader.read(hlp.nextC(), 0, (float*)stack + i);
			FW::VertexPNT V;
			float* d = (float*)stack;
			V.p = FW::Vec3f(d[posStart + 0], d[posStart + 1], d[posStart + 2]);
			if(hasUV)
				V.t = FW::Vec2f(d[uvStart + 0], d[uvStart + 0]);
			Vertices[v] = V;
		}
		for(int f = 0; f < faceCount; f++)	//1 -> 0
		{
			istream.getline(line);
			unsigned int verticesPerFace;
			hlp = stringHelper(line);
			listElements.read(hlp.nextC(), &verticesPerFace, 0);		
			for(unsigned int i = 0; i < verticesPerFace; i++)
				listElements.read(hlp.nextC(), stack + i, 0);
			if(verticesPerFace == 3)
			{
				Indices[indexCount++] = (stack[0]);
				Indices[indexCount++] = (stack[1]);
				Indices[indexCount++] = (stack[2]);
			}
			else if(verticesPerFace == 4)
			{
				for(unsigned int i = 2; i < 5; i++)
					Indices[indexCount++] = (stack[i % 4]);
				for(unsigned int i = 0; i < 3; i++)
					Indices[indexCount++] = (stack[i]);
			}
			else throw 1;
		}
	}
	else
	{
		char* FILE_BUF = (char*)malloc(istream.getFileSize() - istream.getPos());
		istream.Read(FILE_BUF, istream.getFileSize() - istream.getPos() - 1);
		unsigned int file_pos = 0;

		FW::Vec3f p;
		for(int v = 0; v < vertexCount; v++)
		{
			Vertices[v].p = *(FW::Vec3f*)(FILE_BUF + file_pos);
			Vertices[v].n = FW::Vec3f(0.0f);
			Vertices[v].t = FW::Vec2f(0.0f);
			file_pos += 12;
		}
		for(int f = 0; f < faceCount; f++)
		{
			unsigned int* dat = (unsigned int*)(FILE_BUF + file_pos + 1);
			if(FILE_BUF[file_pos] == 3)
			{
				Indices[indexCount++] = dat[0];
				Indices[indexCount++] = dat[1];
				Indices[indexCount++] = dat[2];
			}
			else if(FILE_BUF[file_pos] == 4)
			{
				for(unsigned int i = 2; i < 5; i++)
					Indices[indexCount++] = dat[i % 4];
				for(unsigned int i = 0; i < 3; i++)
					Indices[indexCount++] = dat[i];
			}
			else throw 1;
			file_pos += 4 * FILE_BUF[file_pos] + 1;
		}
		free(FILE_BUF);
	}
	unsigned int pos = istream.getPos(), size = istream.getFileSize();
	//if(pos != size)
	//	throw 1;

	M2.addSubmesh();
	M2.addVertices(&Vertices[0], (int)vertexCount);
	float3* Normals = new float3[vertexCount], *Tangents = new float3[vertexCount];
	M2.mutableIndices(0).add((FW::Vec3i*)&Indices[0], indexCount / 3);
	ComputeTangentSpace(&M2, Normals, Tangents);
	e_TriangleData* triData = new e_TriangleData[indexCount / 3];
	float3 p[3];
	float3 n[3];
	float3 ta[3];
	float3 bi[3];
	float2 te[3];
		AABB box = AABB::Identity();
	for(unsigned int t = 0; t < indexCount; t += 3)
	{
		FW::Vec3i index(Indices[t], Indices[t+1], Indices[t+2]);
		for(size_t j = 0; j < 3; j++)
		{
			const int l = index.get((int)j);
			const FW::VertexPNT& v = M2[l];
			p[j] = make_float3(v.p.x, v.p.y, v.p.z);
			te[j] = make_float2(v.t.x, v.t.y);
			ta[j] = Tangents[l];
			n[j] = Normals[l];
			box.Enlarge(v.p);
		}
		triData[t / 3] = e_TriangleData(p, (unsigned char)0, te, n, ta, bi);
	}
	delete [] Normals;
	delete [] Tangents;

	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	ZeroMemory(m_sLights, sizeof(m_sLights));
	e_KernelMaterial defaultMat("Default_Material");
	diffuse mat;
	mat.m_reflectance = CreateTexture(0, Spectrum(1,0,0));
	defaultMat.bsdf.SetData(mat);
	M2.compact();
	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << 0;
	a_Out << (unsigned int)(indexCount / 3);
	a_Out.Write(&triData[0], sizeof(e_TriangleData) * (int)(indexCount / 3));
	a_Out << 1;
	a_Out.Write(&defaultMat, sizeof(e_KernelMaterial) * 1);
	ConstructBVH2(&M2, TmpOutStream(&a_Out));
	::free(Vertices);
	::free(Indices);
	delete [] triData;
}