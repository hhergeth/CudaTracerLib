#include <StdAfx.h>
#include <Engine/Mesh.h>
#include <Engine/TriangleData.h>
#include <Engine/Material.h>
#include <Engine/TriIntersectorData.h>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include "TangentSpaceHelper.h"
#include "BVHBuilderHelper.h"
#include <Base/FileStream.h>

namespace CudaTracerLib {

typedef int format_type;
enum format {
	binary_little_endian_format = 0,
	binary_big_endian_format = 1,
	ascii_format = 2
};

//http://stackoverflow.com/questions/105252/how-do-i-convert-between-big-endian-and-little-endian-values-in-c
//credits to Alexandre C.
template <typename T>
T swap_endian(T u)
{
	static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

	union
	{
		T u;
		unsigned char u8[sizeof(T)];
	} source, dest;

	source.u = u;

	for (size_t k = 0; k < sizeof(T); k++)
		dest.u8[k] = source.u8[sizeof(T) - k - 1];

	return dest.u;
}

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
		if (type[0] == 'u')
			type = type.substr(1, type.length() - 1);
		if (type == "char" || type == "int8")
			t = u8;
		else if (type == "short" || type == "int16")
			t = u16;
		else if (type == "int" || type == "int32")
			t = u32;
		else if (type == "float")
			t = f32;
		else if (type == "double")
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
			throw std::runtime_error("Invalid ply type!");
		}
	}
	void read(void* buf, format_type format, unsigned int* ures, float* fres)
	{
		auto a = swap_endian(*(unsigned long long*)buf);
		switch (t)
		{
		case varReader::u8:
			*ures = *(unsigned char*)buf;
			break;
		case varReader::u16:
			*ures = swap_endian(*(unsigned short*)buf);
			break;
		case varReader::u32:
			*ures = swap_endian(*(unsigned int*)buf);
			break;
		case varReader::f32:
			*fres = int_as_float_((swap_endian(*(unsigned int*)buf)));
			break;
		case varReader::f64:
			*fres = (float)*(double*)&a;
			break;
		default:
			throw std::runtime_error("Invalid ply type!");
		}
	}
};

struct stringHelper
{
	std::string var;
	unsigned int pos, next;
	stringHelper(){}
	stringHelper(std::string v)
	{
		var = v;
		pos = 0;
		next = (unsigned int)-1;
	}
	bool hasNext()
	{
		pos = next + 1;
		next = (unsigned int)var.find(' ', pos);
		return next != (unsigned int)-1;
	}
	char* cString()
	{
		if (next != (unsigned int)-1)
			var[next] = 0;
		return &var[pos];
	}
	char* nextC()
	{
		hasNext();
		return cString();
	}
};

unsigned int LongSwap(unsigned int i)
{
	unsigned char b1, b2, b3, b4;

	b1 = i & 255;
	b2 = (i >> 8) & 255;
	b3 = (i >> 16) & 255;
	b4 = (i >> 24) & 255;

	return ((unsigned int)b1 << 24) + ((unsigned int)b2 << 16) + ((unsigned int)b3 << 8) + b4;
}

float LongSwap(const float inFloat)
{
	float retVal;
	char *floatToConvert = (char*)& inFloat;
	char *returnFloat = (char*)& retVal;

	// swap the bytes into a temporary buffer
	returnFloat[0] = floatToConvert[3];
	returnFloat[1] = floatToConvert[2];
	returnFloat[2] = floatToConvert[1];
	returnFloat[3] = floatToConvert[0];

	return retVal;
}

void compileply(IInStream& istream, FileOutputStream& a_Out)
{
	format_type format;
	std::string line;
	int line_number_ = 0;
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
			if (name == "vertex")
			{
				elementIndex = 0;
				vertexCount = (int)count;
				vertexProp = true;
			}
			else if (name == "face")
				faceCount = (int)count;
		}
		else if (keyword == "property")
		{
			std::string type_or_list;
			char space_property_type_or_list;
			stringstream >> space_property_type_or_list >> std::ws >> type_or_list;
			if (vertexProp)
			{
				if (type_or_list != "list")
				{
					std::string name;
					std::string& type = type_or_list;
					char space_type_name;
					stringstream >> space_type_name >> std::ws >> name >> std::ws;
					if (name[0] >= 'x' && name[0] <= 'z' && name.length() == 1 && (type == "float" || type == "double"))
					{
						hasPos |= 1 << (name[0] - 'x');
						posStart = name[0] == 'x' ? elementIndex : posStart;
					}
					else if (name[0] == 'u' || name[0] == 'v' && name.length() == 1 && (type == "float" || type == "double"))
					{
						hasUV |= 1 << (name[0] - 'u');
						uvStart = name[0] == 'u' ? elementIndex : uvStart;
					}
					else if (name.find("material") != -1)
						throw std::runtime_error(__FUNCTION__);
				}
				else throw std::runtime_error(__FUNCTION__);
				elementIndex++;
			}
			else
			{
				std::string name;
				std::string size_type_string, scalar_type_string;
				char space_list_size_type, space_size_type_scalar_type, space_scalar_type_name;
				stringstream >> space_list_size_type >> std::ws >> size_type_string >> space_size_type_scalar_type >> std::ws >> scalar_type_string >> space_scalar_type_name >> std::ws >> name >> std::ws;
				if (name == "vertex_indices")
				{
					listCount = varReader(size_type_string);
					listElements = varReader(scalar_type_string);
				}
				//else throw std::runtime_error(__FUNCTION__);
			}
		}
		else if (keyword == "end_header")
			break;
	}
	if (hasPos != 7)
		throw std::runtime_error(__FUNCTION__);

	std::vector<Vec3f> vertices(vertexCount);
	std::vector<Vec2f> texCoords(vertexCount);
	std::vector<unsigned int> indices(sizeof(unsigned int) * faceCount * 6);//don't know if they are triangles or quads
	unsigned int indexCount = 0;

	varReader fReader("float");
	if (format == ascii_format)
	{
		//TODO This is slow cause of the use of native stream
		unsigned int* stack = (unsigned int*)alloca(max(elementIndex, 4) * sizeof(unsigned int));
		stringHelper hlp;
		for (int v = 0; v < vertexCount; v++)
		{
			istream.getline(line);
			hlp = stringHelper(line);
			for (int i = 0; i < elementIndex; i++)
				fReader.read(hlp.nextC(), 0, (float*)stack + i);
			float* d = (float*)stack;
			vertices[v] = Vec3f(d[posStart + 0], d[posStart + 1], d[posStart + 2]);
			if (hasUV)
				texCoords[v] = Vec2f(d[uvStart + 0], d[uvStart + 0]);
		}
		for (int f = 0; f < faceCount; f++)	//1 -> 0
		{
			istream.getline(line);
			unsigned int verticesPerFace;
			hlp = stringHelper(line);
			listElements.read(hlp.nextC(), &verticesPerFace, 0);
			for (unsigned int i = 0; i < verticesPerFace; i++)
				listElements.read(hlp.nextC(), stack + i, 0);
			if (verticesPerFace == 3)
			{
				indices[indexCount++] = (stack[2]);
				indices[indexCount++] = (stack[1]);
				indices[indexCount++] = (stack[0]);
			}
			else if (verticesPerFace == 4)
			{
				for (unsigned int i = 2; i < 5; i++)
					indices[indexCount++] = (stack[i % 4]);
				for (unsigned int i = 0; i < 3; i++)
					indices[indexCount++] = (stack[i]);
			}
			else throw std::runtime_error(__FUNCTION__);
		}
	}
	else
	{
		char* FILE_BUF = (char*)malloc(istream.getFileSize() - istream.getPos());
		istream.Read(FILE_BUF, istream.getFileSize() - istream.getPos() - 1);
		unsigned int file_pos = 0;

		memcpy(&vertices[0], FILE_BUF + file_pos, sizeof(Vec3f) * vertexCount);
		float* fData = (float*)&vertices[0];
		if (format == binary_big_endian_format)
			for (int i = 0; i < vertexCount * 3; i++)
				fData[i] = LongSwap(fData[i]);

		file_pos += sizeof(Vec3f) * vertexCount;
		for (int f = 0; f < faceCount; f++)
		{
			unsigned int* dat = (unsigned int*)(FILE_BUF + file_pos + 1);
			if (FILE_BUF[file_pos] == 3)
			{
				for (int idx = 0; idx < 3; idx++)
					indices[indexCount++] = format == binary_little_endian_format ? dat[2 - idx] : LongSwap(dat[2 - idx]);
			}
			else if (FILE_BUF[file_pos] == 4)
			{
				for (unsigned int i = 2; i < 5; i++)
					indices[indexCount++] = format == binary_little_endian_format ? dat[i % 4] : LongSwap(dat[i % 4]);
				for (unsigned int i = 0; i < 3; i++)
					indices[indexCount++] = format == binary_little_endian_format ? dat[i] : LongSwap(dat[i]);
			}
			else throw std::runtime_error(__FUNCTION__);
			for (unsigned int i = indexCount - FILE_BUF[file_pos]; i < indexCount; i++)
				indices[i] = indices[i] > (unsigned int)vertexCount ? 0 : indices[i];
			file_pos += 4 * FILE_BUF[file_pos] + 1;
		}
		free(FILE_BUF);
	}
	size_t pos = istream.getPos(), size = istream.getFileSize();
	//if(pos != size)
	//	throw std::runtime_error(__FUNCTION__);

	Material defaultMat("Default_Material");
	diffuse mat;
	mat.m_reflectance = CreateTexture(Spectrum(1, 0, 0));
	defaultMat.bsdf.SetData(mat);
	Mesh::CompileMesh(&vertices[0], (unsigned int)vertices.size(), hasUV ? &texCoords[0] : 0, &indices[0], indexCount, defaultMat, Spectrum(0.0f), a_Out);
}

}
