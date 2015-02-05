#include <StdAfx.h>
#include "..\e_Mesh.h"
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
#include "..\..\Base\StringUtils.h"
#include "..\e_Mesh.h"
#include "..\e_TriangleData.h"
#include "tiny_obj_loader.h"

e_KernelMaterial cnvMat(tinyobj::material_t M, e_MeshPartLight* lights, unsigned int* lightIndex)
{
	e_KernelMaterial mat(M.name.c_str());/*
	float f = 0.0f;
	if(M.IlluminationModel == 2)
	{
		diffuse d;
		d.m_reflectance = CreateTexture(M.textures[0].c_str(), Spectrum(M.diffuse));
		mat.bsdf.SetData(d);
	}
	else if(M.IlluminationModel == 5)
	{
		mat.bsdf.SetData(conductor(Spectrum(0.0f), Spectrum(1.0f)));
	}
	else if(M.IlluminationModel == 7)
	{
		dielectric d;
		d.m_eta = M.IndexOfRefraction;
		d.m_invEta = 1.0f / M.IndexOfRefraction;
		d.m_specularReflectance = CreateTexture(0, Spectrum(M.specular));
		d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf));
		mat.bsdf.SetData(d);
	}
	else if(M.IlluminationModel == 9)
	{
		dielectric d;
		d.m_eta = M.IndexOfRefraction;
		d.m_invEta = 1.0f / M.IndexOfRefraction;
		d.m_specularReflectance = CreateTexture(0, Spectrum(0.0f));
		d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf));
		mat.bsdf.SetData(d);
	}*/
	return mat;
}

void compileobj3(const char* a_InputFile, OutputStream& a_Out)
{
	std::vector<tinyobj::shape_t> shapes;
	std::string err = tinyobj::LoadObj(shapes, a_InputFile, getDirName(a_InputFile).c_str());

	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
	size_t numTriangles = 0, numVertices = 0;
	size_t numMaxV = 0;
	for(size_t i = 0; i < shapes.size(); i++)
	{
		numTriangles += shapes[i].mesh.indices.size() / 3;
		numVertices += shapes[i].mesh.positions.size() / 3;
		numMaxV = max(numMaxV, (size_t)shapes[i].mesh.indices.size());
	}

	std::vector<Vec3f> positions;
	positions.resize(numVertices);
	std::vector<unsigned int> indices;
	indices.resize(numTriangles * 3);
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	Platform::SetMemory(m_sLights, sizeof(m_sLights));
	unsigned int lightCount = 0;
	std::vector<e_KernelMaterial> matData;
	e_TriangleData* triData = new e_TriangleData[numTriangles];
	unsigned int triIndex = 0;
#ifdef EXT_TRI
	Vec3f* v_Normals = new Vec3f[numMaxV], *v_Tangents = new Vec3f[numMaxV], *v_BiTangents = new Vec3f[numMaxV];
#endif
	size_t posIndex = 0, indIndex = 0;
	AABB box = AABB::Identity();
	for(size_t i = 0; i < shapes.size(); i++)
	{
		tinyobj::shape_t& S = shapes[i];

		//create bvh construction info
		for(unsigned int j = 0; j < S.mesh.positions.size() / 3; j++)
		{
			positions[posIndex + j] = Vec3f(S.mesh.positions[j * 3 + 0], S.mesh.positions[j * 3 + 1], S.mesh.positions[j * 3 + 2]);
			box.Enlarge(positions[posIndex + j]);
		}
		for(unsigned int j = 0; j < S.mesh.indices.size(); j++)
			indices[indIndex + j] = (unsigned int)posIndex + S.mesh.indices[j];
		Vec3f* P = &positions[posIndex];
		posIndex += S.mesh.positions.size() / 3;
		indIndex += S.mesh.indices.size();

		//convert mat
		matData.push_back(cnvMat(S.material, m_sLights, &lightCount));
		
		//build tri data
		Vec2f* T = S.mesh.texcoords.size() ? (Vec2f*)&S.mesh.texcoords[0] : 0;
#ifdef EXT_TRI
		Platform::SetMemory(v_Normals, sizeof(Vec3f) * numMaxV);
		Platform::SetMemory(v_Tangents, sizeof(Vec3f) * numMaxV);
		Platform::SetMemory(v_BiTangents, sizeof(Vec3f) * numMaxV);
		ComputeTangentSpace(P, T, &S.mesh.indices[0], (unsigned int)S.mesh.positions.size(), (unsigned int)S.mesh.indices.size() / 3, v_Normals, v_Tangents, v_BiTangents);
#endif
		for(size_t ti = 0; ti < S.mesh.indices.size() / 3; ti++)
		{
			for(size_t j = 0; j < 3; j++)
			{
				const int l = S.mesh.indices[ti * 3 + j];
				p[j] = P[l];
#ifdef EXT_TRI
				if(T)
					t[j] = T[l];
				ta[j] = normalize(v_Tangents[l]);
				bi[j] = normalize(v_BiTangents[l]);
				n[j] = normalize(v_Normals[l]);
				//n[j] = v.n;
#endif
			}
			triData[triIndex++] = e_TriangleData(p, (unsigned char)i,t, n, ta, bi);
		}
	}

	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lightCount;
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * (unsigned int)numTriangles);
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	ConstructBVH(&positions[0], &indices[0], (unsigned int)numVertices, (unsigned int)numTriangles * 3, a_Out);
#ifdef EXT_TRI
	delete [] v_Normals;
	delete [] v_Tangents;
	delete [] v_BiTangents;
#endif
	delete [] triData;
}