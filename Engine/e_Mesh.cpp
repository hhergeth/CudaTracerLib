#include "StdAfx.h"
#include "e_Mesh.h"
#include "..\Base\FrameworkInterop.h"
#define TS_DEC_FRAMEWORK
#include "..\Base\TangentSpace.h"
#undef TS_DEC_FRAMEWORK
#include <Windows.h>
#include "e_Volumes.h"
#include "e_Light.h"
#include "SceneBuilder\Importer.h"

e_Mesh::e_Mesh(InputStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<int>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4)
{
	m_uType = MESH_STATIC_TOKEN;
	int abc = sizeof(e_TriangleData), abc2 = sizeof(m_sLights);

	a_In >> m_sLocalBox;
	a_In.Read(m_sLights, sizeof(m_sLights));
	a_In >> m_uUsedLights;

	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	m_sTriInfo = a_Stream1->malloc(m_uTriangleCount);
	a_In.Read(m_sTriInfo(0), m_sTriInfo.getSizeInBytes());
	m_sTriInfo.Invalidate();

	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	m_sMatInfo = a_Stream4->malloc(m_uMaterialCount);
	a_In.Read(m_sMatInfo(0), m_sMatInfo.getSizeInBytes());
	m_sMatInfo.Invalidate();

	a_In >> m_uMaterialCount;
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	m_sNodeInfo = a_Stream2->malloc(m_uNodeSize / 64);
	a_In.Read(m_sNodeInfo(0), m_uNodeSize);
	m_sNodeInfo.Invalidate();

	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	float C = ceil((float)m_uIntSize / 48.0f);
	m_sIntInfo = a_Stream0->malloc((int)C);
	a_In.Read(m_sIntInfo(0), m_uIntSize);
	m_sIntInfo.Invalidate();

	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	m_sIndicesInfo = a_Stream3->malloc(m_uIndicesSize / 4);
	a_In.Read(m_sIndicesInfo(0), m_sIndicesInfo.getSizeInBytes());
	m_sIndicesInfo.Invalidate();
	
	createKernelData();
}

e_SceneInitData e_Mesh::ParseBinary(const char* a_InputFile)
{
	InputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	a_In.Move(sizeof(e_MeshPartLight) * MAX_AREALIGHT_NUM + 8);
#define PRINT(n, t) { a_In.Move<t>(n); char msg[255]; msg[0] = 0; sprintf(msg, "Buf : %s, length : %d, size : %d[MB]\n", #t, (n), (n) * sizeof(t) / (1024 * 1024)); OutputDebugString(msg);}
#define PRINT2(n, t) { a_In.Move(n); char msg[255]; msg[0] = 0; sprintf(msg, "Buf : %s, length : %d, size : %d[MB]\n", #t, (n) / sizeof(t), (n) / (1024 * 1024)); OutputDebugString(msg);}
	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	PRINT(m_uTriangleCount, e_TriangleData)
	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	PRINT(m_uMaterialCount, e_KernelMaterial)
	a_In >> m_uMaterialCount;
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	PRINT(m_uNodeSize / 64, e_BVHNodeData)
	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	PRINT2(m_uIntSize, e_TriIntersectorData)
	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	PRINT(m_uIndicesSize / 4, int)
#undef PRINT
#undef PRINT2
	a_In.Close();
	char msg[2048];
	sprintf(msg, "return CreateForSpecificMesh(%d, %d, %d, %d, 255, a_Lights);\n", m_uTriangleCount, (int)ceilf((float)m_uIntSize / 48.0f), m_uNodeSize / 64, m_uIndicesSize / 4);
	OutputDebugString(msg);
	return e_SceneInitData::CreateForSpecificMesh(m_uTriangleCount, (int)ceilf((float)m_uIntSize / 48.0f), m_uNodeSize / 64, m_uIndicesSize / 4, 255, 128);
}

void e_Mesh::Free(e_Stream<e_TriIntersectorData>& a_Stream0, e_Stream<e_TriangleData>& a_Stream1, e_Stream<e_BVHNodeData>& a_Stream2, e_Stream<int>& a_Stream3, e_Stream<e_KernelMaterial>& a_Stream4)
{
	a_Stream0.dealloc(m_sIntInfo);
	a_Stream1.dealloc(m_sTriInfo);
	a_Stream2.dealloc(m_sNodeInfo);
	a_Stream3.dealloc(m_sIndicesInfo);
	a_Stream4.dealloc(m_sMatInfo);
}

void e_Mesh::CompileObjToBinary(const char* a_InputFile, OutputStream& a_Out)
{
	FW::Mesh<FW::VertexPNT>* MB = new FW::Mesh<FW::VertexPNT>(*FW::importMesh(FW::String(a_InputFile)));
	unsigned int m_numTriangles = MB->numTriangles();
	unsigned int m_numVertices = MB->numVertices();

	e_TriangleData* triData = new e_TriangleData[m_numTriangles];
	std::vector<e_KernelMaterial> matData;
	float3 p[3];
	float3 n[3];
	float3 ta[3];
	float3 bi[3];
	float2 t[3];
#ifdef EXT_TRI
	float3* v_Normals = new float3[m_numVertices], *v_Tangents = new float3[m_numVertices], *v_BiTangents = new float3[m_numVertices];
	ZeroMemory(v_Normals, sizeof(float3) * m_numVertices);
	ZeroMemory(v_Tangents, sizeof(float3) * m_numVertices);
	ZeroMemory(v_BiTangents, sizeof(float3) * m_numVertices);
	ComputeTangentSpace(MB, v_Normals, v_Tangents, v_BiTangents);
#endif
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	int c = 0, lc = 0;
	for (int submesh = 0; submesh < MB->numSubmeshes(); submesh++)
	{
		FW::MeshBase::Material& M = MB->material(submesh);
		e_KernelMaterial mat(M.Name.getPtr());
		float f = 0.0f;
		if(M.IlluminationModel == 2)
		{
			diffuse d;
			d.m_reflectance = CreateTexture(M.textures[0].getID().getPtr(), Spectrum(M.diffuse));
			mat.bsdf.SetData(d);
		}
		else if(M.IlluminationModel == 5)
		{
			dielectric d;
			d.m_invEta = d.m_eta = 1;
			d.m_specularReflectance = CreateTexture(0, Spectrum(M.specular));
			d.m_specularTransmittance = CreateTexture(0, Spectrum(0.0f));
			mat.bsdf.SetData(d);
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
		}
		if(length(M.emission) != 0)
		{
			ZeroMemory(m_sLights + lc, sizeof(e_MeshPartLight));
			m_sLights[lc].L = M.emission;
			strcpy(m_sLights[lc].MatName, M.Name.getPtr());
			mat.NodeLightIndex = lc++;
		}
		if(0&&M.textures[3].getID().getLength())
		{
			char* c = new char[M.textures[3].getID().getLength()+1];
			ZeroMemory(c, M.textures[3].getID().getLength()+1);
			memcpy(c, M.textures[3].getID().getPtr(), M.textures[3].getID().getLength());
			OutputDebugString(c);
			OutputDebugString(M.textures[3].getID().getPtr());
			mat.SetHeightMap(CreateTexture(c, make_float3(0)));
			mat.HeightScale = 0.5f;
		}
		/*
#ifdef EXT_TRI
		if(M.textures[0].hasData())
			memcpy(m.m_cDiffusePath, M.textures[0].getID().getPtr(), M.textures[0].getID().getLength());
		if(M.textures[3].hasData())
			memcpy(m.m_cNormalPath, M.textures[3].getID().getPtr(), M.textures[3].getID().getLength());
#endif
		*/
		matData.push_back(mat);
		size_t matIndex = matData.size() - 1;

		const FW::Array<FW::Vec3i>& indices = MB->indices(submesh);
		for (size_t i = 0; i < indices.getSize(); i++)
		{
			const FW::Vec3i& vi = indices[(int)i];
			for(size_t j = 0; j < 3; j++)
			{
				const int l = vi.get((int)j);
				const FW::VertexPNT& v = MB[0][l];
				p[j] = make_float3(v.p.x, v.p.y, v.p.z);
#ifdef EXT_TRI
				t[j] = make_float2(v.t.x, v.t.y);
				ta[j] = v_Tangents[l];
				bi[j] = v_BiTangents[l];
				n[j] = v_Normals[l];
				//n[j] = v.n;
#endif
			}
			triData[c++] = e_TriangleData(p, (unsigned char)matIndex,t, n, ta, bi);
		}
	}
	FW::Vec3f a, b;
	MB->getBBox(a, b);
	AABB box;
	*(FW::Vec3f*)&box.minV = a;
	*(FW::Vec3f*)&box.maxV = b;

	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lc;

	a_Out << m_numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * m_numTriangles);
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	TmpOutStream to(&a_Out);
	ConstructBVH2(MB, to);

	delete [] v_Normals;
	delete [] v_Tangents;
	delete [] v_BiTangents;
	delete [] triData;
	delete MB;
}