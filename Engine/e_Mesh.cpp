#include "StdAfx.h"
#include "e_Mesh.h"
#include "..\Base\FrameworkInterop.h"
#define TS_DEC_FRAMEWORK
#include "..\Base\TangentSpace.h"
#undef TS_DEC_FRAMEWORK
#include <Windows.h>
#include "e_Volumes.h"
#include "e_Light.h"

const unsigned int e_KernelMaterial_Glass::TYPE = e_KernelMaterial_Glass_TYPE;
const unsigned int e_KernelMaterial_Matte::TYPE = e_KernelMaterial_Matte_TYPE;
const unsigned int e_KernelMaterial_Mirror::TYPE = e_KernelMaterial_Mirror_TYPE;
const unsigned int e_KernelMaterial_Metal::TYPE = e_KernelMaterial_Metal_TYPE;
const unsigned int e_KernelMaterial_ShinyMetal::TYPE = e_KernelMaterial_ShinyMetal_TYPE;
const unsigned int e_KernelMaterial_Plastic::TYPE = e_KernelMaterial_Plastic_TYPE;
const unsigned int e_KernelMaterial_Substrate::TYPE = e_KernelMaterial_Substrate_TYPE;

const unsigned int e_HomogeneousVolumeDensity::TYPE = e_HomogeneousVolumeDensity_TYPE;
const unsigned int e_SphereVolumeDensity::TYPE = e_SphereVolumeDensity_TYPE;

const unsigned int e_PointLight::TYPE = e_PointLight_TYPE;
const unsigned int e_DiffuseLight::TYPE = e_DiffuseLight_TYPE;
const unsigned int e_DistantLight::TYPE = e_DistantLight_TYPE;
const unsigned int e_SpotLight::TYPE = e_SpotLight_TYPE;

#include "niflib.h"
#include "obj\NiObject.h"
#include "obj\NiTriShape.h"
#include "obj\NiSkinInstance.h"
#include "obj\NiTriShapeData.h"
#include "obj\NiBone.h"
#include "obj\NiSkinPartition.h"
#include "obj\NiTriStripsData.h"
#include "obj\BSFadeNode.h"
#include "obj\NiTriStrips.h"
#include "obj\BSLodTriShape.h"
#include "obj\BSLightingShaderProperty.h"
#include "obj\BSShaderTextureSet.h"
#include "obj\NiAlphaProperty.h"

using namespace Niflib;

#include "SceneBuilder\Importer.h"

e_Mesh::e_Mesh(InputStream& a_In, e_DataStream<e_TriIntersectorData>* a_Stream0, e_DataStream<e_TriangleData>* a_Stream1, e_DataStream<e_BVHNodeData>* a_Stream2, e_DataStream<int>* a_Stream3, e_DataStream<e_KernelMaterial>* a_Stream4)
{
	a_In >> m_sLocalBox;

	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	m_sTriInfo = a_Stream1->malloc(m_uTriangleCount);
	a_In.Read(m_sTriInfo(0), m_sTriInfo.getSizeInBytes());
	a_Stream1->Invalidate(DataStreamRefresh_Buffered, m_sTriInfo);

	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	m_sMatInfo = a_Stream4->malloc(m_uMaterialCount);
	a_In.Read(m_sMatInfo(0), m_sMatInfo.getSizeInBytes());
	a_Stream4->Invalidate(DataStreamRefresh_Buffered, m_sMatInfo);

	a_In >> m_uMaterialCount;
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	m_sNodeInfo = a_Stream2->malloc(m_uNodeSize / 64);
	a_In.Read(m_sNodeInfo(0), m_uNodeSize);
	a_Stream2->Invalidate(DataStreamRefresh_Buffered, m_sNodeInfo);

	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	float C = ceil((float)m_uIntSize / 48.0f);
	m_sIntInfo = a_Stream0->malloc((int)C);
	a_In.Read(m_sIntInfo(0), m_uIntSize);
	a_Stream0->Invalidate(DataStreamRefresh_Buffered, m_sIntInfo);

	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	m_sIndicesInfo = a_Stream3->malloc(m_uIndicesSize / 4);
	a_In.Read(m_sIndicesInfo(0), m_sIndicesInfo.getSizeInBytes());
	a_Stream3->Invalidate(DataStreamRefresh_Buffered, m_sIndicesInfo);
	
	createKernelData();
}

e_SceneInitData e_Mesh::ParseBinary(const char* a_InputFile)
{
	InputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
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

void e_Mesh::Free(e_DataStream<e_TriIntersectorData>& a_Stream0, e_DataStream<e_TriangleData>& a_Stream1, e_DataStream<e_BVHNodeData>& a_Stream2, e_DataStream<int>& a_Stream3, e_DataStream<e_KernelMaterial>& a_Stream4)
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
	int c = 0;
	for (int submesh = 0; submesh < MB->numSubmeshes(); submesh++)
	{
		FW::MeshBase::Material& M = MB->material(submesh);
		e_KernelMaterial mat(M.Name.getPtr());
		float f = 0.0f;
		if(M.IlluminationModel == 2)
		{
			mat.SetData(e_KernelMaterial_Matte(e_Sampler<float3>(!M.diffuse), e_Sampler<float>(f)));
		}
		else if(M.IlluminationModel == 5)
		{
			mat.SetData(e_KernelMaterial_Mirror(e_Sampler<float3>(M.specular)));
		}
		else if(M.IlluminationModel == 7)
		{
			mat.SetData(e_KernelMaterial_Glass(e_Sampler<float3>(M.specular), e_Sampler<float3>(M.Tf), e_Sampler<float>(M.IndexOfRefraction)));
		}
		else if(M.IlluminationModel == 9)
		{
			mat.SetData(e_KernelMaterial_Glass(e_Sampler<float3>(make_float3(0)), e_Sampler<float3>(M.Tf), e_Sampler<float>(M.IndexOfRefraction)));
		}
		mat.Emission = M.emission;
		/*
#ifdef EXT_TRI
		if(M.textures[0].hasData())
			memcpy(m.m_cDiffusePath, M.textures[0].getID().getPtr(), M.textures[0].getID().getLength());
		if(M.textures[3].hasData())
			memcpy(m.m_cNormalPath, M.textures[3].getID().getPtr(), M.textures[3].getID().getLength());
#endif
		*/
		matData.push_back(mat);
		unsigned int matIndex = matData.size() - 1;

		const FW::Array<FW::Vec3i>& indices = MB->indices(submesh);
		for (int i = 0; i < indices.getSize(); i++)
		{
			const FW::Vec3i& vi = indices[i];
			for(int j = 0; j < 3; j++)
			{
				const unsigned int l = vi.get(j);
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
			triData[c++] = e_TriangleData(p, matIndex,t, n, ta, bi);
		}
	}
	FW::Vec3f a, b;
	MB->getBBox(a, b);
	AABB box;
	*(FW::Vec3f*)&box.minV = a;
	*(FW::Vec3f*)&box.maxV = b;

	a_Out << box;
	a_Out << m_numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * m_numTriangles);
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * matData.size());
	TmpOutStream to(&a_Out);
	ConstructBVH2(MB, to);
}

static float4x4 g_mTransMat = float4x4::Identity();//float4x4::RotateX(-PI/2);
float3 toFloat3(Vector3 v, Matrix44& mat, float w = 0.0f)
{
	v = mat * v;
	if(w == 0)
		v -= mat.GetTranslation();
	return *(float3*)&v;
}
void e_Mesh::CompileNifToBinary(const char* a_InputFile, OutputStream& a_Out)
{
	std::string str(a_InputFile);
	AABB box = AABB::Identity();
	std::vector<e_TriangleData> triData;
	std::vector<e_KernelMaterial> matData;
	e_KernelMaterial defaultMat;
	matData.push_back(defaultMat);
	FW::Mesh<FW::VertexP> M2;
	unsigned int off = 0;
	std::vector<Ref<NiObject>> Root = ReadNifList(str);
	for(int i = 0; i < Root.size(); i++)
	{
		Ref<NiObject> c = Root.at(i);

		bool newMat = false;
		int matIndex = 0;/*
		e_Material ma;
		if(c->IsDerivedType(NiGeometry::TYPE))
			for(int i = 0; i < 2; i++)
			{
				NiProperty* p = ((NiGeometry*)c.operator Niflib::NiObject *())->GetBSProperty(i);
				if(p && p->IsDerivedType(BSLightingShaderProperty::TYPE))
				{
					BSLightingShaderProperty* prop2 = (BSLightingShaderProperty*)p;
					Ref<BSShaderTextureSet> set = prop2->GetTextureSet();
					if(set->GetTexture(0).at(0) == 8)
						continue;
					newMat = true;
					ma.m_cEmission = *(float3*)&prop2->GetEmissiveColor() * prop2->GetEmissiveMultiple();
					//ma.m_cDiffuseColor = make_float4(0,0,0,prop2->GetAlpha());
					if(set->GetTexture(0).length())
						memcpy(ma.m_cDiffusePath, set->GetTexture(0).c_str(), set->GetTexture(0).length());
					if(set->GetTexture(1).length() && strstr(set->GetTexture(1).c_str(), ".dds"))
						memcpy(ma.m_cNormalPath, set->GetTexture(1).c_str(), set->GetTexture(1).length());
				}
				else if(p && p->IsDerivedType(NiAlphaProperty::TYPE))
				{
					NiAlphaProperty* prop2 = (NiAlphaProperty*)p;
					unsigned short flags = prop2->GetFlags();
					unsigned char thresh = prop2->GetTestThreshold();
					if(flags & (1 << 9))
						ma.m_fAlphaThreshold = float(thresh) / 255.0f;
				}
			}
		if(newMat)
		{
			matIndex = matData.size();
			matData.push_back(ma);
		}*/


#pragma region
		if(c->IsDerivedType(BSLODTriShape::TYPE))
		{
			int s2 = M2.addSubmesh();
			BSLODTriShape* c2 = (BSLODTriShape*)c.operator Niflib::NiObject *();
			Matrix44 mat = c2->GetWorldTransform();
			NiTriShapeData* G = (NiTriShapeData*)c2->GetData().operator Niflib::NiGeometryData *();
			std::vector<Vector3> V = G->GetVertices();
			std::vector<float3> V2;
			for(int v = 0; v < V.size(); v++)
				V2.push_back(toFloat3(V[v], mat, 1.0f));
			M2.addVertices((FW::VertexP*)&V2[0], V2.size());
			std::vector<Triangle> Ts = G->GetTriangles();
			std::vector<Vector3> Ns = G->GetNormals(), Tans = G->GetTangents(), BiTans = G->GetBitangents();
			std::vector<TexCoord> Texs;
			bool hasUV = G->GetUVSetCount(), hasTan = Tans.size(), hasBi = BiTans.size();
			if(hasUV)
				Texs = G->GetUVSet(0);
			for(int j = 0; j < Ts.size(); j++)
			{
					float3 P[3], N[3], Tan[3], BiTan[3];
					float2 T[3];
					for(int k = 0; k < 3; k++)
					{
						unsigned int v = Ts[j][k];
						P[k] = V2[v];
						N[k] = toFloat3(Ns[v], mat);
						if(hasUV)
							T[k] = make_float2(Texs[v].u, Texs[v].v);
						if(hasTan)
							Tan[k] = *(float3*)&Tans[v];
						if(hasBi)
							BiTan[k] = *(float3*)&BiTans[v];
						box.Enlarge(P[k]);
					}
					M2.mutableIndices(s2).add(FW::Vec3i(Ts[j].v1, Ts[j].v2, Ts[j].v3) + off);
					e_TriangleData d(P, matIndex, T, N, Tan, BiTan);
				
					triData.push_back(d);
			}
			off += V.size();
		}
#pragma endregion
#pragma region
		else if(c->IsDerivedType(NiTriStrips::TYPE))
		{
			int s2 = M2.addSubmesh();
			NiTriStrips* c2 = (NiTriStrips*)c.operator Niflib::NiObject *();
			Matrix44 mat = c2->GetWorldTransform();
			NiTriStripsData* G = (NiTriStripsData*)c2->GetData().operator Niflib::NiGeometryData *();
			std::vector<Vector3> V = G->GetVertices();
			std::vector<float3> V2;
			for(int v = 0; v < V.size(); v++)
				V2.push_back(toFloat3(V[v], mat, 1.0f));
			M2.addVertices((FW::VertexP*)&V2[0], V2.size());
			std::vector<Triangle> Ts = G->GetTriangles();
			std::vector<Vector3> Ns = G->GetNormals(), Tans = G->GetTangents(), BiTans = G->GetBitangents();
			std::vector<TexCoord> Texs;
			bool hasUV = G->GetUVSetCount(), hasTan = Tans.size(), hasBi = BiTans.size();
			if(hasUV)
				Texs = G->GetUVSet(0);
			for(int j = 0; j < Ts.size(); j++)
			{
					float3 P[3], N[3], Tan[3], BiTan[3];
					float2 T[3];
					for(int k = 0; k < 3; k++)
					{
						unsigned int v = Ts[j][k];
						P[k] = V2[v];
						N[k] = toFloat3(Ns[v], mat);
						if(hasUV)
							T[k] = make_float2(Texs[v].u, Texs[v].v);
						if(hasTan)
							Tan[k] = *(float3*)&Tans[v];
						if(hasBi)
							BiTan[k] = *(float3*)&BiTans[v];
						box.Enlarge(P[k]);
					}
					M2.mutableIndices(s2).add(FW::Vec3i(Ts[j].v1, Ts[j].v2, Ts[j].v3) + off);
					e_TriangleData d(P, matIndex, T, N, Tan, BiTan);
				
					triData.push_back(d);
			}
			off += V.size();
		}
#pragma endregion
#pragma region
		else if(c->IsDerivedType(NiTriShape::TYPE))
		{
			int s2 = M2.addSubmesh();
			NiTriShape* c2 = (NiTriShape*)c.operator Niflib::NiObject *();
			Matrix44 mat = c2->GetWorldTransform();
			NiTriShapeData* G = (NiTriShapeData*)c2->GetData().operator Niflib::NiGeometryData *();
			std::vector<Vector3> V = G->GetVertices();
			std::vector<float3> V2;
			for(int v = 0; v < V.size(); v++)
				V2.push_back(toFloat3(V[v], mat, 1.0f));
			M2.addVertices((FW::VertexP*)&V2[0], V2.size());
			std::vector<Triangle> Ts = G->GetTriangles();
			std::vector<Vector3> Ns = G->GetNormals(), Tans = G->GetTangents(), BiTans = G->GetBitangents();
			std::vector<TexCoord> Texs;
			bool hasUV = G->GetUVSetCount(), hasTan = Tans.size(), hasBi = BiTans.size();
			if(hasUV)
				Texs = G->GetUVSet(0);
			for(int j = 0; j < Ts.size(); j++)
			{
					float3 P[3], N[3], Tan[3], BiTan[3];
					float2 T[3];
					for(int k = 0; k < 3; k++)
					{
						unsigned int v = Ts[j][k];
						P[k] = V2[v];
						N[k] = toFloat3(Ns[v], mat);
						if(hasUV)
							T[k] = make_float2(Texs[v].u, Texs[v].v);
						if(hasTan)
							Tan[k] = *(float3*)&Tans[v];
						if(hasBi)
							BiTan[k] = *(float3*)&BiTans[v];
						box.Enlarge(P[k]);
					}
					M2.mutableIndices(s2).add(FW::Vec3i(Ts[j].v1, Ts[j].v2, Ts[j].v3) + off);
					e_TriangleData d(P, matIndex, T, N, Tan, BiTan);
				
					triData.push_back(d);
			}
			off += V.size();
		}
#pragma endregion
	}
	M2.compact();
	a_Out << box;
	a_Out << (unsigned int)triData.size();
	a_Out.Write(&triData[0], sizeof(e_TriangleData) * triData.size());
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * matData.size());
	ConstructBVH2(&M2, TmpOutStream(&a_Out));
}