#include "StdAfx.h"
#include "e_AnimatedMesh.h"
#include <MathTypes.h>
#include "SceneBuilder/MD5Parser.h"
#include "SceneBuilder/Importer.h"
#define TS_DEC_MD5
#include "..\Base\TangentSpace.h"
#undef TS_DEC_MD5

float4x4 mul(float4x4& a, float4x4& b)
{
	return a * b;
}

struct c_bFrame
{
	float4x4* data;

	c_bFrame(){}
	c_bFrame(bFrame* F, float4x4* a_InverseTransforms)
	{
		data = new float4x4[F->joints.size()];
		for(int i = 0; i < F->joints.size(); i++)
		{
			data[i] = mul(F->joints[i].quat.toMatrix(), float4x4::Translate(*(float3*)&F->joints[i].pos));
			if(F->joints[i].parent != -1)
				data[i] = mul(data[i], data[F->joints[i].parent]);
		}
		for(int i = 0; i < F->joints.size(); i++)
			data[i] = mul(a_InverseTransforms[i], data[i]);
	}

	~c_bFrame()
	{
		//delete data;
	}
};

struct c_Animation
{
	unsigned int m_uNumbFrames;
	unsigned int m_ubFrameRate;
	c_bFrame* data;

	c_Animation(Anim* A, MD5Model* M)
	{
		m_uNumbFrames = (unsigned int)A->numbFrames;
		m_ubFrameRate = (unsigned int)A->bFrameRate;

		float4x4* inverseJoints = new float4x4[A->baseJoints.size()];
		for(unsigned int i = 0; i < A->baseJoints.size(); i++)
			inverseJoints[i] = mul(M->joints[i].quat.toMatrix(), float4x4::Translate(*(float3*)&M->joints[i].pos)).Inverse();
		data = new c_bFrame[m_uNumbFrames];
		for(unsigned int i = 0; i < m_uNumbFrames; i++)
			data[i] = c_bFrame(&A->bFrames[i], inverseJoints);
	}

	~c_Animation()
	{
		//delete [] data;
	}
};

e_AnimatedMesh::e_AnimatedMesh(InputStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
	: e_Mesh(a_In, a_Stream0, a_Stream1, a_Stream2, a_Stream3, a_Stream4)
{
	m_uType = MESH_ANIMAT_TOKEN;
	a_In.Read(&k_Data, sizeof(k_Data));
	for(int i = 0; i < k_Data.m_uAnimCount; i++)
	{
		e_Animation A;
		A.deSerialize(a_In, a_Stream5);
		m_pAnimations.push_back(A);
	}
	m_sVertices = a_Stream5->malloc(sizeof(e_AnimatedVertex) * k_Data.m_uVertexCount);
	m_sVertices.ReadFrom(a_In);
	m_sTriangles = a_Stream5->malloc(sizeof(uint3) * m_sTriInfo.getLength());
	m_sTriangles.ReadFrom(a_In);
	m_sHierchary.deSerialize(a_In, a_Stream5);
	a_Stream5->UpdateInvalidated();
}

void e_AnimatedMesh::CompileToBinary(const char* a_InputFile, std::vector<std::string>& a_Anims, OutputStream& a_Out)
{
	MD5Model M;
	M.loadMesh(a_InputFile);
	for(unsigned int i = 0; i < a_Anims.size(); i++)
		M.loadAnim(a_Anims[i].c_str());
	AABB box;
	box = box.Identity();
	std::vector<uint3> triData2;
	e_AnimatedVertex* v_Data;
	float3* v_Pos;
	std::vector<e_TriangleData> triData;
	std::vector<e_KernelMaterial> matData;
	unsigned int off = 0;
	unsigned int vCount;
	ComputeTangentSpace(&M, &v_Data, &v_Pos, &vCount);
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	unsigned int lc = 0;
	for(int s = 0; s < M.meshes.size(); s++)
	{
		e_KernelMaterial mat;
		mat.NodeLightIndex = -1;
		diffuse ma;
		ma.m_reflectance = CreateTexture("hellknight.tga", Spectrum());
		mat.bsdf.SetData(ma);
		//mat.NormalMap = e_Sampler<float3>("n_hellknight.tga", 1);
		matData.push_back(mat);

		Mesh* sm = M.meshes[s];
		size_t st = triData2.size();

		for(int t = 0; t < sm->tris.size(); t++)
		{
			float3 P[3], N[3], Tan[3], BiTan[3];
			float2 T[3];
			for(int j = 0; j < 3; j++)
			{
				unsigned int v = sm->tris[t].v[j] + off;
				P[j] = v_Data[v].m_fVertexPos;
				T[j] = *(float2*)&sm->verts[v - off].tc;
				box.Enlarge(P[j]);
			}
			e_TriangleData d(P, s, T, N, Tan, BiTan);
			triData.push_back(d);
			uint3 q = *(uint3*)&sm->tris[t].v + make_uint3(off);
			triData2.push_back(q);
		}
		off += (unsigned int)sm->verts.size();
	}

	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lc;

	a_Out << (unsigned int)triData.size();
	a_Out.Write(&triData[0], sizeof(e_TriangleData) * (unsigned int)triData.size());
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	BVH_Construction_Result bvh;
	ConstructBVH(v_Pos, (unsigned int*)&triData2[0], (int)vCount, (int)triData2.size() * 3, a_Out, &bvh);

	e_KernelAnimatedMesh mesh;
	mesh.m_uAnimCount = M.anims.size();
	mesh.m_uJointCount = M.joints.size();
	mesh.m_uVertexCount = vCount;
	a_Out.Write(mesh);
	for(int a = 0; a < M.anims.size(); a++)
	{
		c_Animation A(M.anims[a], &M);
		std::vector<e_Frame> F;
		for(int i = 0; i < A.m_uNumbFrames; i++)
			F.push_back(e_Frame(A.data[i].data, mesh.m_uJointCount));
		e_Animation(A.m_ubFrameRate, a_Anims[a].c_str(), F).serialize(a_Out);
	}
	a_Out.Write(&v_Data[0], vCount * sizeof(e_AnimatedVertex));
	a_Out.Write(&triData2[0], (unsigned int)triData2.size() * sizeof(uint3));
	e_BVHHierarchy hier(bvh.nodes);
	hier.serialize(a_Out);
	bvh.Free();
}

void e_AnimatedMesh::CreateNewMesh(e_AnimatedMesh* A, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
{
	A->m_uType = MESH_ANIMAT_TOKEN;
	A->m_uUsedLights = 0;
	A->m_sLocalBox = m_sLocalBox;
	A->m_sMatInfo = m_sMatInfo;
	A->m_sIndicesInfo = (m_sIndicesInfo);
	A->m_sTriInfo = a_Stream1->malloc(m_sTriInfo, true);
	A->m_sNodeInfo = a_Stream2->malloc(m_sNodeInfo, true);
	A->m_sIntInfo = a_Stream0->malloc(m_sIntInfo, true);
	
	A->k_Data = k_Data;
	A->m_pAnimations = m_pAnimations;
	A->m_sVertices = m_sVertices;
	A->m_sTriangles = m_sTriangles;
	A->m_sHierchary = m_sHierchary;
}

e_BVHHierarchy::e_BVHHierarchy(e_BVHNodeData* ref)
{
	std::queue<e_BVHLevelEntry> bfs;
	bfs.push(e_BVHLevelEntry(-1,0,0, 0));
	while(!bfs.empty())
	{
		e_BVHLevelEntry n = bfs.front(); bfs.pop();
		m_pEntries.push_back(n);
		if(n.m_sNode < 0)
			continue;
		int2 d0 = ref[n.m_sNode].getChildren(), d = make_int2(d0.x < 0 ? d0.x : d0.x / 4, d0.y < 0 ? d0.y : d0.y / 4);
		bfs.push(e_BVHLevelEntry(n.m_sNode, d.x, +1, n.m_sLevel + 1));
		bfs.push(e_BVHLevelEntry(n.m_sNode, d.y, -1, n.m_sLevel + 1));
	}
	for(int i = 0; i < 32; i++)
		levels[i] = 0xffffffff;
	for(unsigned int i = 0; i < m_pEntries.size(); i++)
		levels[m_pEntries[i].m_sLevel] = std::min(levels[m_pEntries[i].m_sLevel], i);
	m_uNumLevels = m_pEntries[m_pEntries.size() - 1].m_sLevel;
	levels[m_uNumLevels] = m_pEntries.size();
}