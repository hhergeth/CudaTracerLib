#include "StdAfx.h"
#include "e_AnimatedMesh.h"

#include "SceneBuilder/MD5Parser.h"
#include "..\Base\FrameworkInterop.h"
#include "SceneBuilder/Importer.h"
#define TS_DEC_MD5
#include "..\Base\TangentSpace.h"
#undef TS_DEC_MD5

float4x4 mul(float4x4& a, float4x4& b)
{
	return a * b;
}

struct c_Frame
{
	float4x4* data;
	unsigned int L;

	c_Frame(){}
	c_Frame(Frame* F, float4x4* a_InverseTransforms)
	{
		L = F->joints.size();
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

	void serialize(OutputStream& a_Out)
	{
		a_Out.Write(data, sizeof(float4x4) * L);
	}
};

struct c_Animation
{
	unsigned int m_uNumFrames;
	unsigned int m_uNumBones;
	unsigned int m_uFrameRate;
	c_Frame* data;

	c_Animation(Anim* A, MD5Model* M)
	{
		m_uNumFrames = A->numFrames;
		m_uNumBones = A->baseJoints.size();
		m_uFrameRate = A->frameRate;

		float4x4* inverseJoints = new float4x4[m_uNumBones];
		for(int i = 0; i < m_uNumBones; i++)
			inverseJoints[i] = mul(M->joints[i].quat.toMatrix(), float4x4::Translate(*(float3*)&M->joints[i].pos)).Inverse();
		data = new c_Frame[m_uNumFrames];
		for(int i = 0; i < m_uNumFrames; i++)
			data[i] = c_Frame(&A->frames[i], inverseJoints);
	}

	void serialize(OutputStream& a_Out)
	{
		for(int i = 0; i < m_uNumFrames; i++)
			data[i].serialize(a_Out);
	}
};

void constructLayout(std::vector<std::vector<e_BVHLevelEntry>>& a_Out, e_BVHNodeData* a_Data, int node, int parent, unsigned int level = 0, unsigned int side = 0)
{
	if(a_Out.size() <= level)
		a_Out.push_back(std::vector<e_BVHLevelEntry>());
	a_Out[level].push_back(e_BVHLevelEntry(parent, node, side));
	if(node >= 0)
	{
		int2 d = a_Data[node / 4].getChildren();
		constructLayout(a_Out, a_Data, d.x, node, level + 1, -1);
		constructLayout(a_Out, a_Data, d.y, node, level + 1, +1);
	}
}

e_AnimatedMesh::e_AnimatedMesh(InputStream& a_In, e_DataStream<e_TriIntersectorData>* a_Stream0, e_DataStream<e_TriangleData>* a_Stream1, e_DataStream<e_BVHNodeData>* a_Stream2, e_DataStream<int>* a_Stream3, e_DataStream<e_KernelMaterial>* a_Stream4, e_DataStream<char>* a_Stream5)
	: e_Mesh(a_In, a_Stream0, a_Stream1, a_Stream2, a_Stream3, a_Stream4)
{
	BASEHOST = a_Stream5->getHost(0);
	BASEDEVICE = a_Stream5->getDevice(0);
	unsigned int numVertices;
	a_In >> numVertices;
	unsigned int BLOCKSIZE;
	a_In >> BLOCKSIZE;
	unsigned int OFF = 0;
	m_pOffset = a_Stream5->malloc(BLOCKSIZE);
	char* DATA = m_pOffset();
#define APPEND(s) { a_In.Read(DATA + OFF, s); OFF += s; /*while(OFF % 16) OFF++;*/ }
	a_In >> k_Data.m_uVertexCount;
	m_pVertices = (e_AnimatedVertex*)(DATA + OFF);
	APPEND(k_Data.m_uVertexCount * sizeof(e_AnimatedVertex))
	a_In >> k_Data.m_uTriangleCount;
	k_Data.m_uTriDataOffset = OFF;
	m_pTriangles = (uint3*)(DATA + OFF);
	APPEND(k_Data.m_uTriangleCount * sizeof(uint3))
	k_Data.m_uBVHLevelOffset = OFF;
	a_In >> k_Data.m_uBVHLevelCount;
	m_pLevels = (int2*)(DATA + OFF);
	APPEND(k_Data.m_uBVHLevelCount * sizeof(uint2))
	m_pLevelEntries = (e_BVHLevelEntry*)(DATA + OFF);
	for(unsigned int l = 0; l < k_Data.m_uBVHLevelCount; l++)
		APPEND(m_pLevels[l].y * sizeof(e_BVHLevelEntry));
	a_In >> k_Data.m_uAnimCount;
	a_In >> k_Data.m_uJointCount;
	k_Data.m_uAnimHeaderOffset = OFF;
	m_pAnimations = (e_Animation*)(DATA + OFF);
	APPEND(k_Data.m_uAnimCount * sizeof(e_Animation))
	k_Data.m_uAnimBodyOffset = OFF;
	m_pAnimData = (float4x4*)(DATA + OFF);
	e_Animation* lA = m_pAnimations + k_Data.m_uAnimCount - 1;
	unsigned int numB = lA->m_uDataOffset + lA->m_uNumFrames * k_Data.m_uJointCount;
	APPEND(sizeof(float4x4) * numB)
	a_Stream5->Invalidate(DataStreamRefresh_Buffered, m_pOffset);
}

void e_AnimatedMesh::CompileToBinary(char* a_InputFile, c_StringArray& a_Anims, OutputStream& a_Out)
{
	MD5Model M;
	M.loadMesh(a_InputFile);
	for(int i = 0; i < a_Anims.data.size(); i++)
		M.loadAnim(a_Anims.data[i]);
	AABB box;
	box = box.Identity();
	std::vector<uint3> triData2;
	e_AnimatedVertex* v_Data;
	float3* v_Pos;
	std::vector<e_TriangleData> triData;
	//std::vector<e_Material> matData;
	FW::Mesh<FW::VertexP> M2;
	unsigned int off = 0;
	unsigned int vCount;
	ComputeTangentSpace(&M, &v_Data, &v_Pos, &vCount);
	for(int s = 0; s < M.meshes.size(); s++)
	{/*
		e_Material mat;
		float f = 0.7;
		matData.push_back(mat);*/

		int s2 = M2.addSubmesh();
		Mesh* sm = M.meshes[s];
		M2.addVertices((FW::VertexP*)(v_Pos + off), sm->verts.size());
		int st = triData2.size();

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
		M2.mutableIndices(s2).add((FW::Vec3i*)&triData2[st], sm->tris.size());
		off += sm->verts.size();
	}
	M2.compact();
	a_Out << box;
	a_Out << (unsigned int)triData.size();
	a_Out.Write(&triData[0], sizeof(e_TriangleData) * triData.size());throw 1;
	//a_Out << (unsigned int)matData.size();
	//a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * matData.size());
	float4* v_BVH;
	ConstructBVH(M2, TmpOutStream(&a_Out), &v_BVH);

	std::vector<std::vector<e_BVHLevelEntry>> V;
	constructLayout(V, (e_BVHNodeData*)v_BVH, 0, -1);

	a_Out << (unsigned int)vCount;
	unsigned int BLOCKSIZE = 4 + vCount * sizeof(e_AnimatedVertex) + 4 + triData2.size() * sizeof(uint3)
		+ 4 + V.size() * 8
		+ 8 + M.anims.size() * 12;
	for(int i = 0; i < V.size(); i++)
		BLOCKSIZE += V[i].size() * sizeof(e_BVHLevelEntry);
	for(int a = 0; a < M.anims.size(); a++)
		BLOCKSIZE += M.anims[a]->numFrames * (M.numJoints * sizeof(float4x4));
	a_Out << BLOCKSIZE;
	unsigned int endTo = BLOCKSIZE + a_Out.numBytesWrote;

	a_Out << (unsigned int)vCount;
	a_Out.Write(&v_Data[0], vCount * sizeof(e_AnimatedVertex));
	a_Out << (unsigned int)triData2.size();
	a_Out.Write(&triData2[0], triData2.size() * sizeof(uint3));
	
	a_Out << (unsigned int)V.size();
	unsigned int off2 = 0;
	for(int i = 0; i < V.size(); i++)
	{
		a_Out << off2;
		a_Out << (unsigned int)V[i].size();
		off2 += V[i].size();
	}
	for(int i = 0; i < V.size(); i++)
		a_Out.Write(&V[i][0], V[i].size() * sizeof(e_BVHLevelEntry));
	
	a_Out << (unsigned int)M.anims.size();
	a_Out << (unsigned int)M.anims[0]->jointInfo.size();
	unsigned int OFF4 = 0;
	for(int a = 0; a < M.anims.size(); a++)
	{
		a_Out << M.anims[a]->numFrames;
		a_Out << M.anims[a]->frameRate;
		a_Out << OFF4;
		OFF4 += M.anims[a]->numFrames * M.joints.size();
	}
	for(int a = 0; a < M.anims.size(); a++)
	{
		c_Animation ANIM(M.anims[a], &M);
		ANIM.serialize(a_Out);
	}
	if(a_Out.numBytesWrote != endTo)
	{
		OutputDebugString("Count error");
		throw 1;
	}
}