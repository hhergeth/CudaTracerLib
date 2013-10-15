#pragma once

#include <MathTypes.h>
#include "e_TriangleData.h"
#include "e_Material.h"
#include "e_Buffer.h"
#include "..\Base\BVHBuilder.h"

#define MAX_AREALIGHT_NUM 4

struct e_KernelMesh
{
	unsigned int m_uTriangleOffset;
	unsigned int m_uBVHNodeOffset;
	unsigned int m_uBVHTriangleOffset;
	unsigned int m_uBVHIndicesOffset;
	unsigned int m_uStdMaterialOffset;
};

struct e_TriIntersectorData
{
//      triWoop[triOfs*16 + 0 ] = Vec4f(woopZ)
//      triWoop[triOfs*16 + 16] = Vec4f(woopU)
//      triWoop[triOfs*16 + 32] = Vec4f(woopV)
	float4 a,b,c,d;
	CUDA_FUNC_IN void setData(float3& v0, float3& v1, float3& v2)
	{
		float3 p = v0 - v2;
		float3 q = v1 - v2;
		float3 d = cross(p, q);
		float3 c = v2;
		float4x4 m( p.x, q.x, d.x, c.x,
					p.y, q.y, d.y, c.y,
					p.z, q.z, d.z, c.z,
					  0,   0,   0,   1  );
		m = m.Inverse();
		this->a = make_float4(m[2].x, m[2].y, m[2].z, -m[2].w);
		this->b = make_float4(m[0].x, m[0].y, m[0].z, m[0].w);
		this->c = make_float4(m[1].x, m[1].y, m[1].z, m[1].w);
		if(this->a.x == -0.0f)
			this->a.x = 0.0f;
	}

	CUDA_FUNC_IN void getData(float3& v0, float3& v1, float3& v2) const
	{
		float4x4 m(b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, a.x, a.y, a.z, -a.w, 0, 0, 0, 1);
		m = m.Inverse();
		float3 e02 = make_float3(m.X.x, m.Y.x, m.Z.x), e12 = make_float3(m.X.y, m.Y.y, m.Z.y);
		v2 = make_float3(m.X.w, m.Y.w, m.Z.w);
		v0 = v2 + e02;
		v1 = v2 + e12;
	}

	CUDA_FUNC_IN bool Intersect(const Ray& r, TraceResult* a_Result) const
	{
		float Oz = a.w - r.origin.x*a.x - r.origin.y*a.y - r.origin.z*a.z;
		float invDz = 1.0f / (r.direction.x*a.x + r.direction.y*a.y + r.direction.z*a.z);
		float t = Oz * invDz;
		if (t > 0.0001f && t < a_Result->m_fDist)
		{
			float Ox = b.w + r.origin.x*b.x + r.origin.y*b.y + r.origin.z*b.z;
			float Dx = r.direction.x*b.x + r.direction.y*b.y + r.direction.z*b.z;
			float u = Ox + t*Dx;
			if (u >= 0.0f)
			{
				float Oy = c.w + r.origin.x*c.x + r.origin.y*c.y + r.origin.z*c.z;
				float Dy = r.direction.x*c.x + r.direction.y*c.y + r.direction.z*c.z;
				float v = Oy + t*Dy;
				if (v >= 0.0f && u + v <= 1.0f)
				{
					a_Result->m_fDist = t;
					a_Result->m_fUV = make_float2(u, v);
					return true;
				}
			}
		}
		return false;
	}
};

#include "cuda_runtime.h"
#include "..\Base\FileStream.h"
#include "e_SceneInitData.h"

struct e_MeshPartLight
{
	e_String MatName;
	float3 L;
};

#define MESH_STATIC_TOKEN 1
#define MESH_ANIMAT_TOKEN 2

class e_Mesh
{
public:
	AABB m_sLocalBox;
	int m_uType;
public:
	e_StreamReference(e_TriangleData) m_sTriInfo;
	e_StreamReference(e_KernelMaterial) m_sMatInfo;
	e_StreamReference(e_BVHNodeData) m_sNodeInfo;
	e_StreamReference(e_TriIntersectorData) m_sIntInfo;
	e_StreamReference(int) m_sIndicesInfo;
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	unsigned int m_uUsedLights;
public:
	e_Mesh(InputStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<int>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4);
	void Free(e_Stream<e_TriIntersectorData>& a_Stream0, e_Stream<e_TriangleData>& a_Stream1, e_Stream<e_BVHNodeData>& a_Stream2, e_Stream<int>& a_Stream3, e_Stream<e_KernelMaterial>& a_Stream4);
	static void CompileObjToBinary(const char* a_InputFile, OutputStream& a_Out);
	static e_SceneInitData ParseBinary(const char* a_InputFile);
	e_KernelMesh getKernelData()
	{
		e_KernelMesh m_sData;
		m_sData.m_uBVHIndicesOffset = m_sIndicesInfo.getIndex();
		m_sData.m_uBVHNodeOffset = m_sNodeInfo.getIndex() * sizeof(e_BVHNodeData) / sizeof(float4);
		m_sData.m_uBVHTriangleOffset = m_sIntInfo.getIndex() * sizeof(e_TriIntersectorData) / sizeof(float4);
		m_sData.m_uTriangleOffset = m_sTriInfo.getIndex();
		m_sData.m_uStdMaterialOffset = m_sMatInfo.getIndex();
		return m_sData;
	}
	unsigned int getTriangleCount()
	{
		return m_sTriInfo.getLength();
	}
};
