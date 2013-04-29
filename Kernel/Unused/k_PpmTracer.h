#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"

#define START_RADIUS 0.1f
#define ALPHA 0.7f

struct k_CamHit
{
	float3 HitPoint;
	float3 Normal;
	float3 Direction;
	float3 weight;
	float3 flux;
	unsigned short x, y;
	unsigned short matIndex;
	float Radius;
	unsigned int N;
	k_CamHit() { }
	CUDA_ONLY_FUNC k_CamHit(float3& p, float3& n, float3& inDir, int _x, int _y, int mat)
	{
		HitPoint = p;
		Normal = n;
		Direction = inDir;
		weight = make_float3(1);
		flux = make_float3(0);
		x = _x;
		y = _y;
		matIndex = mat;
		Radius = START_RADIUS;
		N = 0;
	}
	float3 getPos()
	{
		return HitPoint;
	}
	bool isValid()
	{
		return Radius != 0;
	}
};

class k_PpmTracer : public k_Tracer
{
private:
	unsigned int m_uRaysTraced;
	unsigned int m_uTimePassed;
	unsigned int w, h;
	e_DynamicScene* m_pScene;
	e_Camera* m_pCamera;
	cudaEvent_t start,stop;
	float4* m_pTmpData;
	k_CamHit* m_pDeviceHitData, *m_pHostHitData;
	e_PointBVHNode* m_pDeviceBVHData, *m_pHostBVHData;
	unsigned int* m_pDeviceIndexData, *m_pHostIndexData;
	unsigned int m_uPass;
	int startNode;
	unsigned int m_uNumTraced;
public:
	k_PpmTracer()
	{
		m_pTmpData = 0;
		m_uPass = 0;
		m_pDeviceHitData = m_pHostHitData = 0;
		m_pDeviceIndexData = m_pHostIndexData = 0;
		m_pDeviceBVHData = m_pHostBVHData = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~k_PpmTracer()
	{
		cudaFree(m_pDeviceHitData);
		cudaFree(m_pDeviceIndexData);
		cudaFree(m_pDeviceBVHData);
		delete [] m_pHostHitData;
		delete [] m_pHostIndexData;
		delete [] m_pHostBVHData;
		cudaFree(m_pTmpData);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	virtual void InitializeScene(e_DynamicScene* a_Scene, e_Camera* a_Camera);
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void DoPass(RGBCOL* a_Buf, bool a_NewTrace);
	virtual unsigned int getRaysTracedCount()
	{
		return m_uRaysTraced;
	}
	virtual unsigned int getTimePassed()
	{
		return m_uTimePassed;
	}
	virtual unsigned int getPassesDone(){return m_uPass;}
};

int BuildPointBVHQ(k_CamHit* data, int a_Count, e_PointBVHNode* a_NodeOut, int* a_IndexOut, float maxRadius);