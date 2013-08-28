#pragma once

#include "..\Engine\e_DynamicScene.h"
#include "..\Engine\e_Camera.h"
#include <vector>
#include "..\Base\FrameworkInterop.h"
#include "..\Engine\e_Image.h"

typedef void (*SliderCreateCallback)(float, float, bool, float*, FW::String);

class k_Tracer
{
public:
	static k_TracerRNGBuffer g_sRngs;
	static AABB GetEyeHitPointBox(e_DynamicScene* s, e_Camera* c);
	static TraceResult TraceSingleRay(Ray r, e_DynamicScene* s, e_Camera* c);
	static void InitRngs(unsigned int N = 1 << 16);
public:
	k_Tracer()
	{
		InitRngs();
	}
	virtual ~k_Tracer()
	{

	}
	virtual void InitializeScene(e_DynamicScene* a_Scene, e_Camera* a_Camera) = 0;
	virtual void Resize(unsigned int x, unsigned int y) = 0;
	virtual void DoPass(e_Image* I, bool a_NewTrace) = 0;
	virtual void Debug(int2 pixel){}
	virtual void PrintStatus(std::vector<FW::String>& a_Buf)
	{
	}
	virtual void CreateSliders(SliderCreateCallback a_Callback)
	{

	}
	virtual float getTimeSpentRendering() = 0;
	virtual float getRaysPerSecond() = 0;
	virtual unsigned int getPassesDone() = 0;
};

class k_TracerBase : public k_Tracer
{
protected:
	//number of rays in las "time spent rendering"
	unsigned int m_uNumRaysTraced;
	//in seconds
	float m_fTimeSpentRendering;
	unsigned int w, h;
	e_DynamicScene* m_pScene;
	e_Camera* m_pCamera;
	cudaEvent_t start,stop;
public:
	k_TracerBase()
		: k_Tracer()
	{
		m_uNumRaysTraced = w = h = 0;
		m_pScene = 0;
		m_pCamera = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	virtual ~k_TracerBase()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	virtual void InitializeScene(e_DynamicScene* a_Scene, e_Camera* a_Camera)
	{
		m_pScene = a_Scene;
		m_pCamera = a_Camera;
	}
	virtual void DoPass(e_Image* I, bool a_NewTrace)
	{
		if(a_NewTrace)
		{
			m_uNumRaysTraced = 0;
			m_fTimeSpentRendering = 0;
			StartNewTrace(I);
		}
		m_pScene->UpdateInvalidated();
		cudaEventRecord(start, 0);
		DoRender(I);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop); 
		m_fTimeSpentRendering += elapsedTime / 1000.0f;
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		w = _w;
		h = _h;
	}
	virtual float getRaysPerSecond()
	{
		return getValuePerSecond(m_uNumRaysTraced, 1);
	}
	virtual float getTimeSpentRendering();
protected:
	virtual void DoRender(e_Image* I) = 0;
	virtual void StartNewTrace(e_Image* I) = 0;
	float getValuePerSecond(float val, float invScale);
#ifdef __CUDACC__
#define k_TracerBase_update_TracedRays { cudaMemcpyFromSymbol(&m_uNumRaysTraced, g_RayTracedCounter, sizeof(unsigned int)); }
#else
#define k_TracerBase_update_TracedRays { m_uNumRaysTraced = g_RayTracedCounter; }
#endif
};

class k_OnePassTracer : public k_TracerBase
{
public:
	unsigned int getPassesDone()
	{
		return 1;
	}
protected:
	virtual void StartNewTrace(e_Image* I)
	{
		I->StartNewRendering();
	}
	virtual void DoRender(e_Image* I)
	{
		m_uNumRaysTraced = 0;
		m_fTimeSpentRendering = 0;
	}
};

class k_ProgressiveTracer : public k_TracerBase
{
protected:
	unsigned int m_uPassesDone;
	virtual void StartNewTrace(e_Image* I)
	{
		I->StartNewRendering();
		m_uPassesDone = 0; 
	}
	virtual void DoRender(e_Image* I)
	{
		m_uNumRaysTraced = 0;
		m_fTimeSpentRendering = 0;
	}
public:
	unsigned int getPassesDone()
	{
		return m_uPassesDone;
	}
};

CUDA_FUNC_IN CameraSample nextSample(int x, int y, CudaRNG& rng, bool DoAntialiasing = false, bool DoDOF = false)
{
	CameraSample s;
	s.imageX = x + (DoAntialiasing ? (rng.randomFloat() - 0.5f) : 0.0f);
	s.imageY = y + (DoAntialiasing ? (rng.randomFloat() - 0.5f) : 0.0f);
	s.lensU = DoDOF ? rng.randomFloat() : 0;
	s.lensV = DoDOF ? rng.randomFloat() : 0;
	s.time = 0;
	return s;
}