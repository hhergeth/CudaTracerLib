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
	virtual unsigned int getRaysTracedCount() = 0;
	virtual unsigned int getTimePassed() = 0;
	virtual unsigned int getPassesDone(){return 1;}
	virtual void Debug(int2 pixel){}
	virtual void PrintStatus(std::vector<FW::String>& a_Buf)
	{
	}
	virtual void CreateSliders(SliderCreateCallback a_Callback)
	{

	}
};

class k_TracerBase : public k_Tracer
{
protected:
	unsigned int m_uRaysTraced;
	unsigned int m_uPassesDone;
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
		m_uRaysTraced = m_uPassesDone = w = h = 0;
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
			m_fTimeSpentRendering = 0;
			m_uPassesDone = 0;
			StartNewTrace(I);
		}
		//m_uPassesDone++;
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
	virtual unsigned int getRaysTracedCount()
	{
		return m_uRaysTraced;
	}
	virtual unsigned int getTimePassed()
	{
		return (unsigned int)(m_fTimeSpentRendering * 1000.0f / float(m_uPassesDone));
	}
	virtual unsigned int getPassesDone()
	{
		return m_uPassesDone;
	}
protected:
	virtual void DoRender(e_Image* I) = 0;
	virtual void StartNewTrace(e_Image* I);
	float getValuePerSecond(float val, float invScale);
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