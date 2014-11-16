#pragma once

#include "..\Engine\e_DynamicScene.h"
#include "..\Engine\e_Sensor.h"
#include <vector>
#include "..\Engine\e_Image.h"

typedef void (*SliderCreateCallback)(float, float, bool, float*, std::string);

class k_Tracer
{
public:
	static CudaRNGBuffer g_sRngs;
	static AABB GetEyeHitPointBox(e_DynamicScene* s, e_Sensor* c, bool recursive);
	static float GetLightVisibility(e_DynamicScene* s, e_Sensor* c, int recursion_depth);
	static TraceResult TraceSingleRay(Ray r, e_DynamicScene* s, e_Sensor* c);
	static void InitRngs(unsigned int N = 1 << 16);
public:
	k_Tracer();
	virtual ~k_Tracer()
	{

	}
	virtual void InitializeScene(e_DynamicScene* a_Scene, e_Sensor* a_Camera) = 0;
	virtual void Resize(unsigned int x, unsigned int y) = 0;
	virtual void DoPass(e_Image* I, bool a_NewTrace) = 0;
	virtual void Debug(int2 pixel){}
	virtual void PrintStatus(std::vector<std::string>& a_Buf)
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
	e_Sensor* m_pCamera;
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
	virtual void InitializeScene(e_DynamicScene* a_Scene, e_Sensor* a_Camera)
	{
		m_pScene = a_Scene;
		m_pCamera = a_Camera;
	}
	virtual void DoPass(e_Image* I, bool a_NewTrace)
	{
		g_sRngs.NextPass();
		if(a_NewTrace)
		{
			m_uNumRaysTraced = 0;
			m_fTimeSpentRendering = 0;
			StartNewTrace(I);
		}
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
		return getValuePerSecond((float)m_uNumRaysTraced, 1.0f);
	}
	virtual float getTimeSpentRendering();
protected:
	virtual void DoRender(e_Image* I) = 0;
	virtual void StartNewTrace(e_Image* I) = 0;
	float getValuePerSecond(float val, float invScale);
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
		I->Clear();
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
	virtual void StartNewTrace(e_Image* I);
	virtual void DoRender(e_Image* I);
public:
	unsigned int getPassesDone()
	{
		return m_uPassesDone;
	}
};