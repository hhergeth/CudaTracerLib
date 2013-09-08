#include <StdAfx.h>
#include "k_CpuTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"

Spectrum pixelFunc(CameraSample& s, int w, int h, CudaRNG& rng)
{
	Ray r = g_CameraData.GenRay(s, w, h);

	return PathTrace(r.direction, r.origin, rng);

	TraceResult r2 = k_TraceRay(r);
	if(r2.hasHit())
		return Spectrum(-dot(r2.lerpFrame().n, r.direction));
	else return Spectrum(0,1,0);
	//return make_float3(r2.m_fDist / sc);
}

k_CpuTracer::k_CpuTracer()
	: m_bDirect(false)
{
	m_sem = CreateSemaphore(0, 0, TCOUNT, 0);
	for(int i = 0; i < TCOUNT; i++)
	{
		data[i].i = i;
		data[i].tracer = this;
		data[i].sem = CreateSemaphore(0, 0, 1, 0);
		_beginthread(threadStart, 0, data + i);
	}
}

void k_CpuTracer::threadStart(void* arg)
{
	k_CpuTracer::threadData* dat = (k_CpuTracer::threadData*)arg;
	while(true)
	{
		WaitForSingleObject(dat->tracer->m_sem, INFINITE);
		CudaRNG rng = g_RNGData();
		int w = dat->tracer->w, h = dat->tracer->h;
		int N = w * h / TCOUNT;
		for(int i = N * dat->i; i < (dat->i + 1) * N; i++)
		{
			int x = i % w, y = i / w;
			CameraSample s = nextSample(x, y, rng);
			Spectrum p = pixelFunc(s, w, h, rng);
			dat->tracer->IMG->AddSample(s, pixelFunc(s, w, h, rng));
		}
		g_RNGData(rng);
		ReleaseSemaphore(dat->sem, 1, 0);
	}
}

void k_CpuTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	IMG = I;
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	ReleaseSemaphore(m_sem, TCOUNT, 0);
	HANDLE hand[TCOUNT];
	for(int i = 0; i < TCOUNT; i++)
		hand[i] = data[i].sem;
	WaitForMultipleObjects(TCOUNT, hand, true, INFINITE);
	I->UpdateDisplay();
	k_TracerBase_update_TracedRays
}

void k_CpuTracer::Debug(int2 pixel)
{
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	CudaRNG rng = g_RNGData();
	CameraSample s = nextSample(pixel.x, pixel.y, rng);
	pixelFunc(s, w, h, rng);
}