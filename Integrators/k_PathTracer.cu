#include "k_PathTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT, typename DEBUGGER> CUDA_FUNC_IN Spectrum PathTrace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rnd, DEBUGGER& dbg)
{
	TraceResult r2;
	r2.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	dbg.StartNewPath(&g_SceneData.m_Camera, r.origin, cf);
	while (k_TraceRay(r.direction, r.origin, &r2) && depth++ < 7)
	{
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);// return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if (depth == 1)
			dg.computePartials(r, rX, rY);
		if (!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		dbg.AppendVertex(ITracerDebugger::PathType::Camera, r, r2);
		if (DIRECT)
			cl += cf * UniformSampleAllLights(bRec, r2.getMat(), 1, rnd);
		specularBounce = (bRec.sampledType & EDelta) != 0;
		if (depth > 5)
		{
			if (rnd.randomFloat() < f.max())
				f = f / f.max();
			else break;
		}
		cf = cf * f;
		r = Ray(dg.P, bRec.getOutgoing());
		r2.Init();
	}
	if (!r2.hasHit() && depth == 0)
		cl = cf * g_SceneData.EvalEnvironment(r, rX, rY);
	else cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

void k_PathTracer::Debug(e_Image* I, const Vec2i& p, ITracerDebugger* debugger)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);	
	PathTrace<true, k_KernelTracerDebugger>(r, r, r, rng, k_KernelTracerDebugger(debugger));
}

template<bool DIRECT> __global__ void pathKernel2(unsigned int w, unsigned int h, unsigned int xoff, unsigned int yoff, k_BlockSampleImage img)
{
	Vec2i pixel = k_TracerBase::getPixelPos(xoff, yoff);
	CudaRNG rng = g_RNGData();
	if(pixel.x < w && pixel.y < h)
	{
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
		Spectrum col = imp * PathTrace<DIRECT, k_KernelTracerDebugger_NO_OP>(r, r, r, rng, k_KernelTracerDebugger_NO_OP());
		img.Add(pixel.x, pixel.y, col);
	}
	g_RNGData(rng);
}

void k_PathTracer::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	if (m_Direct)
		pathKernel2<true> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage());
	else pathKernel2<false> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage());
}