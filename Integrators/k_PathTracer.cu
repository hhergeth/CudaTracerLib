#include "k_PathTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTraceTTT(Vec3f& a_Dir, Vec3f& a_Ori, CudaRNG& rnd, float* distTravalled = 0)
{
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	r.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while (k_TraceRay(r0.direction, r0.origin, &r) && depth++ < 7)
	{
		r.getBsdfSample(r0, rnd, &bRec); //return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		cl += cf * r.Le(r0(r.m_fDist), bRec.dg.sys, -r0.direction);
		Spectrum f;
		if (DIRECT)
		{
			DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
			Spectrum value = g_SceneData.m_sLightData[0].sampleDirect(dRec, rnd.randomFloat2());
			bRec.wo = normalize(bRec.dg.toLocal(dRec.d));
			f = r.getMat().bsdf.f(bRec) / g_SceneData.m_sLightData[0].As<e_DiffuseLight>()->shapeSet.Pdf(dRec) * G(bRec.dg.sys.n, dRec.n, bRec.dg.P, dRec.p);
		}
		else f = r.getMat().bsdf.sample(bRec, rnd.randomFloat2());

		cf = cf * f;
		r0 = Ray(r0(r.m_fDist), bRec.getOutgoing());
		r.Init();
	}
	return cl;
}

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rnd, float* distTravalled = 0)
{
	TraceResult r2;
	r2.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while (k_TraceRay(r.direction, r.origin, &r2) && depth++ < 7)
	{
		if (distTravalled && depth == 1)
			*distTravalled = r2.m_fDist;
		r2.getBsdfSample(r, rnd, &bRec);// return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if (depth == 1)
			dg.computePartials(r, rX, rY);
		if (!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		if (DIRECT)
			cl += cf * UniformSampleAllLights(bRec, r2.getMat(), 1);
		specularBounce = (bRec.sampledType & EDelta) != 0;
		float p = f.max();
		if (depth > 5)
		if (rnd.randomFloat() < p)
			f = f / p;
		else break;
		if (f.isZero())
			break;
		cf = cf * f;
		r = Ray(dg.P, bRec.getOutgoing());
		r2.Init();
	}
	if (!r2.hasHit() && depth == 0)
		cl = cf * g_SceneData.EvalEnvironment(r, rX, rY);
	else cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

void k_PathTracer::Debug(e_Image* I, const Vec2i& p)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);	
	PathTrace<true>(r, r, r, rng);
}

template<bool DIRECT> __global__ void pathKernel2(unsigned int w, unsigned int h, unsigned int xoff, unsigned int yoff, k_BlockSampleImage img)
{
	Vec2i pixel = k_TracerBase::getPixelPos(xoff, yoff);
	CudaRNG rng = g_RNGData();
	if(pixel.x < w && pixel.y < h)
	{
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
		Spectrum col = imp * PathTrace<DIRECT>(r, r, r, rng);
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