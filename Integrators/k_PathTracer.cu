#include "k_PathTracer.h"
#include <Kernel/k_TraceHelper.h>
#include <time.h>
#include <Kernel/k_TraceAlgorithms.h>
#include <Engine/e_Light.h>

namespace CudaTracerLib {

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rnd)
{
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	TraceResult r2;
	while (depth++ < 10)
	{
		r2 = k_TraceRay(r);
		float minT, maxT;
		bool isInMedium = V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT);
		if (V.HasVolumes() && isInMedium && V.sampleDistance(r, 0, r2.m_fDist, rnd, mRec))
		{
			cf *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

			if (DIRECT)//direct sampling
			{
				DirectSamplingRecord dRec(mRec.p, Vec3f(0));
				Spectrum value = g_SceneData.sampleEmitterDirect(dRec, rnd.randomFloat2());
				if (!value.isZero())
				{
					float p = V.p(mRec.p, -r.direction, dRec.d, rnd);
					if (p != 0 && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
					{
						const float bsdfPdf = p;//phase functions are normalized
						const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
						cl += cf * value * p * weight * Transmittance(Ray(dRec.ref, dRec.d), 0, dRec.dist);
					}
				}
			}

			cf *= V.Sample(mRec.p, -r.direction, rnd, &r.direction);
			r.origin = mRec.p;
		}
		else if (r2.hasHit())
		{
			if (isInMedium)
				cf *= Transmittance(r, 0, r2.m_fDist);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rnd);
			if (depth == 1)
				dg.computePartials(r, rX, rY);
			if (!DIRECT || (depth == 1 || specularBounce))
				cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);
			Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
			if (DIRECT)
				cl += cf * UniformSampleOneLight(bRec, r2.getMat(), rnd, true);
			specularBounce = (bRec.sampledType & EDelta) != 0;
			cf = cf * f;
			r = Ray(dg.P, bRec.getOutgoing());
		}
		if (depth > 5)
		{
			if (rnd.randomFloat() >= cf.max())
				break;
			cf /= cf.max();
		}
	}
	if (!r2.hasHit())
		cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTraceRegularization(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rnd, float g_fRMollifier)
{
	TraceResult r2;
	r2.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	//bool hadDelta = false;
	while (k_TraceRay(r.direction, r.origin, &r2) && depth++ < 7)
	{
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rnd);// return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if (depth == 1)
			dg.computePartials(r, rX, rY);
		if (!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		if (DIRECT)
		{
			if (r2.getMat().bsdf.As()->hasComponent(ETypeCombinations::EDelta))
			{
				//hadDelta = true;
				PositionSamplingRecord pRec;
				Spectrum l_s = g_SceneData.sampleEmitterPosition(pRec, rnd.randomFloat2());
				e_KernelLight* l = (e_KernelLight*)pRec.object;
				float lDist = distance(pRec.p, bRec.dg.P);
				Vec3f lDir = (pRec.p - bRec.dg.P) / lDist;
				if (!(l->Is<e_DiffuseLight>() || l->Is<e_InfiniteLight>()) && !g_SceneData.Occluded(Ray(bRec.dg.P, lDir), 0, lDist))
				{
					float eps = atanf(g_fRMollifier / lDist);
					float normalization = 1.0f / (2 * PI * (1 - cosf(eps)));
					float l_dot_o = dot(lDir, normalize(bRec.getOutgoing()));
					float indicator = acosf(l_dot_o) <= eps;
					cl += cf * f * l_s * (normalization * indicator);
				}
			}
			else cl += cf * UniformSampleAllLights(bRec, r2.getMat(), 1, rnd);
		}
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
	//return hadDelta ? Spectrum(1, 0, 0) : Spectrum(0.0f);
	if (!r2.hasHit() && depth == 0)
		cl = cf * g_SceneData.EvalEnvironment(r, rX, rY);
	else cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

void k_PathTracer::Debug(e_Image* I, const Vec2i& p)
{
	float m = 1.0f*pow(this->m_uPassesDone, -1.0f / 6);
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);
	r.direction = Vec3f(0, 0, 1);
	PathTrace<true>(r, r, r, rng);
}

template<bool DIRECT, bool REGU> __global__ void pathKernel2(unsigned int w, unsigned int h, unsigned int xoff, unsigned int yoff, k_BlockSampleImage img, float m)
{
	Vec2i pixel = k_TracerBase::getPixelPos(xoff, yoff);
	CudaRNG rng = g_RNGData();
	if (pixel.x < w && pixel.y < h)
	{
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
		Spectrum col = imp * (REGU ? PathTraceRegularization<DIRECT>(r, r, r, rng, m) : PathTrace<DIRECT>(r, r, r, rng));
		img.Add(pixel.x, pixel.y, col);
	}
	g_RNGData(rng);
}

void k_PathTracer::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	AABB m_sEyeBox = g_SceneData.m_sBox;
	float m_fInitialRadius = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / 100;
	float ALPHA = 0.75f;
	float radius2 = math::pow(math::pow(m_fInitialRadius, float(2)) / math::pow(float(m_uPassesDone), 0.5f * (1 - ALPHA)), 1.0f / 2.0f);

	if (m_Regularization)
	{
		if (m_Direct)
			pathKernel2<true, true> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage(), radius2);
		else pathKernel2<false, true> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage(), radius2);
	}
	else
	{
		if (m_Direct)
			pathKernel2<true, false> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage(), radius2);
		else pathKernel2<false, false> << <numBlocks, threadsPerBlock >> > (w, h, x, y, m_pBlockSampler->getBlockImage(), radius2);
	}
}

}