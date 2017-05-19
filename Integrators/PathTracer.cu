#include "PathTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <SceneTypes/Light.h>

namespace CudaTracerLib {

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(NormalizedT<Ray>& r, const NormalizedT<Ray>& rX, const NormalizedT<Ray>& rY, Sampler& rnd, int maxPathLength, int rrStartDepth)
{
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	BSDFSamplingRecord bRec;
	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	TraceResult r2;
	float brdf_scattering_pdf = 0;
	NormalizedT<Vec3f> last_nor;
	while (depth++ < maxPathLength)
	{
		r2 = traceRay(r);
		float minT, maxT;
		bool isInMedium = V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT);
		if (V.HasVolumes() && isInMedium && V.sampleDistance(r, 0, r2.m_fDist, rnd.randomFloat(), mRec))
		{
			cf *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

			if (DIRECT)//direct sampling
			{
				DirectSamplingRecord dRec(mRec.p, NormalizedT<Vec3f>(0.0f));
				Spectrum value = g_SceneData.sampleAttenuatedEmitterDirect(dRec, rnd.randomFloat2());
				if (!value.isZero())
				{
					PhaseFunctionSamplingRecord pRec(-r.dir(), dRec.d);
					float p = V.p(mRec.p, pRec);
					if (p != 0 && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
					{
						const float bsdfPdf = p;//phase functions are normalized
						const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
						cl += cf * value * p * weight;
					}
				}
			}

			PhaseFunctionSamplingRecord pRec(-r.dir());
			cf *= V.Sample(mRec.p, pRec, rnd.randomFloat2());
			r.dir() = pRec.wo;
			r.ori() = mRec.p;
		}
		else if (r2.hasHit())
		{
			if (isInMedium)
				cf *= mRec.transmittance / mRec.pdfFailure;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);
			if (depth == 1)
				bRec.dg.computePartials(r, rX, rY);

			//account for emittance, weighted with MIS if necessary
			if (r2.LightIndex() != UINT_MAX)
			{
				float misWeight = 1.0f;
				if (!DIRECT || depth == 1 || specularBounce)
					misWeight = 1.0f;
				else
				{
					DirectSamplingRecord dRec = DirectSamplingRecFromRay(r, r2.m_fDist, last_nor, bRec.dg.P, bRec.dg.n);
					auto* light = g_SceneData.getLight(r2);
					float direct_pdf = light->pdfDirect(dRec) * g_SceneData.pdfEmitter(light);
					misWeight = MonteCarlo::PowerHeuristic(1, brdf_scattering_pdf, 1, direct_pdf);
				}
				cl += misWeight * cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.dir());
			}

			Spectrum f = r2.getMat().bsdf.sample(bRec, brdf_scattering_pdf, rnd.randomFloat2());
			last_nor = bRec.dg.sys.n;
			if (DIRECT)
				cl += cf * UniformSampleOneLight(bRec, r2.getMat(), rnd, true);
			specularBounce = (bRec.sampledType & EDelta) != 0;
			cf = cf * f;
			r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
		}
		if (depth > rrStartDepth && !specularBounce)
		{
			if (rnd.randomFloat() >= cf.max())
				break;
			cf /= cf.max();
		}
	}
	if (!r2.hasHit())
	{
		float misWeight = 1.0f;
		if (!DIRECT || depth == 1 || specularBounce)
			misWeight = 1.0f;
		else if(g_SceneData.getEnvironmentMap() != 0)
		{
			auto dRec = DirectSamplingRecFromRay(r, r2.m_fDist, last_nor, Vec3f(), NormalizedT<Vec3f>());
			auto* light = g_SceneData.getEnvironmentMap();
			float direct_pdf = light->pdfDirect(dRec) * g_SceneData.pdfEmitter(light);
			misWeight = MonteCarlo::PowerHeuristic(1, brdf_scattering_pdf, 1, direct_pdf);
		}
		cl += misWeight * cf * g_SceneData.EvalEnvironment(r);
	}
	return cl;
}

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTraceRegularization(NormalizedT<Ray>& r, const NormalizedT<Ray>& rX, const NormalizedT<Ray>& rY, Sampler& rnd, float g_fRMollifier, int maxPathLength, int rrStartDepth)
{
	TraceResult r2;
	r2.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	BSDFSamplingRecord bRec;
	//bool hadDelta = false;
	while (traceRay(r.dir(), r.ori(), &r2) && depth++ < maxPathLength)
	{
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);// return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if (depth == 1)
			bRec.dg.computePartials(r, rX, rY);
		if (!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.dir());
		Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		if (DIRECT)
		{
			if (r2.getMat().bsdf.As()->hasComponent(ETypeCombinations::EDelta))
			{
				//hadDelta = true;
				PositionSamplingRecord pRec;
				Spectrum l_s = g_SceneData.sampleEmitterPosition(pRec, rnd.randomFloat2());
				Light* l = (Light*)pRec.object;
				float lDist = distance(pRec.p, bRec.dg.P);
				Vec3f lDir = (pRec.p - bRec.dg.P) / lDist;
				if (!(l->Is<DiffuseLight>() || l->Is<InfiniteLight>()) && !g_SceneData.Occluded(Ray(bRec.dg.P, lDir), 0, lDist))
				{
					float eps = atanf(g_fRMollifier / lDist);
					float normalization = 1.0f / (2 * PI * (1 - cosf(eps)));
					float l_dot_o = dot(lDir, bRec.getOutgoing());
					float indicator = acosf(l_dot_o) <= eps;
					cl += cf * f * l_s * (normalization * indicator);
				}
			}
			else cl += cf * UniformSampleAllLights(bRec, r2.getMat(), 1, rnd);
		}
		specularBounce = (bRec.sampledType & EDelta) != 0;
		cf = cf * f;
		if (depth > rrStartDepth)
		{
			if (rnd.randomFloat() < cf.max())
				cf = cf / cf.max();
			else break;
		}
		r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
		r2.Init();
	}
	//return hadDelta ? Spectrum(1, 0, 0) : Spectrum(0.0f);
	if (!r2.hasHit() && depth == 0)
		cl = cf * g_SceneData.EvalEnvironment(r, rX, rY);
	else cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

void PathTracer::DebugInternal(Image* I, const Vec2i& p)
{
	float m = 1.0f*math::pow((float)m_uPassesDone, -1.0f / 6.0f);
	auto rng = g_SamplerData(p.y * I->getWidth() + p.x);
	NormalizedT<Ray> r, rX, rY;
	Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, Vec2f((float)p.x, (float)p.y), rng.randomFloat2());
	int maxPathLength = m_sParameters.getValue(KEY_MaxPathLength()), rrStart = m_sParameters.getValue(KEY_RRStartDepth());
	PathTrace<true>(r, rX, rY, rng, maxPathLength, rrStart);
}

template<bool DIRECT, bool REGU> __global__ void pathKernel2(unsigned int w, unsigned int h, unsigned int xoff, unsigned int yoff, Image img, float m, int maxPathLength, int rrStart)
{
	Vec2i pixel = TracerBase::getPixelPos(xoff, yoff);
	auto rng = g_SamplerData(TracerBase::getPixelIndex(xoff, yoff, w, h));
	if (pixel.x < w && pixel.y < h)
	{
		NormalizedT<Ray> r, rX, rY;
		Vec2f pX = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		Spectrum imp = g_SceneData.sampleSensorRay(r, rX, rY, pX, rng.randomFloat2());
		Spectrum col = imp * (REGU ? PathTraceRegularization<DIRECT>(r, rX, rY, rng, m, maxPathLength, rrStart) : PathTrace<DIRECT>(r, rX, rY, rng, maxPathLength, rrStart));
		img.AddSample(pX.x, pX.y, col);
	}
}

void PathTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	AABB m_sEyeBox = g_SceneData.m_sBox;
	float m_fInitialRadius = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / 100;
	float ALPHA = 0.75f;
	float radius2 = math::pow(math::pow(m_fInitialRadius, float(2)) / math::pow(float(m_uPassesDone), 0.5f * (1 - ALPHA)), 1.0f / 2.0f);

	int maxPathLength = m_sParameters.getValue(KEY_MaxPathLength()), rrStart = m_sParameters.getValue(KEY_RRStartDepth());

	if (m_sParameters.getValue(KEY_Regularization()))
	{
		if (m_sParameters.getValue(KEY_Direct()))
			pathKernel2<true, true> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> > (w, h, x, y, *I, radius2, maxPathLength, rrStart);
		else pathKernel2<false, true> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> > (w, h, x, y, *I, radius2, maxPathLength, rrStart);
	}
	else
	{
		if (m_sParameters.getValue(KEY_Direct()))
			pathKernel2<true, false> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> > (w, h, x, y, *I, radius2, maxPathLength, rrStart);
		else pathKernel2<false, false> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> > (w, h, x, y, *I, radius2, maxPathLength, rrStart);
	}
}

}