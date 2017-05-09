#include "GameTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Base/CudaMemoryManager.h>
#include <SceneTypes/Light.h>

namespace CudaTracerLib {

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(NormalizedT<Ray>& r, Sampler& rnd, int maxPathLength, int rrStartDepth)
{
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	BSDFSamplingRecord bRec;
	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	TraceResult r2;
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
			if (!DIRECT || (depth == 1 || specularBounce))
				cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.dir());
			Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
			if (DIRECT)
				cl += cf * UniformSampleOneLight(bRec, r2.getMat(), rnd, true);
			specularBounce = (bRec.sampledType & EDelta) != 0;
			cf = cf * f;
			r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
		}
		if (depth > rrStartDepth)
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

CUDA_DEVICE CudaStaticWrapper<PathSpaceFilteringBuffer> g_Buffer;
#define PIXEL_SPACING 16

CUDA_GLOBAL void createHitPointKernel(unsigned int w, unsigned int h)
{
	unsigned int x = PIXEL_SPACING * (threadIdx.x + blockDim.x * blockIdx.x), y = PIXEL_SPACING * (threadIdx.y + blockDim.y * blockIdx.y);

	if (x < w && y < h)
	{
		auto rng = g_SamplerData(y * w + x);
		NormalizedT<Ray> r, rX, rY;
		g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(x, y), rng.randomFloat2());

		auto res = traceRay(r);
		if (res.hasHit())
		{
			BSDFSamplingRecord bRec;
			res.getBsdfSample(r, bRec, ERadiance);
			if (res.getMat().bsdf.hasComponent(EDiffuse))
			{
				res.getMat().bsdf.sample(bRec, rng.randomFloat2());
				auto Li = PathTrace<true>(NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing()), rng, 6, 4);

				bRec.dg.computePartials(r, rX, rY);
				g_Buffer->add_sample(bRec.dg, bRec.wo, Li);
			}
		}
	}
}

void GameTracer::DoRender(Image* I)
{
	static bool init = false;
	if (!init)
	{
		init = true;
		buf.PrepareForRendering(*I, m_pScene);
	}
	buf.StartFrame(I->getWidth(), I->getHeight());
	CopyToSymbol(g_Buffer, buf);
	int p0 = 16, p1 = p0 * PIXEL_SPACING;
	createHitPointKernel << <dim3(I->getWidth() / p1 + 1, I->getHeight() / p1 + 1, 1), dim3(p0, p0, 1) >> >(I->getWidth(), I->getHeight());
	CopyFromSymbol(buf, g_Buffer);
	buf.setOnGPU();
	buf.ComputePixelValues(*I, m_pScene, hasDepthBuffer() ? &this->getDeviceDepthBuffer() : 0);
}

void GameTracer::DebugInternal(Image* I, const Vec2i& pixel)
{

}

void GameTracer::Resize(unsigned int w, unsigned int h)
{
	buf.getParameterCollection().setValue(PathSpaceFilteringBuffer::KEY_PixelFootprintScale(), (float)PIXEL_SPACING / 2);
	buf.getParameterCollection().setValue(PathSpaceFilteringBuffer::KEY_GlobalRadiusScale(), (float)PIXEL_SPACING / 2);
	Tracer<false>::Resize(w, h);
}

}