#include "BDPT.h"
#include "VCMHelper.h"

namespace CudaTracerLib {

CUDA_FUNC_IN float pathWeight(int force_s, int force_t, int s, int t)
{
	if (force_s != -1 && force_t != -1 && (s != force_s || t != force_t))
		return 0;
	else return 1;
}

CUDA_FUNC_IN void BPT(const Vec2f& pixelPosition, Image& img, Sampler& rng, unsigned int w, unsigned int h,
	bool use_mis, int force_s, int force_t, float LScale)
{
	float mLightSubPathCount = 1 * 1;
	const float etaVCM = (PI * 1) * mLightSubPathCount;
	float mMisVmWeightFactor = 0;
	float mMisVcWeightFactor = Mis(1.f / etaVCM);

	BPTVertex lightPath[NUM_V_PER_PATH];
	BPTSubPathState lightPathState;
	sampleEmitter(lightPathState, rng, mMisVcWeightFactor);
	int emitterPathLength = 1, emitterVerticesStored = 0;
	for (; emitterVerticesStored < NUM_V_PER_PATH && emitterPathLength < MAX_SUB_PATH_LENGTH; emitterPathLength++)
	{
		TraceResult r2 = traceRay(lightPathState.r);
		if (!r2.hasHit())
			break;

		BPTVertex& v = lightPath[emitterVerticesStored];
		r2.getBsdfSample(lightPathState.r, v.bRec, ETransportMode::EImportance, &lightPathState.throughput);

		if (emitterPathLength > 1 || true)
			lightPathState.dVCM *= r2.m_fDist * r2.m_fDist;
		lightPathState.dVCM /= math::abs(Frame::cosTheta(v.bRec.wi));
		lightPathState.dVC /= math::abs(Frame::cosTheta(v.bRec.wi));

		//store in list
		if (r2.getMat().bsdf.hasComponent(ESmooth))
		{
			v.dVCM = lightPathState.dVCM;
			v.dVC = lightPathState.dVC;
			v.throughput = lightPathState.throughput;
			v.mat = &r2.getMat();
			v.subPathLength = emitterPathLength + 1;
			emitterVerticesStored++;
		}

		//connect to camera
		if (r2.getMat().bsdf.hasComponent(ESmooth))
			connectToCamera(lightPathState, v.bRec, r2.getMat(), img, rng, mLightSubPathCount, mMisVmWeightFactor, LScale * pathWeight(force_s, force_t, emitterPathLength, 1), use_mis);

		if (!sampleScattering(lightPathState, v.bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			break;
	}

	BPTSubPathState cameraState;
	sampleCamera(cameraState, rng, pixelPosition, mLightSubPathCount);
	Spectrum acc(0.0f);
	for (int camPathLength = 1; camPathLength <= NUM_V_PER_PATH; camPathLength++)
	{
		TraceResult r2 = traceRay(cameraState.r);
		if (!r2.hasHit())
		{
			//sample environment map
			acc += pathWeight(force_s, force_t, 0, camPathLength) * cameraState.throughput * gatherEnvironmentMap(cameraState, camPathLength, use_mis);
			break;
		}

		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(cameraState.r, bRec, ETransportMode::ERadiance);

		cameraState.dVCM *= r2.m_fDist * r2.m_fDist;
		cameraState.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
		cameraState.dVC /= math::abs(Frame::cosTheta(bRec.wi));

		if (r2.LightIndex() != UINT_MAX)
		{
			acc += pathWeight(force_s, force_t, 0, camPathLength) * cameraState.throughput * gatherLight(cameraState, bRec, r2, rng, camPathLength, use_mis);
			break;
		}

		if (r2.getMat().bsdf.hasComponent(ESmooth))
			acc += pathWeight(force_s, force_t, 1, camPathLength) * cameraState.throughput * connectToLight(cameraState, bRec, r2.getMat(), rng, mMisVmWeightFactor, use_mis);

		if (r2.getMat().bsdf.hasComponent(ESmooth))
			for (int emitterVertexIdx = 0; emitterVertexIdx < emitterVerticesStored; emitterVertexIdx++)
			{
				BPTVertex lv = lightPath[emitterVertexIdx];
				acc += pathWeight(force_s, force_t, lv.subPathLength, camPathLength) * cameraState.throughput * lv.throughput * connectVertices(lv, cameraState, bRec, r2.getMat(), mMisVcWeightFactor, mMisVmWeightFactor, use_mis);
			}

		if (!sampleScattering(cameraState, bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			break;
	}

	img.AddSample(pixelPosition.x, pixelPosition.y, acc * LScale);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, Image img,
	bool use_mis, int force_s, int force_t, float LScale)
{
	Vec2i pixel = TracerBase::getPixelPos(xoff, yoff);
	auto rng = g_SamplerData();
	if (pixel.x < w && pixel.y < h)
		BPT(Vec2f(pixel.x + rng.randomFloat(), pixel.y + rng.randomFloat()), img, rng, w, h, use_mis, force_s, force_t, LScale);
	g_SamplerData(rng);
}

void BDPT::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	pathKernel << < BLOCK_SAMPLER_LAUNCH_CONFIG >> >(w, h, x, y, *I,
													 m_sParameters.getValue(KEY_UseMis()), m_sParameters.getValue(KEY_Force_s()), m_sParameters.getValue(KEY_Force_t()), m_sParameters.getValue(KEY_ResultMultiplier()));
}

void BDPT::DebugInternal(Image* I, const Vec2i& pixel)
{
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	auto rng = g_SamplerData();
	BPT(Vec2f(pixel), *I, rng, w, h, 
		m_sParameters.getValue(KEY_UseMis()), m_sParameters.getValue(KEY_Force_s()), m_sParameters.getValue(KEY_Force_t()), m_sParameters.getValue(KEY_ResultMultiplier()));
	g_SamplerData(rng);
}

}