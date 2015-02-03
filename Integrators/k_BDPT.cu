#include "k_BDPT.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"
#include "k_VCMHelper.h"

CUDA_FUNC_IN float pathWeight(int force_s, int force_t, int s, int t)
{
	if (force_s != -1 && force_t != -1 && (s != force_s || t != force_t))
		return 0;
	else return 1;
}

CUDA_FUNC_IN void BPT(const float2& pixelPosition, e_Image& g_Image, CudaRNG& rng, unsigned int w, unsigned int h,
					  bool use_mis, int force_s, int force_t, float LScale)
{
	float mLightSubPathCount = 1 * 1;
	const float etaVCM = (PI * 1) * mLightSubPathCount;
	float mMisVmWeightFactor = 0;
	float mMisVcWeightFactor = Mis(1.f / etaVCM);

	const int NUM_V_PER_PATH = 5;
	BPTVertex lightPath[NUM_V_PER_PATH];
	BPTSubPathState lightPathState;
	sampleEmitter(lightPathState, rng, mMisVcWeightFactor);
	int emitterPathLength = 1, emitterVerticesStored = 0;
	for (; emitterVerticesStored < NUM_V_PER_PATH; emitterPathLength++)
	{
		TraceResult r2 = k_TraceRay(lightPathState.r);
		if (!r2.hasHit())
			break;

		BPTVertex& v = lightPath[emitterVerticesStored];
		r2.getBsdfSample(lightPathState.r, rng, &v.bRec);

		if (emitterPathLength > 1 || true)
			lightPathState.dVCM *= r2.m_fDist * r2.m_fDist;
		lightPathState.dVCM /= fabsf(fabsf(Frame::cosTheta(v.bRec.wi)));
		lightPathState.dVC /= fabsf(fabsf(Frame::cosTheta(v.bRec.wi)));

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
			connectToCamera(lightPathState, v.bRec, r2.getMat(), g_Image, rng, mLightSubPathCount, mMisVmWeightFactor, LScale * pathWeight(force_s, force_t, emitterPathLength, 1), use_mis);

		if (!sampleScattering(lightPathState, v.bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			break;
	}

	BPTSubPathState cameraState;
	sampleCamera(cameraState, rng, pixelPosition, mLightSubPathCount);
	Spectrum acc(0.0f);
	for (int camPathLength = 1; camPathLength <= NUM_V_PER_PATH; camPathLength++)
	{
		TraceResult r2 = k_TraceRay(cameraState.r);
		if (!r2.hasHit())
		{
			//sample environment map

			break;
		}

		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(cameraState.r, rng, &bRec);

		cameraState.dVCM *= r2.m_fDist * r2.m_fDist;
		cameraState.dVCM /= fabsf(fabsf(Frame::cosTheta(bRec.wi)));
		cameraState.dVC /= fabsf(fabsf(Frame::cosTheta(bRec.wi)));

		if (r2.LightIndex() != 0xffffffff)
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

	g_Image.AddSample(pixelPosition.x, pixelPosition.y, acc * LScale);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image,
		bool use_mis, int force_s, int force_t, float LScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		BPT(make_float2(x, y), g_Image, rng, w, h, use_mis, force_s, force_t, LScale);
	g_RNGData(rng);
}

void k_BDPT::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene, g_sRngs);
	int p = 16;
	if(w < 200 && h < 200)
		pathKernel << < dim3((w + p - 1) / p, (h + p - 1) / p, 1), dim3(p, p, 1) >> >(w, h, 0, 0, *I, use_mis, force_s, force_t, LScale);
	else
	{
		unsigned int q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				pathKernel << < dim3(q, q, 1), dim3(p, p, 1) >> >(w, h, pq * i, pq * j, *I, use_mis, force_s, force_t, LScale);
	}
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(1.0f / float(m_uPassesDone));
}

void k_BDPT::Debug(e_Image* I, int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	BPT(make_float2(pixel), *I, rng, w, h, use_mis, force_s, force_t, LScale);
	g_RNGData(rng);
}