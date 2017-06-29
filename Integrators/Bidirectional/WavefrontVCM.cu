#include "WavefrontVCM.h"

namespace CudaTracerLib {

WavefrontVCM::WavefrontVCM(unsigned int a_NumLightRays)
	: m_uNumLightRays(a_NumLightRays), m_sLightBuf(a_NumLightRays, 1), m_sCamBuf(BLOCK_SAMPLER_BlockSize * BLOCK_SAMPLER_BlockSize, BLOCK_SAMPLER_BlockSize * BLOCK_SAMPLER_BlockSize * (MAX_LIGHT_SUB_PATH_LENGTH + 1)),
	  m_sPhotonMapsNext(Vec3u(200), a_NumLightRays * MAX_LIGHT_SUB_PATH_LENGTH)
{
	ThrowCudaErrors(CUDA_MALLOC(&m_pDeviceLightVertices, sizeof(BPTVertex) * MAX_LIGHT_SUB_PATH_LENGTH * a_NumLightRays));
}

WavefrontVCM::~WavefrontVCM()
{
	m_sLightBuf.Free();
	m_sCamBuf.Free();
}

CUDA_CONST float mMisVcWeightFactor;
CUDA_CONST float mMisVmWeightFactor;
CUDA_CONST float mLightSubPathCount;

CUDA_DEVICE CudaStaticWrapper<k_WVCM_LightBuffer> g_sLightBuf;

CUDA_DEVICE CudaStaticWrapper<VCMSurfMap> g_NextMap2;

CUDA_DEVICE CudaStaticWrapper<k_WVCM_CamBuffer> g_sCamBuf;

CUDA_GLOBAL void createLightRays(unsigned int g_DeviceNumLightPaths)
{
	unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	auto rng = g_SamplerData(idx);
	if (idx < g_DeviceNumLightPaths)
	{
		k_BPTPathState state;
		sampleEmitter(state.state, rng, mMisVcWeightFactor);
		state.m_uVertexStart = idx;
		g_sLightBuf->insertPayloadElement(state, state.state.r);
	}
}

CUDA_GLOBAL void extendLighRays(BPTVertex* g_pLightVertices, Image I, int iteration)
{
	k_BPTPathState ent;
	NormalizedT<Ray> __;
	TraceResult r2;
	unsigned int rayIdx;
	if (g_sLightBuf->tryFetchPayloadElement(ent, __, r2, &rayIdx))
	{
		auto rng = g_SamplerData(rayIdx);
		unsigned int vIdx = ent.m_uVertexStart & 0x00ffffff, vOff = (ent.m_uVertexStart & 0xff000000) >> 24;
		if (r2.hasHit())
		{
			BPTVertex v;
			v.mat = 0;
			r2.getBsdfSample(ent.state.r, v.bRec, ETransportMode::EImportance, &ent.state.throughput);

			if (vOff > 1 || true)
				ent.state.dVCM *= r2.m_fDist * r2.m_fDist;
			ent.state.dVCM /= math::abs(Frame::cosTheta(v.bRec.wi));
			ent.state.dVC /= math::abs(Frame::cosTheta(v.bRec.wi));

			if (r2.getMat().bsdf.hasComponent(ESmooth))
			{
				v.dVCM = ent.state.dVCM;
				v.dVC = ent.state.dVC;
				v.throughput = ent.state.throughput;
				v.mat = &r2.getMat();
				v.subPathLength = vOff + 1;
			}
			g_pLightVertices[vIdx * MAX_LIGHT_SUB_PATH_LENGTH + vOff] = v;

			auto ph = k_MISPhoton(v.throughput, -ent.state.r.dir(), v.bRec.dg.sys.n, v.dVC, v.dVCM, v.dVM);
			Vec3u cell_idx = g_NextMap2->getHashGrid().Transform(v.bRec.dg.P);
			ph.setPos(g_NextMap2->getHashGrid(), cell_idx, v.bRec.dg.P);
			g_NextMap2->Store(cell_idx, ph);

			if (r2.getMat().bsdf.hasComponent(ESmooth))
				connectToCamera(ent.state, v.bRec, r2.getMat(), I, rng, mLightSubPathCount, mMisVmWeightFactor, 1, true);

			if (vOff < MAX_LIGHT_SUB_PATH_LENGTH - 1 && sampleScattering(ent.state, v.bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			{
				ent.m_uVertexStart = ((vOff + 1) << 24) | vIdx;
				g_sLightBuf->insertPayloadElement(ent, ent.state.r);
			}
		}
	}
}

void WavefrontVCM::DoRender(Image* I)
{
	m_uLightOff = 0;
	float a_Radius = this->getCurrentRadius(2);
	const float etaVCM = (PI * a_Radius * a_Radius) * m_uNumLightRays;
	float MisVmWeightFactor = 1;
	float MisVcWeightFactor = 1.0f / etaVCM;
	float one = 1;
	ThrowCudaErrors(cudaMemset(m_pDeviceLightVertices, 0, sizeof(BPTVertex) * MAX_LIGHT_SUB_PATH_LENGTH * m_uNumLightRays));
	ThrowCudaErrors(cudaMemcpyToSymbol(mMisVcWeightFactor, &MisVcWeightFactor, sizeof(MisVcWeightFactor)));
	ThrowCudaErrors(cudaMemcpyToSymbol(mMisVmWeightFactor, &MisVmWeightFactor, sizeof(MisVmWeightFactor)));
	ThrowCudaErrors(cudaMemcpyToSymbol(mLightSubPathCount, &one, sizeof(one)));

	m_sLightBuf.StartFrame();
	CopyToSymbol(g_sLightBuf, m_sLightBuf);
	createLightRays << <m_uNumLightRays / (32 * 6) + 1, dim3(32, 6) >> >(m_uNumLightRays);
	ThrowCudaErrors(cudaThreadSynchronize());
	CopyFromSymbol(m_sLightBuf, g_sLightBuf);

	m_sPhotonMapsNext.ResetBuffer();
	CopyToSymbol(g_NextMap2, m_sPhotonMapsNext);

	int i = 0;
	while(!m_sLightBuf.isEmpty())
	{
		m_sLightBuf.FinishIteration();
		CopyToSymbol(g_sLightBuf, m_sLightBuf);
		extendLighRays << <m_sLightBuf.getNumPayloadElementsInQueue() / (32 * 6) + 1, dim3(32, 6) >> >( m_pDeviceLightVertices, *I, i++);
		ThrowCudaErrors(cudaThreadSynchronize());
		CopyFromSymbol(m_sLightBuf, g_sLightBuf);

	}

	CopyFromSymbol(m_sPhotonMapsNext, g_NextMap2);
	m_sPhotonMapsNext.setOnGPU();

	Tracer<true>::DoRender(I);
}

void WavefrontVCM::StartNewTrace(Image* I)
{
	Tracer<true>::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
	m_sEyeBox = m_sEyeBox.Extend(0.1f);
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	m_sPhotonMapsNext.SetGridDimensions(m_sEyeBox);
}

CUDA_GLOBAL void createCameraRays(int xoff, int yoff, int blockW, int blockH, int w, int h)
{
	auto rng = g_SamplerData(TracerBase::getPixelIndex(xoff, yoff, w, h));
	Vec2i pixel = TracerBase::getPixelPos(xoff, yoff);
	if (pixel.x < w && pixel.y < h)
	{
		k_BPTCamSubPathState ent;
		sampleCamera(ent.state, rng, Vec2f(pixel.x, pixel.y), mLightSubPathCount);
		ent.x = pixel.x;
		ent.y = pixel.y;
		ent.acc = Spectrum(0.0f);
		ent.last_light_vertex_start = -1;
		ent.prev_throughput = Spectrum(1.0f);
		g_sCamBuf->insertPayloadElement(ent, ent.state.r);
	}
}

CUDA_GLOBAL void performPPMEstimate(float a_Radius, float nPhotons)
{
	BSDFSamplingRecord bRec;
	k_BPTCamSubPathState ent;
	NormalizedT<Ray> __;
	TraceResult r2;
	if (g_sCamBuf->tryFetchPayloadElement(ent, __, r2))
	{
		if (r2.hasHit())
		{
			r2.getBsdfSample(ent.state.r, bRec, ETransportMode::ERadiance);

			auto stored = Vec3f(ent.state.dVCM, ent.state.dVC, ent.state.dVM);
			ent.state.dVCM *= r2.m_fDist * r2.m_fDist;
			ent.state.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVC /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVM /= math::abs(Frame::cosTheta(bRec.wi));

			Spectrum phL;
			if (!r2.getMat().bsdf.hasComponent(EGlossy))
				phL = L_Surface2<false>(g_NextMap2, ent.state, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
			else phL = L_Surface2<true>(g_NextMap2, ent.state, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
			ent.acc += ent.state.throughput * phL;
			ent.state.dVCM = stored.x; ent.state.dVC = stored.y; ent.state.dVM = stored.z;
			g_sCamBuf->insertPayloadElement(ent, ent.state.r, &r2);
		}
	}
}

CUDA_GLOBAL void extendCameraRays(Image I, int iteration, bool lastIteration, float a_Radius, unsigned int lightOff, unsigned int numLightPaths, BPTVertex* g_pLightVertices)
{
	BSDFSamplingRecord bRec;
	k_BPTCamSubPathState ent;
	NormalizedT<Ray> camera_ray;
	TraceResult r2;
	unsigned int rayIdx;
	if (g_sCamBuf->tryFetchPayloadElement(ent, camera_ray, r2, &rayIdx))
	{
		auto rng = g_SamplerData(rayIdx);

		/*if (iteration > 0 && ent.last_light_vertex_start != -1)
		{
			auto vOff = ent.last_light_vertex_start, i = 0;
			while (i < MAX_LIGHT_SUB_PATH_LENGTH && g_pLightVertices[vOff + i].mat)
			{
				NormalizedT<Ray> shadowRay;
				TraceResult shadowRayRes;
				BPTVertex& v = g_pLightVertices[vOff + i];
				if (ent.trace_res[i] != UINT_MAX && g_sCamBuf->accessSecondaryRay(ent.trace_res[i], shadowRay, shadowRayRes) && !g_SceneData.Occluded(shadowRay, 0.0f, distance(v.bRec.dg.P, camera_ray.ori()), shadowRayRes.m_fDist))
				{
					BPTVertex lv;
					lv.bRec = v.bRec;
					lv.dVC = v.dVC;
					lv.dVM = v.dVM;
					lv.dVCM = v.dVCM;
					lv.mat = v.mat;
					lv.subPathLength = v.subPathLength;
					lv.throughput = v.throughput;
					ent.acc += ent.prev_throughput * lv.throughput * connectVertices<false>(lv, ent.state, bRec, r2.getMat(), mMisVcWeightFactor, mMisVmWeightFactor, true);
				}
				i++;
			}
		}*/

		bool extended = false;
		if (r2.hasHit())
		{
			r2.getBsdfSample(ent.state.r, bRec, ETransportMode::ERadiance);

			ent.state.dVCM *= r2.m_fDist * r2.m_fDist;
			ent.state.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVC /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVM /= math::abs(Frame::cosTheta(bRec.wi));

			if (r2.LightIndex() != UINT_MAX)
				ent.acc += ent.state.throughput * gatherLight(ent.state, bRec, r2, rng, iteration + 1, true);

			if (r2.getMat().bsdf.hasComponent(ESmooth))
			{
				ent.acc += ent.state.throughput * connectToLight(ent.state, bRec, r2.getMat(), rng, mMisVmWeightFactor, true);

				unsigned int vOff = ((lightOff + rayIdx) % numLightPaths) * MAX_LIGHT_SUB_PATH_LENGTH, i = 0;
				ent.last_light_vertex_start = vOff;
				ent.prev_throughput = ent.state.throughput;
				while (i < 1 && g_pLightVertices[vOff + i].mat)
				{
					//BPTVertex& v = g_pLightVertices[vOff + i];
					unsigned int ABC;
					if (!g_sCamBuf->insertSecondaryRay(camera_ray, ABC))//normalize(v.bRec.dg.P - bRec.dg.P)
						ABC = UINT_MAX;
				}
			}

			if (!lastIteration && sampleScattering(ent.state, bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			{
				extended = true;
				g_sCamBuf->insertPayloadElement(ent, ent.state.r);
			}
		}
		if (!extended)
			I.AddSample(ent.x, ent.y, ent.acc);
	}
}

void WavefrontVCM::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	CopyToSymbol(g_sCamBuf, m_sCamBuf);
	m_sCamBuf.StartFrame();
	createCameraRays << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(x, y, blockW, blockH, w, h);
	ThrowCudaErrors(cudaThreadSynchronize());
	CopyFromSymbol(m_sCamBuf, g_sCamBuf);

	int i = 0;
	while(!m_sCamBuf.isEmpty())
	{
		m_sCamBuf.FinishIteration();
		unsigned int N = m_sCamBuf.getNumPayloadElementsInQueue();
		CopyToSymbol(g_sCamBuf, m_sCamBuf);
		performPPMEstimate << <N / (32 * 6) + 1, dim3(32, 6) >> >(getCurrentRadius(2), (float)m_uNumLightRays);
		CopyFromSymbol(m_sCamBuf, g_sCamBuf);
		m_sCamBuf.FinishIteration<false>();
		CopyToSymbol(g_sCamBuf, m_sCamBuf);
		extendCameraRays << <N / (32 * 6) + 1, dim3(32, 6) >> >(*I, i++, i == 4, getCurrentRadius(2), m_uLightOff, m_uNumLightRays, m_pDeviceLightVertices);
		ThrowCudaErrors(cudaThreadSynchronize());
		CopyFromSymbol(m_sCamBuf, g_sCamBuf);

	}
	m_uLightOff += blockW * blockH;
}

float WavefrontVCM::getSplatScale() const
{
	return Tracer<true>::getSplatScale() * (w * h) / m_uNumLightRays;
}

}