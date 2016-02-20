#include "WavefrontVCM.h"

namespace CudaTracerLib {

WavefrontVCM::WavefrontVCM(unsigned int a_NumLightRays)
	: m_uNumLightRays(a_NumLightRays), m_sLightBufA(a_NumLightRays), m_sLightBufB(a_NumLightRays), m_sCamBufA(BLOCK_SAMPLER_BlockSize * BLOCK_SAMPLER_BlockSize), m_sCamBufB(BLOCK_SAMPLER_BlockSize * BLOCK_SAMPLER_BlockSize)
{
	ThrowCudaErrors(CUDA_MALLOC(&m_pDeviceLightVertices, sizeof(BPTVertex) * MAX_LIGHT_SUB_PATH_LENGTH * a_NumLightRays));

	int gridLength = 200;
	int numPhotons = a_NumLightRays * MAX_LIGHT_SUB_PATH_LENGTH;
	m_sPhotonMapsNext = VCMSurfMap(Vec3u(gridLength), numPhotons);
}

WavefrontVCM::~WavefrontVCM()
{
	m_sLightBufA.Free();
	m_sLightBufB.Free();
	m_sCamBufA.Free();
	m_sCamBufB.Free();
}

CUDA_CONST float mMisVcWeightFactor;
CUDA_CONST float mMisVmWeightFactor;
CUDA_CONST float mLightSubPathCount;

CUDA_DEVICE k_WVCM_LightBuffer g_sLightBufA, g_sLightBufB;

CUDA_DEVICE VCMSurfMap g_NextMap2;

CUDA_DEVICE k_WVCM_CamBuffer g_sCamBufA, g_sCamBufB;

CUDA_GLOBAL void createLightRays(unsigned int g_DeviceNumLightPaths)
{
	CudaRNG rng = g_RNGData();
	unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx < g_DeviceNumLightPaths)
	{
		BPTSubPathState state;
		sampleEmitter(state, rng, mMisVcWeightFactor);
		auto& ent = g_sLightBufA(idx);
		ent.m_uVertexStart = idx;
		ent.state = state;
		g_sLightBufA.operator()(idx, 0).a = Vec4f(state.r.ori(), 0);
		g_sLightBufA.operator()(idx, 0).b = Vec4f(state.r.dir(), FLT_MAX);
	}
	g_RNGData(rng);
}

CUDA_GLOBAL void extendLighRays(unsigned int N, BPTVertex* g_pLightVertices, Image I, int iteration)
{
	CudaRNG rng = g_RNGData();
	unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		auto ent = g_sLightBufA(idx);
		unsigned int vIdx = ent.m_uVertexStart & 0x00ffffff, vOff = (ent.m_uVertexStart & 0xff000000) >> 24;
		auto res = g_sLightBufA.res(idx, 0);
		if (res.dist)
		{
			TraceResult r2;
			res.toResult(&r2, g_SceneData);
			BPTVertex v;
			v.mat = 0;
			r2.getBsdfSample(ent.state.r, v.bRec, ETransportMode::EImportance, &rng, &ent.state.throughput);

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

			auto ph = k_MISPhoton(v.throughput, -ent.state.r.dir(), v.bRec.dg.sys.n, PhotonType::pt_Diffuse, v.dVC, v.dVCM, v.dVM);
			Vec3u cell_idx = g_NextMap2.getHashGrid().Transform(v.bRec.dg.P);
			ph.setPos(g_NextMap2.getHashGrid(), cell_idx, v.bRec.dg.P);
			if (!g_NextMap2.store(cell_idx, ph))
				printf("WVCM : not enough photon storage allocated!\n");

			if (r2.getMat().bsdf.hasComponent(ESmooth))
				connectToCamera(ent.state, v.bRec, r2.getMat(), I, rng, mLightSubPathCount, mMisVmWeightFactor, 1, true);

			if (vOff < MAX_LIGHT_SUB_PATH_LENGTH - 1 && sampleScattering(ent.state, v.bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			{
				ent.m_uVertexStart = ((vOff + 1) << 24) | vIdx;
				unsigned int newRayIdx = g_sLightBufB.insertRay(0);
				g_sLightBufB(newRayIdx) = ent;
				g_sLightBufB(newRayIdx, 0).a = Vec4f(ent.state.r.ori(), 0);
				g_sLightBufB(newRayIdx, 0).b = Vec4f(ent.state.r.dir(), FLT_MAX);
			}
		}
	}
	g_RNGData(rng);
}

void WavefrontVCM::DoRender(Image* I)
{
	m_uLightOff = 0;
	k_INITIALIZE(m_pScene, g_sRngs);
	float a_Radius = this->getCurrentRadius(2);
	const float etaVCM = (PI * a_Radius * a_Radius) * m_uNumLightRays;
	float MisVmWeightFactor = 1;
	float MisVcWeightFactor = 1.0f / etaVCM;
	float one = 1;
	ThrowCudaErrors(cudaMemset(m_pDeviceLightVertices, 0, sizeof(BPTVertex) * MAX_LIGHT_SUB_PATH_LENGTH * m_uNumLightRays));
	ThrowCudaErrors(cudaMemcpyToSymbol(mMisVcWeightFactor, &MisVcWeightFactor, sizeof(MisVcWeightFactor)));
	ThrowCudaErrors(cudaMemcpyToSymbol(mMisVmWeightFactor, &MisVmWeightFactor, sizeof(MisVmWeightFactor)));
	ThrowCudaErrors(cudaMemcpyToSymbol(mLightSubPathCount, &one, sizeof(one)));

	m_sLightBufA.Clear();
	CopyToSymbol(g_sLightBufA, m_sLightBufA);
	createLightRays << <m_uNumLightRays / (32 * 6) + 1, dim3(32, 6) >> >(m_uNumLightRays);
	ThrowCudaErrors(cudaThreadSynchronize());
	m_sLightBufA.setNumRays(m_uNumLightRays, 0);

	m_sPhotonMapsNext.ResetBuffer();
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NextMap2, &m_sPhotonMapsNext, sizeof(m_sPhotonMapsNext)));

	k_WVCM_LightBuffer* srcBuf = &m_sLightBufA, *destBuf = &m_sLightBufB;
	int i = 0;
	do
	{
		destBuf->Clear();
		srcBuf->IntersectBuffers<false>(false);
		ThrowCudaErrors(cudaMemcpyToSymbol(g_sLightBufA, srcBuf, sizeof(*srcBuf)));
		ThrowCudaErrors(cudaMemcpyToSymbol(g_sLightBufB, destBuf, sizeof(*destBuf)));
		extendLighRays << <srcBuf->getNumRays(0) / (32 * 6) + 1, dim3(32, 6) >> >(srcBuf->getNumRays(0), m_pDeviceLightVertices, *I, i++);
		ThrowCudaErrors(cudaThreadSynchronize());
		ThrowCudaErrors(cudaMemcpyFromSymbol(srcBuf, g_sLightBufA, sizeof(*srcBuf)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(destBuf, g_sLightBufB, sizeof(*destBuf)));
		swapk(srcBuf, destBuf);

	} while (srcBuf->getNumRays(0));

	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sPhotonMapsNext, g_NextMap2, sizeof(m_sPhotonMapsNext)));

	Tracer<true, true>::DoRender(I);
}

void WavefrontVCM::StartNewTrace(Image* I)
{
	Tracer<true, true>::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
	m_sEyeBox = m_sEyeBox.Extend(0.1f);
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	m_sPhotonMapsNext.SetSceneDimensions(m_sEyeBox);
}

CUDA_GLOBAL void createCameraRays(int xoff, int yoff, int blockW, int blockH, int w, int h)
{
	CudaRNG rng = g_RNGData();
	Vec2i pixel = TracerBase::getPixelPos(xoff, yoff);
	unsigned int idx = (pixel.y - yoff) * blockW + (pixel.x - xoff);
	if (pixel.x < w && pixel.y < h)
	{
		BPTSubPathState state;
		sampleCamera(state, rng, Vec2f(pixel.x, pixel.y), mLightSubPathCount);
		auto& ent = g_sCamBufA(idx);
		ent.x = pixel.x;
		ent.y = pixel.y;
		ent.acc = Spectrum(0.0f);
		ent.state = state;
		g_sCamBufA.operator()(idx, 0).a = Vec4f(state.r.ori(), 0);
		g_sCamBufA.operator()(idx, 0).b = Vec4f(state.r.dir(), FLT_MAX);
	}
	g_RNGData(rng);
}

CUDA_GLOBAL void performPPMEstimate(unsigned int N, float a_Radius, float nPhotons)
{
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	CudaRNG rng = g_RNGData();
	unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		auto ent = g_sCamBufA(idx);
		auto res = g_sCamBufA.res(idx, 0);
		if (res.dist)
		{
			TraceResult r2;
			res.toResult(&r2, g_SceneData);

			r2.getBsdfSample(ent.state.r, bRec, ETransportMode::ERadiance, &rng);

			ent.state.dVCM *= r2.m_fDist * r2.m_fDist;
			ent.state.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVC /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVM /= math::abs(Frame::cosTheta(bRec.wi));

			Spectrum phL;
			if (!r2.getMat().bsdf.hasComponent(EGlossy))
				phL = L_Surface2<false>(g_NextMap2, ent.state, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
			else phL = L_Surface2<true>(g_NextMap2, ent.state, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
			g_sCamBufA(idx).acc = ent.acc + ent.state.throughput * phL;
		}
	}
	g_RNGData(rng);
}

CUDA_GLOBAL void extendCameraRays(unsigned int N, Image I, int iteration, bool lastIteration, float a_Radius, unsigned int lightOff, unsigned int numLightPaths, BPTVertex* g_pLightVertices)
{
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	CudaRNG rng = g_RNGData();
	unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		bool extended = false;
		auto ent = g_sCamBufA(idx);
		auto res = g_sCamBufA.res(idx, 0);
		if (res.dist)
		{
			TraceResult r2;
			res.toResult(&r2, g_SceneData);

			r2.getBsdfSample(ent.state.r, bRec, ETransportMode::ERadiance, &rng);

			ent.state.dVCM *= r2.m_fDist * r2.m_fDist;
			ent.state.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVC /= math::abs(Frame::cosTheta(bRec.wi));
			ent.state.dVM /= math::abs(Frame::cosTheta(bRec.wi));

			if (r2.LightIndex() != UINT_MAX)
				ent.acc += ent.state.throughput * gatherLight(ent.state, bRec, r2, rng, iteration + 1, true);

			if (r2.getMat().bsdf.hasComponent(ESmooth))
			{
				ent.acc += ent.state.throughput * connectToLight(ent.state, bRec, r2.getMat(), rng, mMisVmWeightFactor, true);

				unsigned int vOff = ((lightOff + idx) % numLightPaths) * MAX_LIGHT_SUB_PATH_LENGTH, i = 0;
				while (i < MAX_LIGHT_SUB_PATH_LENGTH && g_pLightVertices[vOff + i].mat)
				{
					BPTVertex& v = g_pLightVertices[vOff + i];
					BPTVertex lv;
					lv.dg = v.dg;
					lv.dVC = v.dVC;
					lv.dVM = v.dVM;
					lv.dVCM = v.dVCM;
					lv.mat = v.mat;
					lv.subPathLength = v.subPathLength;
					lv.throughput = v.throughput;
					lv.bRec = BSDFSamplingRecord(lv.dg);
					lv.bRec.eta = v.bRec.eta;
					lv.bRec.mode = v.bRec.mode;
					lv.bRec.rng = v.bRec.rng;
					lv.bRec.sampledType = v.bRec.sampledType;
					lv.bRec.typeMask = v.bRec.typeMask;
					lv.bRec.wi = v.bRec.wi;
					lv.bRec.wo = v.bRec.wo;
					ent.acc += ent.state.throughput * lv.throughput * connectVertices(lv, ent.state, bRec, r2.getMat(), mMisVcWeightFactor, mMisVmWeightFactor, true);
					i++;
				}
			}

			if (!lastIteration && sampleScattering(ent.state, bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			{
				extended = true;
				unsigned int newRayIdx = g_sCamBufB.insertRay(0);
				g_sCamBufB(newRayIdx) = ent;
				g_sCamBufB(newRayIdx, 0).a = Vec4f(ent.state.r.ori(), 0);
				g_sCamBufB(newRayIdx, 0).b = Vec4f(ent.state.r.dir(), FLT_MAX);
			}
		}
		if (!extended)
			I.AddSample(ent.x, ent.y, ent.acc);
	}
	g_RNGData(rng);
}

void WavefrontVCM::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	ThrowCudaErrors(cudaMemcpyToSymbol(g_sCamBufA, &m_sCamBufA, sizeof(m_sCamBufA)));
	m_sCamBufA.Clear();
	createCameraRays << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(x, y, blockW, blockH, w, h);
	ThrowCudaErrors(cudaThreadSynchronize());
	m_sCamBufA.setNumRays(BLOCK_SAMPLER_BlockSize * BLOCK_SAMPLER_BlockSize, 0);

	k_WVCM_CamBuffer* srcBuf = &m_sCamBufA, *destBuf = &m_sCamBufB;
	int i = 0;
	do
	{
		destBuf->Clear();
		srcBuf->IntersectBuffers<false>(false);
		ThrowCudaErrors(cudaMemcpyToSymbol(g_sCamBufA, srcBuf, sizeof(*srcBuf)));
		ThrowCudaErrors(cudaMemcpyToSymbol(g_sCamBufB, destBuf, sizeof(*destBuf)));
		performPPMEstimate << <srcBuf->getNumRays(0) / (32 * 6) + 1, dim3(32, 6) >> >(srcBuf->getNumRays(0), getCurrentRadius(2), (float)m_uNumLightRays);
		extendCameraRays << <srcBuf->getNumRays(0) / (32 * 6) + 1, dim3(32, 6) >> >(srcBuf->getNumRays(0), *I, i++, i == 4, getCurrentRadius(2), m_uLightOff, m_uNumLightRays, m_pDeviceLightVertices);
		ThrowCudaErrors(cudaThreadSynchronize());
		ThrowCudaErrors(cudaMemcpyFromSymbol(srcBuf, g_sCamBufA, sizeof(*srcBuf)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(destBuf, g_sCamBufB, sizeof(*destBuf)));
		swapk(srcBuf, destBuf);

	} while (srcBuf->getNumRays(0) && i < 6);
	m_uLightOff += blockW * blockH;
}

float WavefrontVCM::getSplatScale()
{
	return Tracer<true, true>::getSplatScale() * (w * h) / m_uNumLightRays;
}

}