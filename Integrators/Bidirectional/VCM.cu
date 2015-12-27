#include "VCM.h"

namespace CudaTracerLib {

CUDA_DEVICE VCMSurfMap g_CurrentMap, g_NextMap;

CUDA_FUNC_IN void _VCM(const Vec2f& pixelPosition, BlockSampleImage& img, CudaRNG& rng, int w, int h, float a_Radius, int a_NumIteration, float nPhotons)
{
	float mLightSubPathCount = 1;
	const float etaVCM = (PI * a_Radius * a_Radius) * w * h;
	float mMisVmWeightFactor = 1;
	float mMisVcWeightFactor = 1.0f / etaVCM;

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
		r2.getBsdfSample(lightPathState.r, v.bRec, ETransportMode::EImportance, &rng, &lightPathState.throughput);

		if (emitterPathLength > 1 || true)
			lightPathState.dVCM *= r2.m_fDist * r2.m_fDist;
		lightPathState.dVCM /= math::abs(Frame::cosTheta(v.bRec.wi));
		lightPathState.dVC /= math::abs(Frame::cosTheta(v.bRec.wi));
		lightPathState.dVM /= math::abs(Frame::cosTheta(v.bRec.wi));

		//store in list
		if (r2.getMat().bsdf.hasComponent(ESmooth))
		{
			v.dVCM = lightPathState.dVCM;
			v.dVC = lightPathState.dVC;
			v.dVM = lightPathState.dVM;
			v.throughput = lightPathState.throughput;
			v.mat = &r2.getMat();
			v.subPathLength = emitterPathLength + 1;
			emitterVerticesStored++;

#ifdef ISCUDA
			auto ph = k_MISPhoton(v.throughput, -lightPathState.r.direction, v.bRec.dg.sys.n, PhotonType::pt_Diffuse, v.dVC, v.dVCM, v.dVM);
			Vec3u cell_idx = g_NextMap.getHashGrid().Transform(v.bRec.dg.P);
			ph.setPos(g_NextMap.getHashGrid(), cell_idx, v.bRec.dg.P);
			if (!g_NextMap.store(cell_idx, ph))
				printf("VCM : not enough photon storage allocated!\n");
#endif
		}

		//connect to camera
		if (r2.getMat().bsdf.hasComponent(ESmooth))
			connectToCamera(lightPathState, v.bRec, r2.getMat(), img.img, rng, mLightSubPathCount, mMisVmWeightFactor, 1, true);

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

			break;
		}

		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(cameraState.r, bRec, ETransportMode::ERadiance, &rng);

		cameraState.dVCM *= r2.m_fDist * r2.m_fDist;
		cameraState.dVCM /= math::abs(Frame::cosTheta(bRec.wi));
		cameraState.dVC /= math::abs(Frame::cosTheta(bRec.wi));
		cameraState.dVM /= math::abs(Frame::cosTheta(bRec.wi));

		if (r2.LightIndex() != UINT_MAX)
		{
			acc += cameraState.throughput * gatherLight(cameraState, bRec, r2, rng, camPathLength, true);
			break;
		}

		if (r2.getMat().bsdf.hasComponent(ESmooth))
		{
			//acc += cameraState.throughput * connectToLight(cameraState, bRec, r2.getMat(), rng, mMisVmWeightFactor, true);

			for (int emitterVertexIdx = 0; emitterVertexIdx < emitterVerticesStored; emitterVertexIdx++)
			{
				BPTVertex lv = lightPath[emitterVertexIdx];
				acc += cameraState.throughput * lv.throughput * connectVertices(lv, cameraState, bRec, r2.getMat(), mMisVcWeightFactor, mMisVmWeightFactor, true);
			}
			
			//scale by 2 to account for no merging in the first iteration
#ifdef ISCUDA
			if(a_NumIteration > 1)
			{
				Spectrum phL;
				if (!r2.getMat().bsdf.hasComponent(EGlossy))
					phL = L_Surface2<false>(g_CurrentMap, cameraState, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
				else phL = L_Surface2<true>(g_CurrentMap, cameraState, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, nPhotons, true);
				acc += cameraState.throughput * (a_NumIteration == 2 ? 2 : 1) * phL;
			}
#endif
		}

		if (!sampleScattering(cameraState, bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			break;
	}

	img.Add(pixelPosition.x, pixelPosition.y, acc);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, BlockSampleImage img, float a_Radius, int a_NumIteration, float nPhotons)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if (x < w && y < h)
		_VCM(Vec2f(x, y), img, rng, w, h, a_Radius, a_NumIteration, nPhotons);
	g_RNGData(rng);
}

void VCM::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	float radius = getCurrentRadius(2);
	pathKernel << < BLOCK_SAMPLER_LAUNCH_CONFIG >> >(w, h, x, y, m_pBlockSampler->getBlockImage(), radius, m_uPassesDone, w * h);
}

void VCM::DoRender(Image* I)
{
	m_sPhotonMapsNext.ResetBuffer();
	ThrowCudaErrors(cudaMemcpyToSymbol(g_CurrentMap, &m_sPhotonMapsCurrent, sizeof(m_sPhotonMapsCurrent)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NextMap, &m_sPhotonMapsNext, sizeof(m_sPhotonMapsNext)));

	Tracer<true, true>::DoRender(I);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sPhotonMapsNext, g_NextMap, sizeof(m_sPhotonMapsNext)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sPhotonMapsCurrent, g_CurrentMap, sizeof(m_sPhotonMapsCurrent)));

	swapk(m_sPhotonMapsNext, m_sPhotonMapsCurrent);
	m_uPhotonsEmitted += w * h;
}

void VCM::StartNewTrace(Image* I)
{
	Tracer<true, true>::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
	m_sEyeBox = m_sEyeBox.Extend(0.1f);
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	m_sPhotonMapsCurrent.SetSceneDimensions(m_sEyeBox);
	m_sPhotonMapsNext.SetSceneDimensions(m_sEyeBox);
}

VCM::VCM()
{
	int gridLength = 250;
	int numPhotons = 1024 * 1024 * MAX_SUB_PATH_LENGTH;
	m_sPhotonMapsCurrent = VCMSurfMap(gridLength, numPhotons);
	m_sPhotonMapsNext = VCMSurfMap(gridLength, numPhotons);
}

}