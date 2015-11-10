#include "k_VCM.h"

CUDA_DEVICE k_PhotonMapCollection<false, k_MISPhoton> g_CurrentMap, g_NextMap;

CUDA_FUNC_IN void VCM(const Vec2f& pixelPosition, k_BlockSampleImage& img, CudaRNG& rng, int w, int h, float a_Radius, int a_NumIteration)
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
		TraceResult r2 = k_TraceRay(lightPathState.r);
		if (!r2.hasHit())
			break;

		BPTVertex& v = lightPath[emitterVerticesStored];
		r2.getBsdfSample(lightPathState.r, v.bRec, ETransportMode::EImportance, &rng);

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
			k_MISPhoton* photon;
			if (emitterPathLength > 1 && storePhoton(v.bRec.dg.P, v.throughput, -lightPathState.r.direction, v.bRec.dg.sys.n, PhotonType::pt_Diffuse, g_NextMap, &photon))
			{
				photon->dVC = v.dVC;
				photon->dVCM = v.dVCM;
				photon->dVM = v.dVM;
			}
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
		TraceResult r2 = k_TraceRay(cameraState.r);
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
			acc += cameraState.throughput * connectToLight(cameraState, bRec, r2.getMat(), rng, mMisVmWeightFactor, true);

			for (int emitterVertexIdx = 0; emitterVertexIdx < emitterVerticesStored; emitterVertexIdx++)
			{
				BPTVertex lv = lightPath[emitterVertexIdx];
				acc += cameraState.throughput * lv.throughput * connectVertices(lv, cameraState, bRec, r2.getMat(), mMisVcWeightFactor, mMisVmWeightFactor, true);
			}

			//scale by 2 to account for no merging in the first iteration
#ifdef ISCUDA
			acc += cameraState.throughput * (a_NumIteration == 2 ? 2 : 1) * L_Surface2(g_CurrentMap, cameraState, bRec, a_Radius, &r2.getMat(), mMisVcWeightFactor, true);
#endif
		}

		if (!sampleScattering(cameraState, bRec, r2.getMat(), rng, mMisVcWeightFactor, mMisVmWeightFactor))
			break;
	}

	img.Add(pixelPosition.x, pixelPosition.y, acc);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, k_BlockSampleImage img, float a_Radius, int a_NumIteration)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if (x < w && y < h)
		VCM(Vec2f(x, y), img, rng, w, h, a_Radius, a_NumIteration);
	g_RNGData(rng);
}

/*__global__ void buildHashGrid2()
{
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x + blockDim.x * blockDim.y * blockIdx.x;
	if (idx < g_NextMap.m_uPhotonNumEmitted)
	{
		k_pPpmPhoton& e = g_NextMap.m_pPhotons[idx];
		Vec3f pos = g_NextMap.m_pPhotonPositions[idx];
		const k_PhotonMap<k_HashGrid_Reg>& map = (&g_NextMap.m_sSurfaceMap)[e.getType()];
		e.setPos(map.m_sHash, map.m_sHash.Transform(pos), pos);
		unsigned int i = map.m_sHash.Hash(pos);
		unsigned int k = atomicExch(map.m_pDeviceHashGrid + i, idx);
		e.setNext(k);
	}
}*/

void k_VCM::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	float radius = getCurrentRadius(2);
	pathKernel << < numBlocks, threadsPerBlock >> >(w, h, x, y, m_pBlockSampler->getBlockImage(), radius, m_uPassesDone);
}

void k_VCM::DoRender(e_Image* I)
{
	m_sPhotonMapsNext.m_uPhotonNumEmitted = w * h;
	cudaMemcpyToSymbol(g_CurrentMap, &m_sPhotonMapsCurrent, sizeof(m_sPhotonMapsCurrent));
	cudaMemcpyToSymbol(g_NextMap, &m_sPhotonMapsNext, sizeof(m_sPhotonMapsNext));

	k_Tracer<true, true>::DoRender(I);
	cudaMemcpyFromSymbol(&m_sPhotonMapsNext, g_NextMap, sizeof(m_sPhotonMapsNext));
	//buildHashGrid2 << <m_sPhotonMapsNext.m_uPhotonBufferLength / (32 * 6) + 1, dim3(32, 6, 1) >> >();
	cudaMemcpyFromSymbol(&m_sPhotonMapsCurrent, g_CurrentMap, sizeof(m_sPhotonMapsCurrent));
	cudaMemcpyFromSymbol(&m_sPhotonMapsNext, g_NextMap, sizeof(m_sPhotonMapsNext));

	swapk(m_sPhotonMapsNext, m_sPhotonMapsCurrent);
	m_uPhotonsEmitted += m_sPhotonMapsCurrent.m_uPhotonNumEmitted;

	m_sPhotonMapsNext.StartNewPass();
}

void k_VCM::StartNewTrace(e_Image* I)
{
	k_Tracer<true, true>::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
	m_sEyeBox = m_sEyeBox.Extend(0.1f);
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	m_sPhotonMapsCurrent.StartNewRendering(m_sEyeBox, m_sEyeBox, r);
	m_sPhotonMapsCurrent.StartNewPass();
	m_sPhotonMapsNext.StartNewRendering(m_sEyeBox, m_sEyeBox, r);
	m_sPhotonMapsNext.StartNewPass();
}

k_VCM::k_VCM()
{
	int gridLength = 100;
	int numPhotons = 1024 * 1024 * 5;
	m_sPhotonMapsCurrent = k_PhotonMapCollection<false, k_MISPhoton>(numPhotons, gridLength*gridLength*gridLength, UINT_MAX);
	m_sPhotonMapsNext = k_PhotonMapCollection<false, k_MISPhoton>(numPhotons, gridLength*gridLength*gridLength, UINT_MAX);
}