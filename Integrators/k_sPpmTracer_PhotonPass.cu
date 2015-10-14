#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

CUDA_FUNC_IN float SH(float f)
{
	return f >= 0 ? 1 : 0;
}
void k_BeamBeamGrid::StoreBeam(const k_Beam& b, bool firstStore)
{
	unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
	if (beam_idx < m_uBeamLength)
	{
		m_pDeviceBeams[beam_idx] = b;

		AABB box = m_sStorage.hashMap.getAABB();
		Ray r(b.pos, b.dir);
		float m_fGridSize = m_sStorage.hashMap.m_fGridSize;
		float tEnd = b.t;

		Vec3f m_vCellSize = box.Size() / (m_fGridSize - 1);
		Vec3i Step(sign<int>(r.direction.x), sign<int>(r.direction.y), sign<int>(r.direction.z));
		Vec3f inv_d = r.direction;
		const float ooeps = math::exp2(-40.0f);//80 is too small, will create underflow on GPU
		inv_d.x = 1.0f / (math::abs(inv_d.x) > ooeps ? inv_d.x : copysignf(ooeps, inv_d.x));
		inv_d.y = 1.0f / (math::abs(inv_d.y) > ooeps ? inv_d.y : copysignf(ooeps, inv_d.y));
		inv_d.z = 1.0f / (math::abs(inv_d.z) > ooeps ? inv_d.z : copysignf(ooeps, inv_d.z));
		Vec3f DeltaT = abs(m_vCellSize * inv_d);

		Vec3f NextCrossingT[5];
		Vec3u Pos[5];
		float rayT[5];
		float maxT[5];
		//coordinate system which has left axis pointing towards (-1, 0, 0) and up to (0, 1, 0)
		Frame T(Vec3f(-math::abs(r.direction.z), 0, math::sign(r.direction.z) * r.direction.x),
			Vec3f(-r.direction.x * r.direction.y, math::sqr(r.direction.x) + math::sqr(r.direction.z), -r.direction.y * r.direction.z),
			r.direction);

		int nRaysTerminated = 0;
		float r_ = m_fCurrentRadiusVol;
		for (int i = 0; i < 5; i++)
		{
			Vec3f pos = i == 0 ? r.origin : (r.origin + T.toWorld(Vec3f(-r_ + ((i - 1) / 2) * 2 * r_, -r_ + ((i - 1) % 2) * 2 * r_, 0)));
			if (!box.Intersect(Ray(pos, r.direction), rayT + i, maxT + i))
			{
				rayT[i] = -1;
				nRaysTerminated++;
				continue;
			}
			rayT[i] = math::clamp(rayT[i], 0.0f, tEnd);
			maxT[i] = math::clamp(maxT[i], 0.0f, tEnd);
			Vec3f q = (r.direction * rayT[i] + pos - box.minV) / box.Size() * (m_fGridSize - 1);
			Pos[i] = clamp(Vec3u(unsigned int(q.x), unsigned int(q.y), unsigned int(q.z)), Vec3u(0), Vec3u(m_fGridSize - 1));
			auto A = box.minV + (Vec3f(Pos[i].x, Pos[i].y, Pos[i].z) + Vec3f(SH(r.direction.x), SH(r.direction.y), SH(r.direction.z))) * m_vCellSize,
				B = pos - r.direction * rayT[i];
			NextCrossingT[i] = max(Vec3f(0.0f), Vec3f(rayT[i]) + (A - B) * inv_d);
		}
		int N = 0;
		Vec3u lastMin(UINT_MAX), lastMax(UINT_MAX);
		while (nRaysTerminated != 5 && N++ < 250)
		{
			Vec3u minG(UINT_MAX), maxG(0);
			for (int i = 0; i < 5; i++)
				if (rayT[i] >= 0)
				{
					minG = min(minG, Pos[i]);
					maxG = max(maxG, Pos[i]);
				}
			for (unsigned int a = minG.x; a <= maxG.x; a++)
				for (unsigned int b = minG.y; b <= maxG.y; b++)
					for (unsigned int c = minG.z; c <= maxG.z; c++)
					{
						if (lastMin.x <= a && a <= lastMax.x && lastMin.y <= b && b <= lastMax.y && lastMin.z <= c && c <= lastMax.z)
							continue;
						m_sStorage.store(Vec3u(a, b, c), beam_idx);
					}
			lastMin = minG; lastMax = maxG;

			for (int i = 0; i < 5; i++)
			{
				if (rayT[i] < 0)
					continue;
				int bits = ((NextCrossingT[i][0] < NextCrossingT[i][1]) << 2) + ((NextCrossingT[i][0] < NextCrossingT[i][2]) << 1) + ((NextCrossingT[i][1] < NextCrossingT[i][2]));
				int stepAxis = (0x00000a66 >> (2 * bits)) & 3;
				Pos[i][stepAxis] += Step[stepAxis];
				if (Pos[i][stepAxis] >= m_fGridSize || rayT[i] > maxT[i])
				{
					nRaysTerminated++;
					rayT[i] = -1;
					continue;
				}
				rayT[i] = NextCrossingT[i][stepAxis];
				NextCrossingT[i][stepAxis] += DeltaT[stepAxis];
			}
		}

		if (firstStore)
			atomicInc(&m_uNumEmitted, (unsigned int)-1);
	}
}

CUDA_DEVICE unsigned int g_NumPhotonEmitted;
CUDA_DEVICE e_SpatialLinkedMap<k_pPpmPhoton> g_SurfaceMap;
CUDA_DEVICE CUDA_ALIGN(16) unsigned char g_VolEstimator[Dmax4(sizeof(k_PointStorage), sizeof(k_BeamGrid), sizeof(k_BeamBeamGrid), sizeof(k_BeamBVHStorage))];

template<typename VolEstimator> __global__ void k_PhotonPass(int photons_per_thread,  bool DIRECT)
{
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = photons_per_thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;
	CUDA_SHARED unsigned int numStoredSurface;
	numStoredSurface = 0;
	__syncthreads();

	while (atomicInc(&local_Counter, (unsigned int)-1) < local_Todo && !g_SurfaceMap.isFull() && !((VolEstimator*)g_VolEstimator)->isFullK())
	{
		Ray r;
		const e_KernelLight* light;
		Vec2f sps = rng.randomFloat2(), sds = rng.randomFloat2();
		Spectrum Le = g_SceneData.sampleEmitterRay(r, light, sps, sds),
			throughput(1.0f);
		int depth = -1;
		bool wasStoredSurface = false, wasStoredVolume = false;
		bool delta = false;
		MediumSamplingRecord mRec;
		bool medium = false;
		const e_VolumeRegion* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && !g_SurfaceMap.isFull() && !Le.isZero() && !((VolEstimator*)g_VolEstimator)->isFullK())
		{
			TraceResult r2 = k_TraceRay(r);
			float minT, maxT;
			if ((!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT) && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && bssrdf->sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec)))
			{
				((VolEstimator*)g_VolEstimator)->StoreBeam(k_Beam(r.origin, r.direction, mRec.t, throughput * Le), !wasStoredVolume);
				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				((VolEstimator*)g_VolEstimator)->StorePhoton(mRec.p, -r.direction, throughput * Le, !wasStoredVolume);
				wasStoredVolume = true;
				if (bssrdf)
				{
					PhaseFunctionSamplingRecord mRec(-r.direction);
					throughput *= bssrdf->As()->Func.Sample(mRec, rng);
					r.direction = mRec.wi;
				}
				else throughput *= V.Sample(mRec.p, -r.direction, rng, &r.direction);
				r.origin = mRec.p;
				delta = false;
				medium = true;
			}
			else if (!r2.hasHit())
				break;
			else
			{
				if (medium)
					throughput *= mRec.transmittance / mRec.pdfFailure;
				Vec3f wo = bssrdf ? r.direction : -r.direction;
				r2.getBsdfSample(-wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &rng);
				if ((DIRECT && depth > 0) || !DIRECT)
					if (r2.getMat().bsdf.hasComponent(ESmooth) && dot(bRec.dg.sys.n, wo) > 0.0f)
					{
						auto ph = k_pPpmPhoton(throughput * Le, wo, bRec.dg.sys.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse);
						ph.Pos = dg.P;
						bool b = g_SurfaceMap.store(dg.P, ph);
						if (b && !wasStoredSurface)
							atomicInc(&numStoredSurface, (unsigned int)-1);
						wasStoredSurface = true;
					}
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				delta = bRec.sampledType & ETypeCombinations::EDelta;
				if (!bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
					bRec.wo.z *= -1.0f;
				else
				{
					if (!bssrdf)
						throughput *= f;
					bssrdf = 0;
					medium = false;
				}

				r = Ray(bRec.dg.P, bRec.getOutgoing());
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
		atomicAdd(&g_NumPhotonEmitted, numStoredSurface);

	g_RNGData(rng);
}

void k_sPpmTracer::doPhotonPass()
{
	m_sSurfaceMap.ResetBuffer();
	m_pVolumeEstimator->StartNewPass(this, m_pScene);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ZeroSymbol(g_NumPhotonEmitted);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	while (!m_sSurfaceMap.isFull() && !m_pVolumeEstimator->isFull())
	{
		if (dynamic_cast<k_PointStorage*>(m_pVolumeEstimator))
			k_PhotonPass<k_PointStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bDirect);
		else if (dynamic_cast<k_BeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<k_BeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bDirect);
		else if (dynamic_cast<k_BeamBeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<k_BeamBeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bDirect);
		else if (dynamic_cast<k_BeamBVHStorage*>(m_pVolumeEstimator))
			k_PhotonPass<k_BeamBVHStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bDirect);
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sSurfaceMap, g_SurfaceMap, sizeof(m_sSurfaceMap)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(m_pVolumeEstimator, g_VolEstimator, m_pVolumeEstimator->getSize()));
	}
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPass, g_NumPhotonEmitted, sizeof(m_uPhotonEmittedPass)));
	m_pVolumeEstimator->PrepareForRendering();
	m_uPhotonEmittedPass = max(m_uPhotonEmittedPass, m_pVolumeEstimator->getNumEmitted());
	if (m_uTotalPhotonsEmitted == 0)
		doPerPixelRadiusEstimation();
}