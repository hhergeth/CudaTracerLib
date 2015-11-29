#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>

namespace CudaTracerLib {

CUDA_FUNC_IN float SH(float f)
{
	return f >= 0 ? 1 : 0;
}

CUDA_ONLY_FUNC void BeamBeamGrid::StoreBeam(const Beam& b, bool firstStore)
{
	unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
	if (beam_idx < m_uBeamLength)
	{
		m_pDeviceBeams[beam_idx] = b;
		bool storedAll = true;
#ifdef ISCUDA
		/*TraverseGrid(Ray(b.pos, b.dir), m_sStorage.hashMap, 0.0f, b.t, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
		{
		if (!m_sStorage.store(cell_pos, beam_idx))
		{
		storedAll = false;
		cancelTraversal = true;
		}
		});*/
#endif

		/*AABB box = m_sStorage.hashMap.getAABB();
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
		while (nRaysTerminated != 5)
		{
		N++;
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
		if (Pos[i][stepAxis] >= m_fGridSize || NextCrossingT[i][stepAxis] > maxT[i])
		{
		nRaysTerminated++;
		rayT[i] = -1;
		continue;
		}
		rayT[i] = NextCrossingT[i][stepAxis];
		NextCrossingT[i][stepAxis] += DeltaT[stepAxis];
		}
		}*/

#ifdef ISCUDA
		const AABB objaabb = b.getAABB(m_fCurrentRadiusVol);
		const int maxAxis = b.dir.abs().arg_max();
		const int chopCount = (int)(objaabb.Size()[maxAxis] * m_sStorage.hashMap.m_vInvSize[maxAxis]) + 1;
		const float invChopCount = 1.0f / (float)chopCount;

		for (int chop = 0; chop < chopCount; ++chop)
		{
			AABB aabb = b.getSegmentAABB((chop)* invChopCount, (chop + 1) * invChopCount, m_fCurrentRadiusVol);

			m_sStorage.ForAllCells(aabb.minV, aabb.maxV, [&](const Vec3u& pos)
			{
				/*bool found_duplicate = false;
				m_sStorage.ForAll(pos, [&](unsigned int loc_idx, unsigned int b_idx)
				{
					if (found_duplicate) return;
					if (beam_idx == b_idx)
						found_duplicate = true;
				});
				if (!found_duplicate)*/
					storedAll &= m_sStorage.store(pos, beam_idx);
			});
		}

		//auto aabb = b.getAABB(m_fCurrentRadiusVol);
		//m_sStorage.ForAllCells(aabb.minV, aabb.maxV, [&](const Vec3u& pos)
		//{
		//	storedAll &= m_sStorage.store(pos, beam_idx);
		//});

		if (firstStore && storedAll)
			atomicInc(&m_uNumEmitted, (unsigned int)-1);
#endif
	}
}

CUDA_CONST unsigned int g_PassIdx;
CUDA_DEVICE unsigned int g_NumPhotonEmitted;
CUDA_DEVICE SpatialLinkedMap<PPPMPhoton> g_SurfaceMap;
CUDA_DEVICE CUDA_ALIGN(16) unsigned char g_VolEstimator[Dmax4(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid), sizeof(BeamBVHStorage))];

template<typename VolEstimator> __global__ void k_PhotonPass(int photons_per_thread, bool DIRECT)
{
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = photons_per_thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	CUDA_SHARED unsigned int numStoredSurface;
	numStoredSurface = 0;
	__syncthreads();

	while (atomicInc(&local_Counter, (unsigned int)-1) < local_Todo)// && !g_SurfaceMap.isFull() && !((VolEstimator*)g_VolEstimator)->isFullK()
	{
		Ray r;
		const KernelLight* light;
		Vec2f sps = rng.randomFloat2(), sds = rng.randomFloat2();
		Spectrum Le = g_SceneData.sampleEmitterRay(r, light, sps, sds),
			throughput(1.0f);
		int depth = -1;
		bool wasStoredSurface = false, wasStoredVolume = false;
		bool delta = false;
		MediumSamplingRecord mRec;
		bool medium = false;
		const VolumeRegion* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && !Le.isZero())// && !g_SurfaceMap.isFull() && !((VolEstimator*)g_VolEstimator)->isFullK()
		{
			TraceResult r2 = Traceray(r);
			float minT, maxT;
			bool inMedium = (!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT)) || bssrdf;
			if (inMedium)
				((VolEstimator*)g_VolEstimator)->StoreBeam(Beam(r.origin, r.direction, r2.m_fDist, throughput * Le), !wasStoredVolume);//store the beam even if sampled distance is to far ahead!
			if ((!bssrdf && inMedium && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && bssrdf->sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec)))
			{//mRec.t
				throughput *= mRec.transmittance / mRec.pdfSuccess;
				((VolEstimator*)g_VolEstimator)->StorePhoton(mRec.p, -r.direction, throughput * Le, !wasStoredVolume);
				throughput *= mRec.sigmaS;
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
						auto ph = PPPMPhoton(throughput * Le, wo, bRec.dg.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse);
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

void PPPMTracer::doPhotonPass()
{
	m_sSurfaceMap.ResetBuffer();
	m_pVolumeEstimator->StartNewPass(this, m_pScene);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ZeroSymbol(g_NumPhotonEmitted);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_PassIdx, &m_uPassesDone, sizeof(m_uPassesDone)));

	while (!m_sSurfaceMap.isFull() && !m_pVolumeEstimator->isFull())
	{
		if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			k_PhotonPass<PointStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting);
		else if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamBeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting);
		else if (dynamic_cast<BeamBVHStorage*>(m_pVolumeEstimator))
			k_PhotonPass<BeamBVHStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting);
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sSurfaceMap, g_SurfaceMap, sizeof(m_sSurfaceMap)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(m_pVolumeEstimator, g_VolEstimator, m_pVolumeEstimator->getSize()));
	}
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPass, g_NumPhotonEmitted, sizeof(m_uPhotonEmittedPass)));
	m_pVolumeEstimator->PrepareForRendering();
	m_uPhotonEmittedPass = max(m_uPhotonEmittedPass, m_pVolumeEstimator->getNumEmitted());
	if (m_uTotalPhotonsEmitted == 0)
		doPerPixelRadiusEstimation();
}

}