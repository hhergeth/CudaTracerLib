#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

CUDA_DEVICE k_PhotonMapCollection<true, k_pPpmPhoton> g_Map;
CUDA_DEVICE bool g_HasVolumePhotons;
CUDA_DEVICE k_BeamGrid g_BeamGrid;
CUDA_DEVICE k_BeamBVHStorage g_BVHBeams;

CUDA_FUNC_IN bool storeBeam(const Ray& r, float t, const Spectrum& phi, float a_r)
{
	struct hashPos
	{
		unsigned long long flags;
		CUDA_FUNC_IN hashPos() : flags(0) {}
		CUDA_FUNC_IN int idx(const Vec3u& center, const Vec3u& pos)
		{
			Vec3u d = pos - center - Vec3u(7);
			if (d.x > pos.x || d.y > pos.y || d.z > pos.z)
				return -1;
			return d.z * 49 + d.y * 7 + d.x;
		}
		CUDA_FUNC_IN bool isSet(const Vec3u& center, const Vec3u& pos)
		{
			int i = idx(center, pos);
			return i == -1 ? false : (flags >> i) & 1;
		}
		CUDA_FUNC_IN void set(const Vec3u& center, const Vec3u& pos)
		{
			int i = idx(center, pos);
			if (i != -1)
				flags |= 1 << i;
		}
	};

#ifdef ISCUDA
	unsigned int beam_idx = atomicInc(&g_BeamGrid.m_uBeamIdx, 0xffffffff);
	if (beam_idx >= g_BeamGrid.m_uBeamLength)
		return false;
	g_BeamGrid.m_pDeviceBeams[beam_idx] = k_Beam(r.origin, r.direction, t, phi);
	const int N_L = 100;
	unsigned int cell_list[N_L];
	int i_L = 0;
	unsigned int M = 0xffffffff;
	//Vec3u lastMin(M), lastMax(M);
	hashPos H1, H2;
	TraverseGrid(r, g_Map.m_sVolumeMap.m_sHash, 0, t, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		H1 = H2;
		Frame f_ray(r.direction);
		Vec3f disk_a = r(rayT) + f_ray.toWorld(Vec3f(-1, -1, 0)), disk_b = r(rayT) + f_ray.toWorld(Vec3f(1, 1, 0));
		Vec3f min_disk = min(disk_a, disk_b), max_disk = max(disk_a, disk_b);
		Vec3u min_cell = g_Map.m_sVolumeMap.m_sHash.Transform(min_disk), max_cell = g_Map.m_sVolumeMap.m_sHash.Transform(max_disk);
		for (unsigned int ax = min_cell.x; ax <= max_cell.x; ax++)
			for (unsigned int ay = min_cell.y; ay <= max_cell.y; ay++)
				for (unsigned int az = min_cell.z; az <= max_cell.z; az++)
				{
					//if (lastMin.x <= ax && ax <= lastMax.x && lastMin.y <= ay && ay <= lastMax.y && lastMin.z <= az && az <= lastMax.z)
					//	continue;

					unsigned int grid_idx = g_Map.m_sVolumeMap.m_sHash.Hash(Vec3u(ax,ay,az));

					bool found = false;
					for (int i = 0; i < i_L; i++)
						if (cell_list[i] == grid_idx)
						{
							found = true;
							break;
						}
					if (found)
						continue;
					if (i_L < N_L)
						cell_list[i_L++] = grid_idx;
					unsigned int grid_entriy_idx = atomicInc(&g_BeamGrid.m_uGridIdx, 0xffffffff);
					if (grid_entriy_idx >= g_BeamGrid.m_uGridLength)
						return;
					unsigned int next_grid_entry_idx = atomicExch(&g_BeamGrid.m_pGrid[grid_idx].y, grid_entriy_idx);
					g_BeamGrid.m_pGrid[grid_entriy_idx] = Vec2i(beam_idx, next_grid_entry_idx);
				}
		if (i_L == N_L)
			printf("cell_list full\n");
		i_L = 0;
		//lastMin = min_cell;
		//lastMax = max_cell;
	});
	//printf("i_L = %d   ", i_L);
#endif
	return true;
}
CUDA_FUNC_IN AABB calculatePlaneAABBInCell(const AABB& cell, const Vec3f& p, const Vec3f& n, float r)
{
	Vec3f cellSize = cell.Size();
	AABB res = AABB::Identity();
	for (int a = 0; a < 3; a++)
	{
		Vec3f d(0.0f); d[a] = 1;
		for (int l = 0; l < 4; l++)
		{
			Vec3f a1(0.0f); a1[(a + 1) % 3] = cellSize[(a + 1) % 3];
			Vec3f o = l == 0 ? cell.minV : (l == 2 ? cell.maxV : (l == 1 ? cell.minV + a1 : cell.maxV - a1));
			d = l < 2 ? d : -d;
			if (n[a] != 0)
			{
				float lambda = p[a] - o[a];
				if (lambda > 0 && lambda < cellSize[a])
				{
					Vec3f x = o + lambda * d;
					x = p + (p - x) * min(distance(p, x), r);
					res = res.Extend(x);
				}
			}
		}
	}
	return res;
}
template<bool DIRECT> __global__ void k_PhotonPass(int photons_per_thread, bool final_gather, float a_r)
{
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = photons_per_thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;
	bool hasVolPhotons;

	while (atomicInc(&local_Counter, (unsigned int)-1) < local_Todo && g_Map.m_uPhotonNumStored < g_Map.m_uPhotonBufferLength)
	{
		Ray r;
		const e_KernelLight* light;
		Vec2f sps = rng.randomFloat2(), sds = rng.randomFloat2();
		Spectrum Le = g_SceneData.sampleEmitterRay(r, light, sps, sds),
			throughput(1.0f);
		int depth = -1;
		bool wasStored = false, wasStored2 = false;
		bool delta = false;
		MediumSamplingRecord mRec;
		bool medium = false;
		const e_KernelBSSRDF* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && g_Map.m_uPhotonNumStored < g_Map.m_uPhotonBufferLength && !Le.isZero())
		{
			TraceResult r2 = k_TraceRay(r);
			float minT, maxT;
			if ((!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT) && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && sampleDistanceHomogenous(r, 0, r2.m_fDist, rng.randomFloat(), mRec, bssrdf->sig_a, bssrdf->sigp_s)))
			{
				if (g_BeamGrid.m_pDeviceBeams)
					wasStored |= storeBeam(r, mRec.t, throughput * Le, a_r);
				wasStored2 |= g_BVHBeams.insertBeam(k_Beam(r.origin, r.direction, mRec.t, throughput * Le));
				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				if (!g_BeamGrid.m_pDeviceBeams)
					wasStored |= storePhoton(mRec.p, throughput * Le, -r.direction, Vec3f(0, 0, 0), PhotonType::pt_Volume, g_Map, final_gather);
				hasVolPhotons = true;
				if (bssrdf)
					r.direction = Warp::squareToUniformSphere(rng.randomFloat2());
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
						wasStored |= storePhoton(dg.P, throughput * Le, wo, bRec.dg.sys.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse, g_Map, final_gather);

						/*AABB plane_box = calculatePlaneAABBInCell(g_Map.m_sSurfaceMap.m_sHash.getCell(g_Map.m_sSurfaceMap.m_sHash.Transform(dg.P)), dg.P, dg.sys.n, a_r);
						Vec2f xy(0.0f);

						unsigned int cell_idx = g_Map.m_sSurfaceMap.m_sHash.Hash(dg.P);
						if (cell_idx < g_Map.m_sSurfaceMap.m_uGridLength)
						{
							auto& entry = g_SurfaceEntries[cell_idx];
							auto f = throughput * Le * Frame::cosTheta(bRec.wi);
							float coeffs[] = {1 + xy.x * xy.y - xy.x - xy.y, xy.x - xy.x * xy.y, xy.x * xy.y, xy.x * xy.y};
							for (int corner = 0; corner < 4; corner++)
							{
								for (int s_i = 0; s_i < 3; s_i++)
									atomicAdd(&entry.m_sValues[corner][s_i], f[s_i] * coeffs[corner]);
							}
						}*/
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
		if (wasStored)
			atomicInc(&g_Map.m_uPhotonNumEmitted, 0xffffffff);
		if (wasStored2)
			atomicInc(&g_BVHBeams.m_uNumEmitted, 0xffffffff);
	}
	if (threadIdx.x == 0)
		atomicOr((int*)&g_HasVolumePhotons, __any(hasVolPhotons));

	g_RNGData(rng);
}

CUDA_DEVICE int nnSearch(float r, const Vec3f& p)
{
	Vec3u mi = g_Map.m_sVolumeMap.m_sHash.Transform(p - Vec3f(r));
	Vec3u ma = g_Map.m_sVolumeMap.m_sHash.Transform(p + Vec3f(r));
	int N = 0;
	for (unsigned int x = mi.x; x <= ma.x; x++)
		for (unsigned int y = mi.y; y <= ma.y; y++)
			for (unsigned int z = mi.z; z <= ma.z; z++)
			{
				unsigned int p_idx = g_Map.m_sVolumeMap.m_pDeviceHashGrid[g_Map.m_sVolumeMap.m_sHash.Hash(Vec3u(x,y,z))];
				while (p_idx != 0xffffffff && p_idx != 0xffffff)
				{
					Vec3f p2 = g_Map.m_pPhotons[p_idx].getPos(g_Map.m_sVolumeMap.m_sHash, Vec3u(x, y, z));
					if (distanceSquared(p, p2) < r * r)
						N++;
					p_idx = g_Map.m_pPhotons[p_idx].getNext();
				}
			}
	return N;
}

CUDA_DEVICE k_BeamMap g_Beams;
__global__ void buildBeams(float r)
{
	Vec3u cell_idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	int dimN = g_Map.m_sVolumeMap.m_sHash.m_fGridSize;
	if (cell_idx.x < dimN && cell_idx.y < dimN && cell_idx.z < dimN)
	{
		float r_new=r;
		Vec3f cellCenter = g_Map.m_sVolumeMap.m_sHash.InverseTransform(cell_idx) + g_Map.m_sVolumeMap.m_sHash.m_vCellSize / 2.0f;
		float N = nnSearch(r, cellCenter);
		r_new = math::sqrt(1 * r * r / max(N, 1.0f));

		unsigned int p_idx = g_Map.m_sVolumeMap.m_pDeviceHashGrid[g_Map.m_sVolumeMap.m_sHash.Hash(cell_idx)];
		while (p_idx != 0xffffffff && p_idx != 0xffffff)
		{
			Vec3f pPos = g_Map.m_pPhotons[p_idx].getPos(g_Map.m_sVolumeMap.m_sHash, cell_idx);
			g_Map.m_pPhotons[p_idx].accessNormalStorage() = half(r_new * r_new).bits();
			Vec3u mi = g_Map.m_sVolumeMap.m_sHash.Transform(pPos - Vec3f(r_new));
			Vec3u ma = g_Map.m_sVolumeMap.m_sHash.Transform(pPos + Vec3f(r_new));
			for (unsigned int x = mi.x; x <= ma.x; x++)
				for (unsigned int y = mi.y; y <= ma.y; y++)
					for (unsigned int z = mi.z; z <= ma.z; z++)
					{
						unsigned int d_idx = atomicInc(&g_Beams.m_uIndex, -1);
						if (d_idx >= g_Beams.m_uNumEntries)
							return;
						unsigned int n_idx = atomicExch(&g_Beams.m_pDeviceData[g_Map.m_sVolumeMap.m_sHash.Hash(Vec3u(x, y, z))].y, d_idx);
						g_Beams.m_pDeviceData[d_idx] = Vec2i(p_idx, n_idx);
					}
			p_idx = g_Map.m_pPhotons[p_idx].getNext();
		}
	}
}

__global__ void checkGrid()
{
	Vec3u cell_idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	int dimN = g_Map.m_sVolumeMap.m_sHash.m_fGridSize;
	if (cell_idx.x < dimN && cell_idx.y < dimN && cell_idx.z < dimN)
	{
		Vec3f min_cell = g_Map.m_sVolumeMap.m_sHash.InverseTransform(cell_idx),
			  max_cell = g_Map.m_sVolumeMap.m_sHash.InverseTransform(cell_idx + Vec3u(1));
		unsigned int p_idx = g_Map.m_sVolumeMap.m_pDeviceHashGrid[g_Map.m_sVolumeMap.m_sHash.Hash(cell_idx)];
		while (p_idx != 0xffffffff && p_idx != 0xffffff)
		{
			k_pPpmPhoton p = g_Map.m_pPhotons[p_idx];
			Vec3f p2 = p.getPos(g_Map.m_sVolumeMap.m_sHash, cell_idx);
			if ((p2.x < min_cell.x || p2.y < min_cell.y || p2.z < min_cell.z || p2.x > max_cell.x || p2.y > max_cell.y || p2.z > max_cell.z) && g_Map.m_sVolumeMap.m_sHash.getAABB().Contains(p2))
				printf("min_cell = {%f,%f,%f}, max_cell = {%f,%f,%f}, p = {%f,%f,%f}\n", min_cell.x, min_cell.y, min_cell.z, max_cell.x, max_cell.y, max_cell.z, p2.x, p2.y, p2.z);
			p_idx = p.getNext();
		}
	}
}

void k_sPpmTracer::doPhotonPass()
{
	bool hasVol = false;
	m_sBVHBeams.StartRendering();
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BVHBeams, &m_sBVHBeams, sizeof(m_sBVHBeams)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(m_sMaps)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_HasVolumePhotons, &hasVol, sizeof(bool)));
	m_sPhotonBeams.m_uBeamIdx = 0;
	m_sPhotonBeams.m_uGridIdx = m_sPhotonBeams.m_uGridOffset;
	if (m_sPhotonBeams.m_pGrid)
		ThrowCudaErrors(cudaMemset(m_sPhotonBeams.m_pGrid, -1, sizeof(Vec2i) * m_sPhotonBeams.m_uGridLength));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BeamGrid, &m_sPhotonBeams, sizeof(m_sPhotonBeams)));

	while (!m_sMaps.PassFinished())// && m_sPhotonBeams.m_uBeamIdx < m_sPhotonBeams.m_uBeamLength
	{
		if (m_bDirect)
			k_PhotonPass<true> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bFinalGather, getCurrentRadius2(3));
		else k_PhotonPass<false> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_bFinalGather, getCurrentRadius2(3));
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sMaps, g_Map, sizeof(m_sMaps)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sPhotonBeams, g_BeamGrid, sizeof(m_sPhotonBeams)));
	}
	
	if (m_sPhotonBeams.m_uGridIdx >= m_sPhotonBeams.m_uGridLength)
		std::cout << "Photn beam grid full!\n";

	ThrowCudaErrors(cudaMemcpyFromSymbol(&hasVol, g_HasVolumePhotons, sizeof(bool)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sBVHBeams, g_BVHBeams, sizeof(m_sBVHBeams)));
	m_sBVHBeams.BuildStorage(getCurrentRadius2(3)*10, m_pScene);

	if (hasVol && m_sBeams.m_pDeviceData)
	{
		int l = 6, l2 = m_sMaps.m_sVolumeMap.m_sHash.m_fGridSize / l + 1;
		m_sBeams.m_uIndex = m_sBeams.m_uGridEntries;
		ThrowCudaErrors(cudaMemcpyToSymbol(g_Beams, &m_sBeams, sizeof(m_sBeams)));
		ThrowCudaErrors(cudaMemset(m_sBeams.m_pDeviceData, -1, sizeof(Vec2i) * m_sBeams.m_uNumEntries));
		buildBeams << <dim3(l2, l2, l2), dim3(l, l, l) >> >(getCurrentRadius2(3));
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sBeams, g_Beams, sizeof(m_sBeams)));
		ThrowCudaErrors(cudaDeviceSynchronize());
		if (m_sBeams.m_uIndex >= m_sBeams.m_uNumEntries)
			std::cout << "Beam indices full!\n";
		//std::cout << "Beam indices index = " << m_sBeams.m_uIndex << ", length = " << m_sBeams.m_uNumEntries << "\n";
	}
}

CUDA_GLOBAL void estimateRadius(unsigned int w, unsigned int h, k_AdaptiveEntry* E, k_PhotonMapCollection<true, k_pPpmPhoton> photonMap, float maxR, float targetNumPhotons)
{
	k_PhotonMap<k_HashGrid_Reg>& map = photonMap.m_sSurfaceMap;
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = k_TraceRay(r);
		int N = 0, d = 0;
		while (r2.hasHit() && d++ < 10)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ERadiance, 0);
			if (r2.getMat().bsdf.hasComponent(EDelta))
			{
				r2.getMat().bsdf.sample(bRec, Vec2f(0));
				r = Ray(dg.P, bRec.getOutgoing());
				r2 = k_TraceRay(r);
			}
			else
			{
				Frame sys = Frame(bRec.dg.n);
				sys.t *= maxR;
				sys.s *= maxR;
				Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
				Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
				Vec3u lo = map.m_sHash.Transform(low), hi = map.m_sHash.Transform(high);
				for (unsigned int a = lo.x; a <= hi.x; a++)
					for (unsigned int b = lo.y; b <= hi.y; b++)
						for (unsigned int c = lo.z; c <= hi.z; c++)
						{
							unsigned int i0 = map.m_sHash.Hash(Vec3u(a, b, c)), i = map.m_pDeviceHashGrid[i0];
							while (i != 0xffffffff && i != 0xffffff)
							{
								k_pPpmPhoton e = photonMap.m_pPhotons[i];
								if (distanceSquared(e.getPos(map.m_sHash, Vec3u(a,b,c)), bRec.dg.P) < maxR * maxR)//&& dot(n, bRec.dg.sys.n) > 0.8f
								{
									N++;
								}
								i = e.getNext();
							}
						}
				break;
			}
		}
		k_AdaptiveEntry& e = E[y * w + x];
		if (N == 0)
			e.r = e.rd = maxR;
		else
		{
			//A_max = PI * maxR^2, density = N / A_max
			//target = density * A_correct = density * PI * r * r
			//r = sqrt(target / (density * PI))
			float d = float(N) / (PI * maxR * maxR);
			e.r = e.rd = math::sqrt(targetNumPhotons / (PI * d));
		}
	}
}

void k_sPpmTracer::estimatePerPixelRadius()
{
	int p0 = 16;
	estimateRadius << <dim3(w / p0 + 1, h / p0 + 1), dim3(p0, p0) >> >(w, h, m_pEntries, m_sMaps, 5 * m_fInitialRadius, 20);
}

CUDA_DEVICE unsigned int g_uMaxGridCounter;

CUDA_GLOBAL void visGrid(unsigned int w, unsigned int h, e_Image I, k_PhotonMapCollection<true, k_pPpmPhoton> photonMap, float scale)
{
	k_PhotonMap<k_HashGrid_Reg>& map = photonMap.m_sSurfaceMap;
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = k_TraceRay(r);
		unsigned int N = 0, d = 0;
		while (r2.hasHit() && d++ < 10)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ERadiance, 0);
			if (r2.getMat().bsdf.hasComponent(EDelta))
			{
				r2.getMat().bsdf.sample(bRec, Vec2f(0));
				r = Ray(dg.P, bRec.getOutgoing());
				r2 = k_TraceRay(r);
			}
			else
			{
				Vec3u idx = map.m_sHash.Transform(dg.P);
				unsigned int i0 = map.m_sHash.Hash(idx), i = map.m_pDeviceHashGrid[i0];
				while (i != 0xffffffff && i != 0xffffff)
				{
					k_pPpmPhoton e = photonMap.m_pPhotons[i];
					N++;
					i = e.getNext();
				}
				break;
			}
		}
		atomicMax(&g_uMaxGridCounter, N);
		I.AddSample(x, y, Spectrum(0, 0, N > scale / 2 ? 1 : 0));//N / scale
	}
}

void k_sPpmTracer::visualizeGrid(e_Image* I)
{
	I->Clear();
	ZeroSymbol(g_uMaxGridCounter);
	int p0 = 16;
	visGrid << <dim3(w / p0 + 1, h / p0 + 1), dim3(p0, p0) >> >(w, h, *I, m_sMaps, m_uVisLastMax);
	cudaMemcpyFromSymbol(&m_uVisLastMax, g_uMaxGridCounter, sizeof(unsigned int));
}