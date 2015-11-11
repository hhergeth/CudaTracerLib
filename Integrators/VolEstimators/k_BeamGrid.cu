#include "k_BeamGrid.h"

namespace CudaTracerLib {

CUDA_CONST e_SpatialLinkedMap<k_PointStorage::volPhoton> g_PhotonStorage;
CUDA_DEVICE e_SpatialLinkedMap<int> g_BeamGridStorage;

__global__ void buildBeams(float r, int dimN, float nnSearch)
{
	Vec3u cell_idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	if (cell_idx.x < dimN && cell_idx.y < dimN && cell_idx.z < dimN)
	{
		Vec3f cellCenter = g_PhotonStorage.hashMap.InverseTransform(cell_idx) + g_PhotonStorage.hashMap.m_vCellSize / 2.0f;
#ifdef ISCUDA
		int N = 0;
		g_PhotonStorage.ForAll(cellCenter - Vec3f(r), cellCenter + Vec3f(r), [&](unsigned int pIdx, const k_PointStorage::volPhoton& ph)
		{
			if (distanceSquared(cellCenter, ph.p) < r * r)
				N++;
		});
		float r_new = math::sqrt(nnSearch * r * r / max((float)N, 1.0f));

		g_PhotonStorage.ForAll(cell_idx, [&](unsigned int p_idx, k_PointStorage::volPhoton& ph)
		{
			ph.rad = r_new * r_new;
			g_BeamGridStorage.ForAllCells(ph.p - Vec3f(r_new), ph.p + Vec3f(r_new), [&](const Vec3u& cell_idx_store)
			{
				g_BeamGridStorage.store(cell_idx_store, p_idx);
			});
		});
#endif
	}
}

void k_BeamGrid::PrepareForRendering()
{
	int l = 6, l2 = m_sStorage.hashMap.m_fGridSize / l + 1;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_PhotonStorage, &m_sStorage, sizeof(m_sStorage)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BeamGridStorage, &m_sBeamGridStorage, sizeof(m_sBeamGridStorage)));
	buildBeams << <dim3(l2, l2, l2), dim3(l, l, l) >> >(m_fCurrentRadiusVol, m_sStorage.hashMap.m_fGridSize, photonDensNum);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sStorage, g_PhotonStorage, sizeof(m_sStorage)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sBeamGridStorage, g_BeamGridStorage, sizeof(m_sBeamGridStorage)));
}

}