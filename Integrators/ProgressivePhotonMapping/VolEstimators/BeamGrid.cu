#include "BeamGrid.h"
#include <set>

namespace CudaTracerLib {

CUDA_CONST CudaStaticWrapper<decltype(BeamGrid::m_sStorage)> g_PhotonStorage;
CUDA_DEVICE CudaStaticWrapper<decltype(BeamGrid::m_sBeamGridStorage)> g_BeamGridStorage;

__global__ void buildBeams(float r_search, Vec3u dimN, float kNNTarget, unsigned int numIter)
{
	Vec3u cell_idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	if (cell_idx.x < dimN.x && cell_idx.y < dimN.y && cell_idx.z < dimN.z)
	{
#ifdef ISCUDA
		Vec3f cellCenter = g_PhotonStorage->getHashGrid().getCell(cell_idx).Center();
		float density = 0;
		g_PhotonStorage->ForAll(cellCenter - Vec3f(r_search), cellCenter + Vec3f(r_search), [&](const Vec3u& cell_idx, unsigned int pIdx, const _VolumetricPhoton& ph)
		{
			auto d2 = distanceSquared(cellCenter, ph.getPos(g_PhotonStorage->getHashGrid(), cell_idx));
			if (d2 < r_search * r_search)
				density++;
		});
		float r_new = math::sqrt(kNNTarget * r_search * r_search / max((float)density, 1.0f));

		g_PhotonStorage->ForAllCellEntries(cell_idx, [&](unsigned int p_idx, _VolumetricPhoton& ph)
		{
			ph.setRad1(r_new);
			Vec3f ph_pos = ph.getPos(g_PhotonStorage->getHashGrid(), cell_idx);
			g_BeamGridStorage->ForAllCells(ph_pos - Vec3f(r_new), ph_pos + Vec3f(r_new), [&](const Vec3u& cell_idx_store)
			{
				g_BeamGridStorage->Store(cell_idx_store, BeamGrid::entry(p_idx));
			});
		});
#endif
	}
}

void BeamGrid::PrepareForRendering()
{
	PointStorage::PrepareForRendering();
	int l = 6;
	auto l2 = m_sStorage.getHashGrid().m_gridDim / l + Vec3u(1);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_PhotonStorage, &m_sStorage, sizeof(m_sStorage)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BeamGridStorage, &m_sBeamGridStorage, sizeof(m_sBeamGridStorage)));
	buildBeams << <dim3(l2.x, l2.y, l2.z), dim3(l, l, l) >> >(getRadVol<2>(), m_sStorage.getHashGrid().m_gridDim, m_kNNClb(), m_numPass);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sStorage, g_PhotonStorage, sizeof(m_sStorage)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sBeamGridStorage, g_BeamGridStorage, sizeof(m_sBeamGridStorage)));

	m_sBeamGridStorage.PrepareForUse();
	ThrowCudaErrors(cudaDeviceSynchronize());

	if (m_sBeamGridStorage.getNumStoredEntries() >= m_sBeamGridStorage.getNumEntries())
		std::cout << "Warning : Stored " << m_sBeamGridStorage.getNumStoredEntries() << " indices but only capacity for " << m_sBeamGridStorage.getNumEntries() << "!\n";

	/*typedef decltype(m_sBeamGridStorage)::linkedEntry ent;
	unsigned int N = m_sBeamGridStorage.getNumEntries(), N2 = m_sBeamGridStorage.getHashGrid().m_nElements;
	static ent* hostData = 0;
	static unsigned int* hostGrid = 0;
	if (hostData == 0)
	{
		hostData = new ent[N];
		hostGrid = new unsigned int[N2];
	}
	CUDA_MEMCPY_TO_HOST(hostData, m_sBeamGridStorage.getDeviceData(), N * sizeof(ent));
	CUDA_MEMCPY_TO_HOST(hostGrid, m_sBeamGridStorage.getDeviceGrid(), N2 * sizeof(unsigned int));
	std::set<unsigned int> data;
	for (int cell_idx = 0; cell_idx < N2; cell_idx++)
	{
		data.clear();
		unsigned int i = hostGrid[cell_idx];
		while (i != UINT_MAX)
		{
			if (data.find(i) != data.end())
				throw 1;
			data.insert(i);
			i = hostData[i].nextIdx;
		}
	}*/
}

void BeamGrid::StartNewPass(DynamicScene* scene)
{
	PointStorage::StartNewPass(scene);
	m_sBeamGridStorage.ResetBuffer();
}

}