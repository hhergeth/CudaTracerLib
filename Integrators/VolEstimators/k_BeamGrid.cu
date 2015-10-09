#include "k_BeamGrid.h"

struct GridData
{
	unsigned int m_uIndex;
	unsigned int m_uNumEntries, m_uGridEntries;
	Vec2i* m_pDeviceData;

	int photonDensNum;

	e_SpatialLinkedMap<k_PointStorage::volPhoton> m_sStorage;

	GridData(){}

	GridData(k_BeamGrid* a)
		: m_uIndex(a->m_uGridEntries), m_uNumEntries(a->m_uNumEntries), m_uGridEntries(a->m_uGridEntries), m_pDeviceData(a->m_pDeviceData), 
		  photonDensNum(a->photonDensNum), m_sStorage(a->m_sStorage)
	{

	}
};
CUDA_DEVICE GridData g_BeamData;

__global__ void buildBeams(float r, int dimN)
{
	Vec3u cell_idx(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	if (cell_idx.x < dimN && cell_idx.y < dimN && cell_idx.z < dimN)
	{
		float r_new = r;
		Vec3f cellCenter = g_BeamData.m_sStorage.hashMap.InverseTransform(cell_idx) + g_BeamData.m_sStorage.hashMap.m_vCellSize / 2.0f;
#ifdef ISCUDA
		int N = 0;
		g_BeamData.m_sStorage.ForAll(cellCenter - Vec3f(r), cellCenter + Vec3f(r), [&](unsigned int pIdx, const k_PointStorage::volPhoton& ph)
		{
			if (distanceSquared(cellCenter, ph.p) < r * r)
				N++;
		});

		r_new = math::sqrt(g_BeamData.photonDensNum * r * r / max((float)N, 1.0f));
		
		g_BeamData.m_sStorage.ForAll(cell_idx, [&](unsigned int p_idx, k_PointStorage::volPhoton& ph)
		{
			ph.rad = r_new * r_new;

			Vec3u mi = g_BeamData.m_sStorage.hashMap.Transform(ph.p - Vec3f(r_new));
			Vec3u ma = g_BeamData.m_sStorage.hashMap.Transform(ph.p + Vec3f(r_new));

			for (unsigned int x = mi.x; x <= ma.x; x++)
				for (unsigned int y = mi.y; y <= ma.y; y++)
					for (unsigned int z = mi.z; z <= ma.z; z++)
					{

				unsigned int d_idx = atomicInc(&g_BeamData.m_uIndex, (unsigned int)-1);
				if (d_idx >= g_BeamData.m_uNumEntries)
					return;
				unsigned int n_idx = atomicExch(&g_BeamData.m_pDeviceData[g_BeamData.m_sStorage.hashMap.Hash(Vec3u(x, y, z))].y, d_idx);
				g_BeamData.m_pDeviceData[d_idx] = Vec2i(p_idx, n_idx);

					}
		});
#endif
	}
}

void k_BeamGrid::PrepareForRendering()
{
	int l = 6, l2 = m_sStorage.hashMap.m_fGridSize / l + 1;
	GridData dat(this);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BeamData, &dat, sizeof(dat)));
	ThrowCudaErrors(cudaMemset(m_pDeviceData, -1, sizeof(Vec2i) * m_uNumEntries));
	buildBeams << <dim3(l2, l2, l2), dim3(l, l, l) >> >(m_fCurrentRadiusVol, m_sStorage.hashMap.m_fGridSize);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&dat, g_BeamData, sizeof(dat)));
	this->m_sStorage = dat.m_sStorage;
	this->m_uIndex = dat.m_uIndex;
	if (m_uIndex >= m_uNumEntries)
		std::cout << "Beam indices full!\n";
}