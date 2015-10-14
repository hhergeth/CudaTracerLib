#include <StdAfx.h>
#include "k_BeamBeamGrid.h"
#include <set>

void k_BeamBeamGrid::StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene)
{
	m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
	m_uNumEmitted = 0;
	m_uBeamIdx = 0;
	m_sStorage.ResetBuffer();

	float r = radProvider->getCurrentRadius(1);
	Vec3f dim = m_sStorage.hashMap.m_vCellSize;
	float d = dim.min();
	std::cout << "r = " << r << ", d = " << d << "\n";
}

void k_BeamBeamGrid::PrepareForRendering()
{
	/*e_SpatialLinkedMap<int>::linkedEntry* entries = new e_SpatialLinkedMap<int>::linkedEntry[m_sStorage.numData];
	cudaMemcpy(entries, m_sStorage.deviceData, m_sStorage.numData * sizeof(e_SpatialLinkedMap<int>::linkedEntry), cudaMemcpyDeviceToHost);
	unsigned int I = m_sStorage.gridSize * m_sStorage.gridSize * m_sStorage.gridSize;
	unsigned int* grid = new unsigned int[I];
	cudaMemcpy(grid, m_sStorage.deviceMap, I * sizeof(unsigned int), cudaMemcpyDeviceToHost);*/
	/*m_sStorage.deviceData = entries;
	m_sStorage.deviceMap = grid;
	Ray r = g_SceneData.GenerateSensorRay(200, 200);
	int n = 0;
	TraverseGrid(r, m_sStorage.hashMap, 0, FLT_MAX, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
		{
			n += beam_idx == 0;
			//if(pixel.x == 200 && pixel.y == 200)
			//	printf(", %d", ABC);
		});
	});
	std::cout << n;*/
	/*std::set<unsigned int> buf;
	std::vector<unsigned int> cells;
	for (unsigned int i = 0; i < I; i++)
	{
		buf.clear();
		unsigned int n = grid[i];
		while (n != UINT_MAX)
		{
			if (entries[n].value == 1)
				cells.push_back(i);
			if (buf.find(entries[n].value) != buf.end())
				std::cout << "dup = " << i << "\n";
			buf.insert(entries[n].value);
			n = entries[n].nextIdx;
		}
	}*/
}