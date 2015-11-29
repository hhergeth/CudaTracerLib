#include <StdAfx.h>
#include "BeamBeamGrid.h"
#include <set>
#include <Base/Timer.h>
#include <Math/Compression.h>
#include <Math/half.h>
#include <bitset>
#include <set>

namespace CudaTracerLib {

void BeamBeamGrid::StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
{
	m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
	m_uNumEmitted = 0;
	m_uBeamIdx = 0;
	m_sStorage.ResetBuffer();

	float r = radProvider->getCurrentRadius(1);
	Vec3f dim = m_sStorage.hashMap.m_vCellSize;
	float d = dim.min();
	if (r > d)
	{
		std::cout << "WARNING beam beam traversal can be bugged!";
		std::cout << "r = " << r << ", d = " << d << "\n";
	}
}

void BeamBeamGrid::PrepareForRendering()
{
	/*std::cout << "m_uNumEmitted = " << m_uNumEmitted << "\n";
	SpatialLinkedMap<int>::linkedEntry* entries = new SpatialLinkedMap<int>::linkedEntry[m_sStorage.numData];
	ThrowCudaErrors(cudaMemcpy(entries, m_sStorage.deviceData, m_sStorage.numData * sizeof(SpatialLinkedMap<int>::linkedEntry), cudaMemcpyDeviceToHost));
	unsigned int I = m_sStorage.gridSize * m_sStorage.gridSize * m_sStorage.gridSize;
	unsigned int* grid = new unsigned int[I];
	ThrowCudaErrors(cudaMemcpy(grid, m_sStorage.deviceMap, I * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int* denseGrid = new unsigned int[m_sStorage.numData + I];
	std::cout << "denseGrid size = " << (((m_sStorage.numData + I) * sizeof(unsigned int)) / (1024 * 1024)) << "[MB]\n";
	Platform::SetMemory(denseGrid, sizeof(unsigned int) * (m_sStorage.numData + I), UINT_MAX);

	unsigned int d_idx = I;
	InstructionTimer T;
	T.StartTimer();
	m_sStorage.ForAllCells([&](const Vec3u& cell_idx)
	{
		unsigned int i = m_sStorage.hashMap.Hash(cell_idx), j = grid[i], start = d_idx, n = 0;
		m_sStorage.ForAll(cell_idx, [&](unsigned int abc, int p_idx)
		{
			denseGrid[d_idx++] = p_idx;
			n++;
		});
		denseGrid[i] = start != d_idx ? ((n << 24) | start) : UINT_MAX;
	});
	std::cout << "dense grid creation took : " << T.EndTimer() << "[Sec]\n";*/

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
	
	if (!m_sStorage.deviceDataIdx) return;

	std::vector<SpatialLinkedMap<int>::linkedEntry> hostIndices;
	std::vector<unsigned int> hostMap;
	hostIndices.resize(m_sStorage.deviceDataIdx);
	CUDA_MEMCPY_TO_HOST(&hostIndices[0], m_sStorage.deviceData, hostIndices.size() * sizeof(SpatialLinkedMap<int>::linkedEntry));
	hostMap.resize(m_sStorage.gridSize * m_sStorage.gridSize * m_sStorage.gridSize);
	CUDA_MEMCPY_TO_HOST(&hostMap[0], m_sStorage.deviceMap, hostMap.size() * sizeof(unsigned int));
	auto bitset = new std::bitset<1024 * 1024 * 10>();
	m_sStorage.ForAllCells([&](const Vec3u& pos)
	{
		bitset->reset();
		unsigned int idx = hostMap[m_sStorage.hashMap.Hash(pos)], lastIdx = UINT_MAX;
		while (idx != UINT_MAX)
		{
			int beam_idx = hostIndices[idx].value;
			if (bitset->at(beam_idx))
			{
				if (lastIdx == UINT_MAX)
					throw 1;
				hostIndices[lastIdx].nextIdx = hostIndices[idx].nextIdx;
			}
			else
			{
				bitset->set(beam_idx, true);
				lastIdx = idx;
			}
			idx = hostIndices[idx].nextIdx;
		}
	});
	/*auto A = m_sStorage.deviceData; m_sStorage.deviceData = &hostIndices[0];
	auto B = m_sStorage.deviceMap; m_sStorage.deviceMap = &hostMap[0];
	std::set<unsigned int> set;
	m_sStorage.ForAllCells(Vec3u(0), Vec3u(m_sStorage.gridSize - 1), [&](const Vec3u& pos)
	{
		set.clear();
		m_sStorage.ForAll(pos, [&](unsigned int, int val)
		{
			if (set.count(val) != 0)
				throw 2;
			set.insert(val);
		});
	});
	m_sStorage.deviceData = A;
	m_sStorage.deviceMap = B;*/
	CUDA_MEMCPY_TO_DEVICE(m_sStorage.deviceData, &hostIndices[0], hostIndices.size() * sizeof(SpatialLinkedMap<int>::linkedEntry));
}

}