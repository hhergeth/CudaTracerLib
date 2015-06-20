#include <StdAfx.h>
#include "../Engine/e_Buffer.h"
#include "k_sPpmTracer.h"
#include "..\Base\Timer.h"
#include "../Engine/e_DynamicScene.h"
#include "../Engine/e_Node.h"
#include "../Engine/e_Mesh.h"

#define LNG 200
#define SER_NAME "photonMapBuf.dat"

k_sPpmTracer::k_sPpmTracer()
	: m_pEntries(0), m_bFinalGather(false), m_bVisualizeGrid(false)
{
#ifdef NDEBUG
	m_uBlocksPerLaunch = 180;
#else
	m_uBlocksPerLaunch = 1;
#endif
	m_uPhotonsEmitted = -1;
	m_bLongRunning = false;
	unsigned int numPhotons = (m_uBlocksPerLaunch + 2) * PPM_slots_per_block;
	unsigned int linkedListLength = numPhotons * 10;
	m_sMaps = k_PhotonMapCollection<true, k_pPpmPhoton>(numPhotons, LNG*LNG*LNG, linkedListLength);
}

void k_sPpmTracer::PrintStatus(std::vector<std::string>& a_Buf) const
{
	double pC = floor((double)m_uPhotonsEmitted / 1000000.0);
	a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	double pCs = m_uPhotonsEmitted / m_fAccRuntime / 1000000.0f;
	a_Buf.push_back(format("Photons/Sec : %f", (float)pCs));
	a_Buf.push_back(format("Light Visibility : %f", m_fLightVisibility));
	a_Buf.push_back(format("Photons per pass : %d*100,000", m_sMaps.m_uPhotonBufferLength / 100000));
	if (m_bVisualizeGrid)
		a_Buf.push_back(format("Max #Photons per cell : %d", m_uVisLastMax));
}

void k_sPpmTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0.1f, 10.0f, true, &m_fInitialRadiusScale, "Initial radius = %g units");
	//a_Callback(0.1f, 100.0f, true, &m_uNewPhotonsPerRun, "Number of photons per pass = %g [M]");
}

void k_sPpmTracer::Resize(unsigned int _w, unsigned int _h)
{
	k_Tracer<true, true>::Resize(_w, _h);
	if(m_pEntries)
		CUDA_FREE(m_pEntries);
	CUDA_MALLOC(&m_pEntries, sizeof(k_AdaptiveEntry) * _w * _h);
}
/*
void print(k_PhotonMapCollection& m_sMaps, k_PhotonMap<k_HashGrid_Reg>& m_Map, std::string name)
{
	k_pPpmPhoton* photons = new k_pPpmPhoton[m_sMaps.m_uPhotonBufferLength];
	unsigned int* grid = new unsigned int[m_Map.m_uGridLength];
	cudaMemcpy(photons, m_sMaps.m_pPhotons, sizeof(k_pPpmPhoton) * m_sMaps.m_uPhotonBufferLength, cudaMemcpyDeviceToHost);
	cudaMemcpy(grid, m_Map.m_pDeviceHashGrid, sizeof(unsigned int) * m_Map.m_uGridLength, cudaMemcpyDeviceToHost);
			
	unsigned int usedCells = 0, maxCount = 0;
	unsigned char* counts = new unsigned char[m_Map.m_uGridLength];
	for(unsigned int i = 0; i < m_Map.m_uGridLength; i++)
	{
		if(grid[i] != -1)
		{
			usedCells++;
			k_pPpmPhoton* p = photons + grid[i];
			unsigned int c = 1;
			for( ; p->next != -1; c++)
				p = photons + p->next;
			counts[i] = c;
			maxCount = max(maxCount, c);
		}
		else
		{
			counts[i] = 0;
		}
	}
	OutputStream os(name.c_str());
	float avgNum = float(m_sMaps.m_uPhotonBufferLength) / float(usedCells), f1 = float(usedCells) / float(m_Map.m_uGridLength);
	os << m_Map.m_sHash.m_fGridSize;
	os << (int)avgNum * 2;
	os.Write(counts, sizeof(unsigned char) * m_Map.m_uGridLength);
	os.Close();
	float var = 0;
	int avg = (int)avgNum;
	for(unsigned int i = 0; i < m_Map.m_uGridLength; i++)
		if(counts[i])
			var += (counts[i] - avg) * (counts[i] - avg) / float(usedCells);
	std::string s = format("max : %d, avg : %f, used cells : %f, var : %f\n", maxCount, avgNum, f1, math::sqrt(var));
	std::cout << s;
	OutputDebugString(s.c_str());
}*/

void k_sPpmTracer::DoRender(e_Image* I)
{
	doPhotonPass();
	if (m_uPhotonsEmitted == 0)
	{
		//estimatePerPixelRadius();
	}
	m_uPhotonsEmitted += m_sMaps.m_uPhotonNumEmitted;
	if (m_bVisualizeGrid)
		visualizeGrid(I);
	else k_Tracer<true, true>::DoRender(I);
	m_sMaps.StartNewPass();
}

void k_sPpmTracer::StartNewTrace(e_Image* I)
{
	m_bDirect = !m_pScene->getVolumes().hasElements();
#ifndef _DEBUG
	m_fLightVisibility = k_Tracer::GetLightVisibility(m_pScene, 1);
#endif
	if (m_bDirect)
		m_bDirect = m_fLightVisibility > 0.5f;
	//m_bDirect = 0;
	k_Tracer<true, true>::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
#ifndef _DEBUG
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
#else
	AABB m_sEyeBox = m_pScene->getSceneBox();
#endif
	m_sEyeBox.Enlarge(0.1f);
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	AABB volBox = m_pScene->getKernelSceneData().m_sVolume.box;
	for (auto it : m_pScene->getNodes())
	{
		e_StreamReference<e_KernelMaterial> mats = m_pScene->getMats(it);
		for (unsigned int j = 0; j < mats.getLength(); j++)
		{
			const e_KernelBSSRDF* bssrdf;
			DifferentialGeometry dg;
			ZERO_MEM(dg);
			if (mats(j)->GetBSSRDF(dg, &bssrdf))
			{
				volBox.Enlarge(m_pScene->getNodeBox(it));
				m_bLongRunning |= 1;
			}
		}
	}
	//volBox = m_pScene->getKernelSceneData().m_sBox;
	m_sMaps.StartNewRendering(m_sEyeBox, volBox, r);
	m_sMaps.StartNewPass();

	float r_scene = length(m_pScene->getKernelSceneData().m_sBox.Size()) / 2;
	r_min = 10e-7f * r_scene;
	r_max = 10e-3f * r_scene;
	float r1 = r_max / 10.0f;
	doStartPass(r1, r1);
}