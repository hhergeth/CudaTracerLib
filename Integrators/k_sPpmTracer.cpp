#include <StdAfx.h>
#include "k_sPpmTracer.h"
#include "..\Base\StringUtils.h"

#define LNG 200
#define SER_NAME "photonMapBuf.dat"

k_sPpmTracer::k_sPpmTracer()
	: k_ProgressiveTracer(), m_uGridLength(LNG*LNG*LNG), m_pEntries(0)
{
	m_uPhotonsEmitted = -1;
	m_uPreviosCount = 0;
	m_bLongRunning = false;
#ifdef DEBUG
	m_uNewPhotonsPerRun = 0.01f;
#else
	m_uNewPhotonsPerRun = 2;
#endif
	m_uModus = 1;
	m_fInitialRadiusScale = 1;
	m_sMaps = k_PhotonMapCollection((int)(1000000.0f * m_uNewPhotonsPerRun), m_uGridLength);
}

void k_sPpmTracer::PrintStatus(std::vector<std::string>& a_Buf)
{
	double pC = floor((double)m_uPhotonsEmitted / 1000000.0);
	a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	double pCs = getValuePerSecond(m_uPhotonsEmitted, 1000000.0);
	a_Buf.push_back(format("Photons/Sec : %f", (float)pCs));
}

void k_PhotonMapCollection::Resize(unsigned int a_BufferLength)
{
	//if(a_BufferLength > m_uRealBufferSize)
	{
		m_uRealBufferSize = a_BufferLength;
		void* old = m_pPhotons;
		CUDA_MALLOC(&m_pPhotons, sizeof(k_pPpmPhoton) * a_BufferLength);
		if(old)
		{
			cudaMemcpy(m_pPhotons, old, sizeof(k_pPpmPhoton) * m_uPhotonBufferLength, cudaMemcpyDeviceToDevice);
			CUDA_FREE(old);
		}
	}
	//else cudaMemset(m_pPhotons, 0, sizeof(k_pPpmPhoton) * a_BufferLength);
	m_uPhotonBufferLength = a_BufferLength;
	m_sVolumeMap.Resize(a_BufferLength, m_pPhotons);
	m_sSurfaceMap.Resize(a_BufferLength, m_pPhotons);
}

void k_PhotonMapCollection::StartNewPass()
{
	m_uPhotonNumEmitted = m_uPhotonNumStored = 0;
	m_sVolumeMap.StartNewPass();
	m_sSurfaceMap.StartNewPass();
}

bool k_PhotonMapCollection::PassFinished()
{
	return m_uPhotonNumStored >= m_uPhotonBufferLength;
}

void k_PhotonMapCollection::Free()
{
	m_sVolumeMap.Free();
	m_sSurfaceMap.Free();
	CUDA_FREE(m_pPhotons);
}

k_PhotonMapCollection::k_PhotonMapCollection(unsigned int a_BufferLength, unsigned int a_HashNum)
	: m_pPhotons(0), m_sVolumeMap(a_BufferLength, a_HashNum, m_pPhotons), m_sSurfaceMap(a_BufferLength, a_HashNum, m_pPhotons)
{
	m_uRealBufferSize = m_uPhotonBufferLength = 0;
	m_uPhotonNumStored = m_uPhotonNumEmitted = 0;
	Resize(a_BufferLength);
}

void k_sPpmTracer::CreateSliders(SliderCreateCallback a_Callback)
{
	a_Callback(0.1f, 10.0f, true, &m_fInitialRadiusScale, "Initial radius = %g units");
	a_Callback(0.1f, 100.0f, true, &m_uNewPhotonsPerRun, "Number of photons per pass = %g [M]");
}

void k_sPpmTracer::Resize(unsigned int _w, unsigned int _h)
{
	k_TracerBase::Resize(_w, _h);
	if(m_pEntries)
		CUDA_FREE(m_pEntries);
	CUDA_MALLOC(&m_pEntries, sizeof(k_AdaptiveEntry) * _w * _h);
}

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
			maxCount = MAX(maxCount, c);
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
	std::string s = format("max : %d, avg : %f, used cells : %f, var : %f\n", maxCount, avgNum, f1, sqrtf(var));
	OutputDebugString(s.c_str());
}

void k_sPpmTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	if(m_uModus == 1)
	{
		updateBuffer();
		//if(!m_uPhotonsEmitted)
		doPhotonPass();
		cudaThreadSynchronize();
		if(m_sMaps.PassFinished())
		{
			m_uPassesDone++;
			
			//print(m_sMaps, m_sMaps.m_sSurfaceMap, "s_map.bin"); print(m_sMaps, m_sMaps.m_sVolumeMap, "v_map.bin");

			m_uPhotonsEmitted += m_sMaps.m_uPhotonNumEmitted;
			doEyePass(I);
			I->DoUpdateDisplay(m_uPassesDone);
			m_sMaps.StartNewPass();
			m_uPreviosCount = m_uPhotonsEmitted;
		}
	}
	else if(m_uModus == 2)
	{
		m_uPassesDone = 1;
		I->Clear();
		doEyePass(I);
		I->DoUpdateDisplay(m_uPassesDone);
	}
}

void k_sPpmTracer::initNewPass(e_Image* I)
{
	k_ProgressiveTracer::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	//AABB m_sEyeBox = m_pCamera->m_sLastFrustum;
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, m_pCamera, true);
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / float(w) * m_fInitialRadiusScale;
	m_sEyeBox.minV -= make_float3(r);
	m_sEyeBox.maxV += make_float3(r);
	m_fInitialRadius = r;
	AABB volBox = m_pScene->getKernelSceneData().m_sVolume.box;
	for(unsigned int i = 0; i < m_pScene->getNodeCount(); i++)
	{
		e_StreamReference(e_Node) n = m_pScene->getNodes()(i);
		e_BufferReference<e_Mesh, e_KernelMesh> m = m_pScene->getMesh(n);
		unsigned int s = n->m_uMaterialOffset, l = m_pScene->getMesh(n)->m_sMatInfo.getLength();
		for(unsigned int j = 0; j < l; j++)
		{
			e_StreamReference(e_KernelMaterial) mat = m_pScene->m_pMaterialBuffer->operator()(s + j, 1);
			const e_KernelBSSRDF* bssrdf;
			MapParameters dg;
			ZERO_MEM(dg);
			if(mat->GetBSSRDF(dg, &bssrdf))
			{
				volBox.Enlarge(m_pScene->getBox(n));
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
	float r1 = r_max/10.0f;
	doStartPass(r1, r1);
}

static bool GGG = false;
static float GGGf0;
static float GGGf1;
void k_sPpmTracer::StartNewTrace(e_Image* I)
{
	m_bDirect = !m_pScene->getVolumes().getLength();//m_bDirect=1;
	if(m_uModus == 1)
	{
		initNewPass(I);
	}
	else if(!GGG || GGGf0 != m_fInitialRadiusScale || GGGf1 != m_uNewPhotonsPerRun)
	{
		initNewPass(I);
		GGGf1 = m_uNewPhotonsPerRun;
		GGGf0 = m_fInitialRadiusScale;
		GGG = true;
		updateBuffer();
		while(!m_sMaps.PassFinished())
			doPhotonPass();
		m_uPassesDone = 1;
	}
}