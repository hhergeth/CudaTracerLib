#include <StdAfx.h>
#include "k_sPpmTracer.h"

#define LNG 200

k_sPpmTracer::k_sPpmTracer()
	: k_ProgressiveTracer(), m_uGridLength(LNG*LNG*LNG), m_pEntries(0)
{
	m_bLongRunning = false;
#ifdef DEBUG
	m_uNewPhotonsPerRun = 0.1f;
#else
	m_uNewPhotonsPerRun = 5;
#endif
	m_uModus = 1;
	m_fInitialRadiusScale = 1;
	m_sMaps = k_PhotonMapCollection((int)(1000000.0f * m_uNewPhotonsPerRun), m_uGridLength);
}

void k_sPpmTracer::PrintStatus(std::vector<FW::String>& a_Buf)
{
	double pC = floor((double)m_uPhotonsEmitted / 1000000.0);
	a_Buf.push_back(FW::sprintf("Photons emitted : %d[Mil]", (int)pC));
	double pCs = getValuePerSecond(m_uPhotonsEmitted, 1000000.0);
	a_Buf.push_back(FW::sprintf("Photons/Sec : %f", (float)pCs));
}

void k_PhotonMapCollection::Resize(unsigned int a_BufferLength)
{
	//if(a_BufferLength > m_uRealBufferSize)
	{
		m_uRealBufferSize = a_BufferLength;
		void* old = m_pPhotons;
		cudaMalloc(&m_pPhotons, sizeof(k_pPpmPhoton) * a_BufferLength);
		if(old)
		{
			cudaMemcpy(m_pPhotons, old, sizeof(k_pPpmPhoton) * m_uPhotonBufferLength, cudaMemcpyDeviceToDevice);
			cudaFree(old);
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
	cudaFree(m_pPhotons);
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
		cudaFree(m_pEntries);
	cudaMalloc(&m_pEntries, sizeof(k_AdaptiveEntry) * _w * _h);
}

void print(k_PhotonMapCollection& m_sMaps)
{
		k_pPpmPhoton* photons = new k_pPpmPhoton[m_sMaps.m_uPhotonBufferLength];
		unsigned int* grid = new unsigned int[m_sMaps.m_sVolumeMap.m_uGridLength];
		cudaMemcpy(photons, m_sMaps.m_pPhotons, sizeof(k_pPpmPhoton) * m_sMaps.m_uPhotonBufferLength, cudaMemcpyDeviceToHost);
		cudaMemcpy(grid, m_sMaps.m_sVolumeMap.m_pDeviceHashGrid, sizeof(unsigned int) * m_sMaps.m_sVolumeMap.m_uGridLength, cudaMemcpyDeviceToHost);
			
		unsigned int usedCells = 0, maxCount = 0;
		unsigned char* counts = new unsigned char[m_sMaps.m_sVolumeMap.m_uGridLength];
		for(int i = 0; i < m_sMaps.m_sVolumeMap.m_uGridLength; i++)
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
		OutputStream os("grid1.bin");
		os << maxCount;
		os.Write(counts, sizeof(unsigned char) * m_sMaps.m_sVolumeMap.m_uGridLength);
		os.Close();
		float avgNum = float(m_sMaps.m_uPhotonBufferLength) / float(usedCells), f1 = float(usedCells) / float(m_sMaps.m_sVolumeMap.m_uGridLength);
		float var = 0;
		int avg = (int)avgNum;
		for(int i = 0; i < m_sMaps.m_sVolumeMap.m_uGridLength; i++)
			if(counts[i])
				var += (counts[i] - avg) * (counts[i] - avg) / float(usedCells);
		FW::String s = FW::sprintf("max : %d, avg : %f, used cells : %f, var : %f", maxCount, avgNum, f1, sqrtf(var));
		OutputDebugString(s.getPtr());
}

void k_sPpmTracer::DoRender(e_Image* I)
{
	//k_ProgressiveTracer::DoRender(I);
	if(m_uModus == 1)
	{
		updateBuffer();
		//if(!m_uPhotonsEmitted)
		doPhotonPass();
		cudaThreadSynchronize();
		if(m_sMaps.PassFinished())
		{
			m_uPassesDone++;
			
			//print(m_sMaps);

			m_uPhotonsEmitted += m_sMaps.m_uPhotonNumEmitted;
			doEyePass(I);
			I->UpdateDisplay();
			m_sMaps.StartNewPass();
		}
	}
	else if(m_uModus == 2)
	{
		m_uPassesDone = 1;
		I->StartNewRendering();
		doEyePass(I);
		I->UpdateDisplay();
	}
}

void k_sPpmTracer::initNewPass(e_Image* I)
{
	k_ProgressiveTracer::StartNewTrace(I);
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = m_pCamera->m_sLastFrustum;
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / float(w) * m_fInitialRadiusScale;
	m_sEyeBox.minV -= make_float3(r);
	m_sEyeBox.maxV += make_float3(r);
	m_fInitialRadius = r;
	AABB volBox = m_pScene->getKernelSceneData().m_sVolume.box;
	for(int i = 0; i < m_pScene->getNodeCount(); i++)
	{
		e_StreamReference(e_Node) n = m_pScene->getNodes()(i);
		e_BufferReference<e_Mesh, e_KernelMesh> m = m_pScene->getMesh(n);
		unsigned int s = n->m_uMaterialOffset, l = m_pScene->getMesh(n)->m_sMatInfo.getLength();
		for(int j = 0; j < l; j++)
		{
			e_StreamReference(e_KernelMaterial) mat = m_pScene->m_pMaterialBuffer->operator()(s + j, 1);
			const e_KernelBSSRDF* bssrdf;
			if(mat->GetBSSRDF(MapParameters(make_float3(1), make_float2(0, 0), Frame()), &bssrdf))
			{
				volBox.Enlarge(n->getWorldBox(m));
				m_bLongRunning |= 1;
			}
		}
	}
	m_sMaps.StartNewRendering(m_sEyeBox, volBox, r);
	m_sMaps.StartNewPass();

	float r_scene = length(m_pScene->getKernelSceneData().m_sBox.Size()) / 2;
	r_min = 10e-7f * r_scene;
	r_max = 10e-3f * r_scene;
	float r1 = r_max / 10;
	doStartPass(r1, r1);
}

static bool GGG = false;
static float GGGf0;
static float GGGf1;
void k_sPpmTracer::StartNewTrace(e_Image* I)
{
	m_bDirect = !m_pScene->getVolumes().getLength();m_bDirect=0;
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