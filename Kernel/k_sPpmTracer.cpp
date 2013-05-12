#include <StdAfx.h>
#include "k_sPpmTracer.h"

#define LNG 200

k_sPpmTracer::k_sPpmTracer()
	: k_RandTracerBase(), m_uGridLength(LNG*LNG*LNG)
{
#ifdef DEBUG
	m_uNewPhotonsPerRun = 0.1f;
#else
	m_uNewPhotonsPerRun = 5;
#endif
	m_uModus = 1;
	m_pDevicePixels = 0;
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