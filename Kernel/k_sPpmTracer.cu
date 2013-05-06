#include "k_sPpmTracer.h"

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

void k_sPpmTracer::DoRender(RGBCOL* a_Buf)
{
	if(m_uModus == 1)
	{
		updateBuffer();
		doPhotonPass();
		cudaThreadSynchronize();
		if(m_sMaps.PassFinished())
		{
			m_uPassesDone++;
			
			//print(m_sMaps);

			m_sRngs.m_uOffset++;
			m_uPhotonsEmitted += m_sMaps.m_uPhotonNumEmitted;
			doEyePass(a_Buf);
			m_sMaps.StartNewPass();
		}
	}
	else
	{
		m_uPassesDone = 1;
		cudaMemset(m_pDevicePixels, 0, sizeof(k_sPpmPixel) * w * h);
		cudaMemset(a_Buf, 0, sizeof(RGBCOL) * w * h);
		doEyePass(a_Buf);
	}
}

void k_sPpmTracer::initNewPass(RGBCOL* a_Buf)
{
	m_uPassesDone = 0;
	m_uPhotonsEmitted = 0;
	AABB m_sEyeBox = GetEyeHitPointBox();
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / w * m_fInitialRadiusScale;
	m_sEyeBox.minV -= make_float3(r);
	m_sEyeBox.maxV += make_float3(r);
	m_fInitialRadius = r;
	m_sMaps.StartNewRendering(m_sEyeBox, m_pScene->getKernelSceneData().m_sVolume.box, r);
	m_sMaps.StartNewPass();
	cudaMemset(m_pDevicePixels, 0, sizeof(k_sPpmPixel) * w * h);
	cudaMemset(a_Buf, 0, sizeof(RGBCOL) * w * h);
}

static bool GGG = false;
static float GGGf0;
static float GGGf1;
void k_sPpmTracer::StartNewTrace(RGBCOL* a_Buf)
{
	m_bDirect = !m_pScene->getVolumes().getLength();m_bDirect=0;
	if(m_uModus == 1)
	{
		initNewPass(a_Buf);
	}
	else if(!GGG || GGGf0 != m_fInitialRadiusScale || GGGf1 != m_uNewPhotonsPerRun)
	{
		initNewPass(a_Buf);
		GGGf1 = m_uNewPhotonsPerRun;
		GGGf0 = m_fInitialRadiusScale;
		GGG = true;
		updateBuffer();
		while(!m_sMaps.PassFinished())
			doPhotonPass();
		m_uPassesDone = 1;
	}
}