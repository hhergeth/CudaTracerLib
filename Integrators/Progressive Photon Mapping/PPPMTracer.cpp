#include <StdAfx.h>
#include <Engine/Buffer.h>
#include "PPPMTracer.h"
#include <Base/Timer.h>
#include <Engine/DynamicScene.h>
#include <Engine/Node.h>
#include <Engine/Mesh.h>

namespace CudaTracerLib {

PPPMTracer::PPPMTracer()
	: m_pEntries(0), m_fLightVisibility(1), k_Intial(25)
{
	m_sParameters << KEY_Direct() << CreateSetBool(true)
				  << KEY_PerPixelRadius() << CreateSetBool(false);
#ifdef NDEBUG
	m_uBlocksPerLaunch = 180;
#else
	m_uBlocksPerLaunch = 1;
#endif
	m_uTotalPhotonsEmitted = -1;
	unsigned int numPhotons = (m_uBlocksPerLaunch + 2) * PPM_slots_per_block;
	m_sSurfaceMap = SpatialLinkedMap<PPPMPhoton>(250, numPhotons * PPM_MaxRecursion / 10);
	//m_pVolumeEstimator = new PointStorage(100, numPhotons * PPM_MaxRecursion / 10);
	//m_pVolumeEstimator = new BeamGrid(100, numPhotons * PPM_MaxRecursion / 10, 20, 10);
	//m_pVolumeEstimator = new BeamBVHStorage(100000);
	m_pVolumeEstimator = new BeamBeamGrid(10, 30000, 3000);
}

void PPPMTracer::PrintStatus(std::vector<std::string>& a_Buf) const
{
	a_Buf.push_back(GET_PER_BLOCKS().ToString());
	double pC = math::floor((double)m_uTotalPhotonsEmitted / 1000000.0);
	a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	double pCs = m_uTotalPhotonsEmitted / m_fAccRuntime / 1000000.0f;
	double pCsLast = m_uPhotonEmittedPass / m_fLastRuntime / 1000000.0f;
	a_Buf.push_back(format("Photons/Sec avg : %f", (float)pCs));
	a_Buf.push_back(format("Photons/Sec lst : %f", (float)pCsLast));
	a_Buf.push_back(format("Light Visibility : %f", m_fLightVisibility));
	a_Buf.push_back(format("Photons per pass : %d*100,000", m_sSurfaceMap.numData / 100000));
	a_Buf.push_back(format("%.2f%% Surf Photons", float(m_sSurfaceMap.deviceDataIdx) / m_sSurfaceMap.numData * 100));
	a_Buf.push_back("Volumeric Estimator : ");
	m_pVolumeEstimator->PrintStatus(a_Buf);
}

void PPPMTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0.1f, 10.0f, true, &m_fInitialRadiusScale, "Initial radius = %g units");
	//a_Callback(0.1f, 100.0f, true, &m_uNewPhotonsPerRun, "Number of photons per pass = %g [M]");
}

void PPPMTracer::Resize(unsigned int _w, unsigned int _h)
{
	Tracer<true, true>::Resize(_w, _h);
	if (m_pEntries)
		CUDA_FREE(m_pEntries);
	CUDA_MALLOC(&m_pEntries, sizeof(k_AdaptiveEntry) * _w * _h);
}

void PPPMTracer::DoRender(Image* I)
{
	//I->Clear();
	{
		auto timer = START_PERF_BLOCK("Photon Pass");
		doPhotonPass();
	}
	m_uTotalPhotonsEmitted += m_uPhotonEmittedPass;
	{
		auto timer = START_PERF_BLOCK("Camera Pass");
		Tracer<true, true>::DoRender(I);
	}
}

void PPPMTracer::getRadiusAt(int x, int y, float& r, float& rd) const
{
	k_AdaptiveEntry e;
	ThrowCudaErrors(cudaMemcpy(&e, m_pEntries + w * y + x, sizeof(e), cudaMemcpyDeviceToHost));
	r = e.compute_r(m_uPassesDone, m_sSurfaceMap.numData, m_uTotalPhotonsEmitted);
	rd = e.compute_rd(m_uPassesDone);
}

void PPPMTracer::StartNewTrace(Image* I)
{
	m_useDirectLighting = !m_pScene->getVolumes().hasElements() && m_sParameters.getValue(KEY_Direct());
#ifndef _DEBUG
	m_fLightVisibility = Tracer::GetLightVisibility(m_pScene, 1);
#endif
	m_useDirectLighting &= m_fLightVisibility > 0.5f;
	//m_bDirect = 0;
	Tracer<true, true>::StartNewTrace(I);
	m_uTotalPhotonsEmitted = 0;
#ifndef _DEBUG
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
#else
	AABB m_sEyeBox = m_pScene->getSceneBox();
#endif
	float r = (m_sEyeBox.maxV - m_sEyeBox.minV).sum() / float(w);
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadius = r;
	AABB volBox = m_pScene->getKernelSceneData().m_sVolume.box;
	for (auto it : m_pScene->getNodes())
	{
		StreamReference<Material> mats = m_pScene->getMaterials(it);
		for (unsigned int j = 0; j < mats.getLength(); j++)
		{
			const VolumeRegion* bssrdf;
			DifferentialGeometry dg;
			ZERO_MEM(dg);
			if (mats(j)->GetBSSRDF(dg, &bssrdf))
			{
				volBox = volBox.Extend(m_pScene->getNodeBox(it));
			}
		}
	}
	m_sSurfaceMap.SetSceneDimensions(m_sEyeBox, r);
	m_pVolumeEstimator->StartNewRendering(volBox, r);

	float r_scene = m_sEyeBox.Size().length() / 2;
	r_min = 1e-6f * r_scene;
	r_max = 1e-1f * r_scene;
}

}