#include <StdAfx.h>
#include <Engine/Buffer.h>
#include "PPPMTracer.h"
#include <Base/Timer.h>
#include <Engine/DynamicScene.h>
#include <Engine/Node.h>
#include <Engine/Mesh.h>
#include <Engine/Material.h>
#include <iomanip>
#include <Engine/SpatialGridTraversal.h>

namespace CudaTracerLib {

unsigned int ComputePhotonBlocksPerPass()
{
#ifdef CUDA_RELEASE_BUILD
	return 180;
#else
	return 1;
#endif
}

PPPMTracer::PPPMTracer()
	: m_pPixelBuffer(0), m_fLightVisibility(1), 
	m_fProbSurface(1), m_fProbVolume(1.0f), m_uBlocksPerLaunch(ComputePhotonBlocksPerPass()),
	m_sSurfaceMap(Vec3u(250), (ComputePhotonBlocksPerPass() + 2) * PPM_slots_per_block), m_sSurfaceMapCaustic(0)
{
	m_sParameters
		<< KEY_Direct()						<< CreateSetBool(true)
		<< KEY_N_FG_Samples()				<< CreateInterval(0, 0, INT_MAX)
		<< KEY_AdaptiveAccProb()			<< CreateSetBool(false)
		<< KEY_RadiiComputationTypeSurf()	<< PPM_Radius_Type::Constant
		<< KEY_RadiiComputationTypeVol()	<< PPM_Radius_Type::Constant
		<< KEY_VolRadiusScale()				<< CreateInterval(1.0f, 0.0f, FLT_MAX)
		<< KEY_kNN_Neighboor_Num_Surf()		<< CreateInterval(50.0f, 0.0f, FLT_MAX)
		<< KEY_kNN_Neighboor_Num_Vol()		<< CreateInterval(50.0f, 0.0f, FLT_MAX);

	m_uTotalPhotonsEmittedSurface = m_uTotalPhotonsEmittedVolume = -1;
	unsigned int numPhotons = (m_uBlocksPerLaunch + 2) * PPM_slots_per_block;
	if (m_sParameters.getValue(KEY_N_FG_Samples()) != 0)
		m_sSurfaceMapCaustic = new SurfaceMapT(Vec3u(250), numPhotons);
	//m_pVolumeEstimator = new PointStorage(150, numPhotons);
	m_pVolumeEstimator = new BeamGrid(150, numPhotons, 10, 2);
	//m_pVolumeEstimator = new BeamBeamGrid(10, 10000, 1000);
	if (m_sParameters.getValue(KEY_AdaptiveAccProb()))
	{
		size_t volLength, volCount;
		m_pVolumeEstimator->getStatusInfo(volLength, volCount);
		m_fProbVolume = math::clamp01(volLength / (float)m_sSurfaceMap.getNumEntries());
	}
}

PPPMTracer::~PPPMTracer()
{
	m_sSurfaceMap.Free();
	if (m_pPixelBuffer)
	{
		m_pPixelBuffer->Free();
		delete m_pPixelBuffer;
	}
	delete m_pVolumeEstimator;
}

void PPPMTracer::PrintStatus(std::vector<std::string>& a_Buf) const
{
	a_Buf.push_back(GET_PERF_BLOCKS().ToString());
	auto radParaSurf = ((EnumTracerParameter<PPM_Radius_Type>*)m_sParameters.operator[](KEY_RadiiComputationTypeSurf().name));
	a_Buf.push_back("Surf Radius Scheme : " + radParaSurf->getStringValue());
	auto radParaVol = ((EnumTracerParameter<PPM_Radius_Type>*)m_sParameters.operator[](KEY_RadiiComputationTypeVol().name));
	a_Buf.push_back("Vol Radius Scheme : " + radParaVol->getStringValue());
	double pC = math::floor((float)((double)m_uTotalPhotonsEmittedSurface / 1000000.0));
	a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	double pCs = m_uTotalPhotonsEmittedSurface / m_fAccRuntime / 1000000.0f;
	double pCsLastSurf = m_uPhotonEmittedPassSurface / m_fLastRuntime / 1000000.0f, pCsLastVol = m_uPhotonEmittedPassVolume / m_fLastRuntime / 1000000.0f;
	a_Buf.push_back(format("Photons/Sec avg : %f", (float)pCs));
	a_Buf.push_back(format("Photons Surf/Sec lst : %f", (float)pCsLastSurf));
	a_Buf.push_back(format("Photons Vol/Sec lst : %f", (float)pCsLastVol));
	a_Buf.push_back(format("Light Visibility : %f", m_fLightVisibility));
	a_Buf.push_back(format("Photons per pass : %d*100,000", m_uPhotonEmittedPassSurface / 100000));
	a_Buf.push_back(format("%.2f%% Surf Photons", float(m_sSurfaceMap.getNumStoredEntries()) / m_sSurfaceMap.getNumEntries() * 100));
	if (m_sParameters.getValue(KEY_N_FG_Samples()) != 0)
		a_Buf.push_back(format("Caustic surf map : %.2f%%", float(m_sSurfaceMapCaustic->getNumStoredEntries()) / m_sSurfaceMapCaustic->getNumEntries() * 100));
	a_Buf.push_back("Volumeric Estimator : ");
	m_pVolumeEstimator->PrintStatus(a_Buf);
}

void PPPMTracer::Resize(unsigned int _w, unsigned int _h)
{
	Tracer<true, true>::Resize(_w, _h);
	if(m_pPixelBuffer)
	{
		m_pPixelBuffer->Free();
		delete m_pPixelBuffer;
	}
	m_pPixelBuffer = new SynchronizedBuffer<APPM_PixelData>(_w * _h);
}

void PPPMTracer::DoRender(Image* I)
{
	//I->Clear();
	{
		auto timer = START_PERF_BLOCK("Photon Pass");
		doPhotonPass(I);
	}
	m_uTotalPhotonsEmittedSurface += m_uPhotonEmittedPassSurface;
	m_uTotalPhotonsEmittedVolume += m_uPhotonEmittedPassVolume;
	setNumSequences();
	{
		auto timer = START_PERF_BLOCK("Camera Pass");
		Tracer<true, true>::DoRender(I);
	}
}

k_AdaptiveStruct PPPMTracer::getAdaptiveData() const
{
	auto r_scene_surf = m_boxSurf.Size().length() / 2;
	auto surf_min = r_scene_surf * 1e-5f, surf_max = r_scene_surf * 1e-1f;

	auto r_scene_vol = m_boxVol.Size().length() / 2;
	auto vol_min = r_scene_vol * 1e-5f, vol_max = r_scene_vol * 1e-1f;

	float k_toFindSurf = m_sParameters.getValue(KEY_kNN_Neighboor_Num_Surf());// / (m_uTotalPhotonsEmittedSurface / m_uPassesDone);
	float k_toFindVol = m_sParameters.getValue(KEY_kNN_Neighboor_Num_Vol());// / (m_uTotalPhotonsEmittedVolume / m_uPassesDone);

	auto radiusTypeSurf = m_sParameters.getValue(KEY_RadiiComputationTypeSurf());
	auto radiusTypeVol = m_sParameters.getValue(KEY_RadiiComputationTypeVol());

	return k_AdaptiveStruct(m_fInitialRadiusSurf, m_fInitialRadiusVol, surf_min, surf_max, vol_min, vol_max, *m_pPixelBuffer, w, m_uPassesDone, m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassVolume, k_toFindSurf, k_toFindVol, radiusTypeSurf, radiusTypeVol);
}

float PPPMTracer::getSplatScale()
{
	return 1.0f / m_uPassesDone * (m_uPhotonEmittedPassVolume ? (float)(w * h) / m_uPhotonEmittedPassVolume : 1);
}

typedef boost::variant<int, float> pixel_variant;
std::map<std::string, pixel_variant> PPPMTracer::getPixelInfo(int x, int y) const
{
	m_pPixelBuffer->Synchronize();
	auto pixelInfo = m_pPixelBuffer->operator[](y * w + x);
	auto res = std::map<std::string, pixel_variant>();
	auto dat = getAdaptiveData();

	res["pl_surf"] = pixelInfo.surf_density.computeDensityEstimate(m_uPassesDone, m_uTotalPhotonsEmittedSurface);
	res["pl_vol"] = pixelInfo.vol_density.computeDensityEstimate(m_uPassesDone, m_uTotalPhotonsEmittedVolume);

	res["RadiiComputationTypeSurf"] = m_sParameters.getValue(KEY_RadiiComputationTypeSurf());
	res["RadiiComputationTypeVol"] = m_sParameters.getValue(KEY_RadiiComputationTypeVol());

	res["r_surf_uni"] = dat.m_radSurf;
	res["r_surf_kNN"] = dat.computekNNRadiusSurf(pixelInfo);

	return res;
}

void PPPMTracer::StartNewTrace(Image* I)
{
	m_useDirectLighting = !m_pScene->getVolumes().hasElements() && m_sParameters.getValue(KEY_Direct());
#ifdef CUDA_RELEASE_BUILD
	m_fLightVisibility = Tracer::GetLightVisibility(m_pScene, 1);
#endif
	m_useDirectLighting &= m_fLightVisibility > 0.5f;
	//m_bDirect = 0;
	Tracer<true, true>::StartNewTrace(I);
	m_pPixelBuffer->Memset(0);
	m_uTotalPhotonsEmittedSurface = m_uTotalPhotonsEmittedVolume = 0;
#ifdef CUDA_RELEASE_BUILD
	m_boxSurf = GetEyeHitPointBox(m_pScene, true);
#else
	m_boxSurf = m_pScene->getSceneBox();
#endif
	float rad = m_boxSurf.Size().length() / 2.0f;
	float r = min(rad / w, rad / h) * 5.0f;
	m_boxSurf.minV -= Vec3f(r);
	m_boxSurf.maxV += Vec3f(r);
	m_fInitialRadiusSurf = r;
	m_fInitialRadiusVol = m_fInitialRadiusSurf * m_sParameters.getValue(KEY_VolRadiusScale());
	m_boxVol = m_pScene->getKernelSceneData().m_sVolume.box;
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
				m_boxVol = m_boxVol.Extend(m_pScene->getNodeBox(it));
			}
		}
	}
	m_sSurfaceMap.SetSceneDimensions(m_boxSurf);
	if (m_sParameters.getValue(KEY_N_FG_Samples()) != 0)
		m_sSurfaceMapCaustic->SetSceneDimensions(m_boxSurf);
	m_pVolumeEstimator->StartNewRenderingBase(m_fInitialRadiusSurf, m_fInitialRadiusVol);
	m_pVolumeEstimator->StartNewRendering(m_boxVol);
}

}