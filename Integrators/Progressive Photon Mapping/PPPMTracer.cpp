#include <StdAfx.h>
#include <Engine/Buffer.h>
#include "PPPMTracer.h"
#include <Base/Timer.h>
#include <Engine/DynamicScene.h>
#include <Engine/Node.h>
#include <Engine/Mesh.h>
#include <Engine/Material.h>

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
	: m_pAdpBuffer(0), m_fLightVisibility(1), 
	m_fProbSurface(1), m_fProbVolume(1), m_uBlocksPerLaunch(ComputePhotonBlocksPerPass()),
	m_sSurfaceMap(Vec3u(250), (ComputePhotonBlocksPerPass() + 2) * PPM_slots_per_block), m_sSurfaceMapCaustic(0),
	m_debugScaleVal(1)
{
	m_sParameters
		<< KEY_Direct()					<< CreateSetBool(true)
		<< KEY_FinalGathering()			<< CreateSetBool(false)
		<< KEY_AdaptiveAccProb()		<< CreateSetBool(false)
		<< KEY_RadiiComputationType()	<< PPM_Radius_Type::kNN
		<< KEY_VolRadiusScale()			<< CreateInterval(1.0f, 0.0f, FLT_MAX)
		<< KEY_kNN_Neighboor_Num_Surf() << CreateInterval(10.0f, 0.0f, FLT_MAX)
		<< KEY_kNN_Neighboor_Num_Vol()  << CreateInterval(1.0f, 0.0f, FLT_MAX);

	m_uTotalPhotonsEmitted = -1;
	unsigned int numPhotons = (m_uBlocksPerLaunch + 2) * PPM_slots_per_block;
	if (m_sParameters.getValue(KEY_FinalGathering()))
		m_sSurfaceMapCaustic = new SurfaceMapT(Vec3u(250), numPhotons);
	m_pVolumeEstimator = new PointStorage(150, numPhotons);
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
	if (m_pAdpBuffer)
	{
		m_pAdpBuffer->Free();
		delete m_pAdpBuffer;
	}
	delete m_pVolumeEstimator;
}

void PPPMTracer::PrintStatus(std::vector<std::string>& a_Buf) const
{
	a_Buf.push_back(GET_PERF_BLOCKS().ToString());
	auto radPara = ((EnumTracerParameter<PPM_Radius_Type>*)m_sParameters.operator[](KEY_RadiiComputationType().name));
	a_Buf.push_back("Radius Scheme : " + radPara->getStringValue());
	double pC = math::floor((float)((double)m_uTotalPhotonsEmitted / 1000000.0));
	a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	double pCs = m_uTotalPhotonsEmitted / m_fAccRuntime / 1000000.0f;
	double pCsLastSurf = m_uPhotonEmittedPassSurface / m_fLastRuntime / 1000000.0f, pCsLastVol = m_uPhotonEmittedPassVolume / m_fLastRuntime / 1000000.0f;
	a_Buf.push_back(format("Photons/Sec avg : %f", (float)pCs));
	a_Buf.push_back(format("Photons Surf/Sec lst : %f", (float)pCsLastSurf));
	a_Buf.push_back(format("Photons Vol/Sec lst : %f", (float)pCsLastVol));
	a_Buf.push_back(format("Light Visibility : %f", m_fLightVisibility));
	a_Buf.push_back(format("Photons per pass : %d*100,000", m_uPhotonEmittedPassSurface / 100000));
	a_Buf.push_back(format("%.2f%% Surf Photons", float(m_sSurfaceMap.getNumStoredEntries()) / m_sSurfaceMap.getNumEntries() * 100));
	if (m_sParameters.getValue(KEY_FinalGathering()))
		a_Buf.push_back(format("Caustic surf map : %.2f%%", float(m_sSurfaceMapCaustic->getNumStoredEntries()) / m_sSurfaceMapCaustic->getNumEntries() * 100));
	a_Buf.push_back("Volumeric Estimator : ");
	m_pVolumeEstimator->PrintStatus(a_Buf);
}

void PPPMTracer::Resize(unsigned int _w, unsigned int _h)
{
	Tracer<true, true>::Resize(_w, _h);
	if(m_pAdpBuffer)
	{
		m_pAdpBuffer->Free();
		delete m_pAdpBuffer;
	}
	m_pAdpBuffer = new BlockLoclizedCudaBuffer<APPM_PixelData>(_w, _h);
}

void PPPMTracer::DoRender(Image* I)
{
	//I->Clear();
	{
		auto timer = START_PERF_BLOCK("Photon Pass");
		doPhotonPass(I);
	}
	m_uTotalPhotonsEmitted += max(m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassVolume);
	setNumSequences();
	{
		auto timer = START_PERF_BLOCK("Camera Pass");
		Tracer<true, true>::DoRender(I);
		//DebugInternal(I, Vec2i(481, 240));
		//DebugInternal(I, Vec2i(323, 309));
		//DebugInternal(I, Vec2i(573, 508));
		//std::cout << getCurrentRadius(2) << "\n";
	}
}

float PPPMTracer::getSplatScale()
{
	return 1.0f / m_uPassesDone * (m_uPhotonEmittedPassVolume ? (float)(w * h) / m_uPhotonEmittedPassVolume : 1);
}

float PPPMTracer::getCurrentRadius(float exp, bool surf) const
{
	return CudaTracerLib::getCurrentRadius(surf ? m_fInitialRadiusSurf : m_fInitialRadiusVol, m_uPassesDone, exp);
}

typedef boost::variant<int, float> pixel_variant;
std::map<std::string, pixel_variant> PPPMTracer::getPixelInfo(int x, int y) const
{
	APPM_PixelData pixelInfo = m_pAdpBuffer->operator()(x, y);
	auto e = pixelInfo.m_surfaceData;
	auto r = e.compute_r<2>((int)m_uPassesDone, (int)m_sSurfaceMap.getNumEntries(), (int)m_uTotalPhotonsEmitted, [&](auto gr) {return Lapl(gr); });
	auto rd = e.compute_rd(m_uPassesDone, m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassSurface * (m_uPassesDone - 1));

	auto res = std::map<std::string, pixel_variant>();
	res["S[DI]"] = pixel_variant(Lapl(e.Sum_DI));
	res["S[E[DI]]"] = pixel_variant(e.Sum_E_DI);
	res["S[E[DI]^2]"] = pixel_variant(e.Sum_E_DI2);
	res["S[psi]"] = pixel_variant(e.Sum_psi);
	res["S[psi^2]"] = pixel_variant(e.Sum_psi2);
	res["S[pl]"] = pixel_variant(e.Sum_pl);
	auto r_k_nn = CudaTracerLib::getCurrentRadius(e.r_std, m_uPassesDone, 2), r_uni = getCurrentRadius(2, true);
	res["r_knn"] = pixel_variant(r_k_nn);
	res["r_uni"] = pixel_variant(r_uni);
	res["r_adp"] = pixel_variant(r);
	res["rd"] = pixel_variant(rd);

	auto rT = m_sParameters.getValue(KEY_RadiiComputationType());
	res["r"] = rT == PPM_Radius_Type::Constant ? r_uni : (rT == PPM_Radius_Type::kNN ? r_k_nn : r);
	res["J"] = pixel_variant((int)m_uPhotonEmittedPassSurface);
	res["N"] = pixel_variant((int)m_uPassesDone);

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
	m_uTotalPhotonsEmitted = 0;
#ifdef CUDA_RELEASE_BUILD
	AABB m_sEyeBox = GetEyeHitPointBox(m_pScene, true);
#else
	AABB m_sEyeBox = m_pScene->getSceneBox();
#endif
	float r = m_sEyeBox.Size().length() / float(w) * 5;
	m_sEyeBox.minV -= Vec3f(r);
	m_sEyeBox.maxV += Vec3f(r);
	m_fInitialRadiusSurf = r;
	m_fInitialRadiusVol = m_fInitialRadiusSurf * m_sParameters.getValue(KEY_VolRadiusScale());
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
	m_sSurfaceMap.SetSceneDimensions(m_sEyeBox);
	if (m_sParameters.getValue(KEY_FinalGathering()))
		m_sSurfaceMapCaustic->SetSceneDimensions(m_sEyeBox);
	m_pVolumeEstimator->StartNewRendering(volBox);

	float r_scene = m_sEyeBox.Size().length() / 2;
	r_min = 1e-6f * r_scene;
	r_max = 2e-2f * r_scene;
}

}