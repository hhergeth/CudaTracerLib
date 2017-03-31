#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <SceneTypes/Light.h>
#include <Engine/SpatialStructures/Grid/SpatialGridTraversal.h>
#include <Base/RuntimeTemplateHelper.h>

namespace CudaTracerLib {

CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMap;
CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMapCaustic;
CUDA_CONST unsigned int g_NumPhotonEmittedSurface2, g_NumPhotonEmittedVolume2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[DMAX3(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid))];

CUDA_FUNC_IN Spectrum L_SurfaceFinalGathering(int N_FG_Samples, BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float rad, TraceResult& r2, Sampler& rng, bool DIRECT, unsigned int numPhotonsEmitted, float& pl_est)
{
	Spectrum LCaustic = g_SurfMapCaustic->estimateRadiance(bRec, wi, rad, r2.getMat(), numPhotonsEmitted, pl_est);
	if (!DIRECT)
		LCaustic += UniformSampleOneLight(bRec, r2.getMat(), rng);//the direct light is not stored in the caustic map
	Spectrum L(0.0f);
	BSDFSamplingRecord bRec2;//constantly reloading into bRec and using less registers has about the same performance
	bRec.typeMask = EGlossy | EDiffuse;
	for (int i = 0; i < N_FG_Samples; i++)
	{
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		NormalizedT<Ray> r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = traceRay(r);
		if (r3.hasHit())
		{
			r3.getBsdfSample(r, bRec2, ETransportMode::ERadiance);
			float _;
			L += f * (g_SurfMap->estimateRadiance(bRec2, -r.dir(), rad, r3.getMat(), numPhotonsEmitted, _) + g_SurfMapCaustic->estimateRadiance(bRec2, -r.dir(), rad, r3.getMat(), numPhotonsEmitted, _));
			L += f * UniformSampleOneLight(bRec2, r3.getMat(), rng);
			//do not account for emission because this was sampled before
		}
	}
	bRec.typeMask = ETypeCombinations::EAll;
	return L / (float)N_FG_Samples + LCaustic;
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, k_AdaptiveStruct a_AdpEntries, Image img, bool DIRECT, int N_FG_Samples)
{
	BSDFSamplingRecord bRec;
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		auto rng = g_SamplerData(TracerBase::getPixelIndex(off.x, off.y, w, h));
		auto adp_ent = a_AdpEntries(pixel.x, pixel.y);
		float rad_surf = a_AdpEntries.getRadiusSurf(adp_ent), rad_vol = a_AdpEntries.getRadiusVol<VolEstimator::DIM()>(adp_ent);
		float vol_dens_est_it = 0;
		int numVolEstimates = 0;

		Vec2f screenPos = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		NormalizedT<Ray> r, rX, rY;
		Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());

		TraceResult r2;
		r2.Init();
		int depth = -1;
		Spectrum L(0.0f);
		while (traceRay(r.dir(), r.ori(), &r2) && depth++ < 5)
		{
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);
			if (depth == 0)
				bRec.dg.computePartials(r, rX, rY);
			if (g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				if (g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
				{
					Spectrum Tr(1.0f);
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(rad_vol, g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), Tr, vol_dens_est_it);
					numVolEstimates++;
					throughput = throughput * Tr;
				}
			}
			if (DIRECT && (!g_SceneData.m_sVolume.HasVolumes() || (g_SceneData.m_sVolume.HasVolumes() && depth == 0)))
			{
				float pdf;
				Vec2f sample = rng.randomFloat2();
				const Light* light = g_SceneData.sampleEmitter(pdf, sample);
				DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
				Spectrum value = light->sampleDirect(dRec, rng.randomFloat2()) / pdf;
				bRec.wo = bRec.dg.toLocal(dRec.d);
				bRec.typeMask = EBSDFType(EAll & ~EDelta);
				Spectrum bsdfVal = r2.getMat().bsdf.f(bRec);
				if (!bsdfVal.isZero())
				{
					const float bsdfPdf = r2.getMat().bsdf.pdf(bRec);
					const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
					if (g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
						value = 0.0f;
					float tmin, tmax;
					if (g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(Ray(bRec.dg.P, dRec.d), 0, dRec.dist, &tmin, &tmax))
					{
						Spectrum Tr;
						Spectrum Li = ((VolEstimator*)g_VolEstimator2)->L_Volume(rad_vol, g_NumPhotonEmittedVolume2, NormalizedT<Ray>(bRec.dg.P, dRec.d), tmin, tmax, VolHelper<true>(), Tr, vol_dens_est_it);
						numVolEstimates++;
						value = value * Tr + Li;
					}
					L += throughput * bsdfVal * weight * value;
				}
				bRec.typeMask = EAll;

				//L += throughput * UniformSampleOneLight(bRec, r2.getMat(), rng);
			}
			L += throughput * r2.Le(bRec.dg.P, bRec.dg.sys, -r.dir());//either it's the first bounce or it's a specular reflection
			const VolumeRegion* bssrdf;
			if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				float pdf;
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				NormalizedT<Ray> rTrans = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = traceRay(rTrans);
				Spectrum Tr;
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(rad_vol, g_NumPhotonEmittedVolume2, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr, vol_dens_est_it);
				numVolEstimates++;
				//throughput = throughput * Tr;
				break;
			}
			bool hasDiffuse = r2.getMat().bsdf.hasComponent(EDiffuse),
				hasSpec = r2.getMat().bsdf.hasComponent(EDelta),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasDiffuse)
			{
				Spectrum L_r;//reflected radiance computed by querying photon map
				float pl_est_it = 0;
				L_r = N_FG_Samples != 0 ? L_SurfaceFinalGathering(N_FG_Samples, bRec, -r.dir(), rad_surf, r2, rng, DIRECT, g_NumPhotonEmittedSurface2, pl_est_it) :
										  g_SurfMap->estimateRadiance(bRec, -r.dir(), rad_surf, r2.getMat(), g_NumPhotonEmittedSurface2, pl_est_it);
				adp_ent.surf_density.addSample(pl_est_it);
				L += throughput * L_r;
				if (!hasSpec && !hasGlossy)
					break;
			}
			if (hasSpec || hasGlossy)
			{
				bRec.sampledType = 0;
				bRec.typeMask = EDelta | EGlossy;
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (!bRec.sampledType)
					break;
				throughput = throughput * t_f;
				r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
				r2.Init();
			}
			else break;
		}

		if (!r2.hasHit())
		{
			Spectrum Tr(1);
			float tmin, tmax;
			if (g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
			{
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(rad_vol, (float)g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), Tr, vol_dens_est_it);
				numVolEstimates++;
			}
			L += Tr * throughput * g_SceneData.EvalEnvironment(r);
		}

		img.AddSample(screenPos.x, screenPos.y, L);
		adp_ent.vol_density.addSample(vol_dens_est_it / numVolEstimates);
		a_AdpEntries(pixel.x, pixel.y) = adp_ent;
	}
}

void PPPMTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	if (m_sSurfaceMapCaustic)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMapCaustic, m_sSurfaceMapCaustic, sizeof(*m_sSurfaceMapCaustic)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedSurface2, &m_uPhotonEmittedPassSurface, sizeof(m_uPhotonEmittedPassSurface)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedVolume2, &m_uPhotonEmittedPassVolume, sizeof(m_uPhotonEmittedPassVolume)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	int fg_samples = m_sParameters.getValue(KEY_N_FG_Samples());

	k_AdaptiveStruct A = getAdaptiveData();
	Vec2i off = Vec2i(x, y);
	auto img = *I;

	//iterateTypes<BeamGrid, PointStorage, BeamBeamGrid>(m_pVolumeEstimator, [off,&A,&img, fg_samples,this](auto* X) {CudaTracerLib::k_EyePass<std::remove_pointer<decltype(X)>::type> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, this->w, this->h, A, img, this->m_useDirectLighting, fg_samples); });

	if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, A, img, m_useDirectLighting, fg_samples);
	else if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
		k_EyePass<PointStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, A, img, m_useDirectLighting, fg_samples);
	else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamBeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, A, img, m_useDirectLighting, fg_samples);

	ThrowCudaErrors(cudaThreadSynchronize());
	m_pPixelBuffer->setOnGPU();
}

}
