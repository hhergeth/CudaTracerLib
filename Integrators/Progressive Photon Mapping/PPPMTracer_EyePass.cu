#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Engine/Light.h>
#include <fstream>
#include <Windows.h>
#include <Engine/SpatialGridTraversal.h>

namespace CudaTracerLib {

template<bool USE_GLOBAL> Spectrum PointStorage::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	Spectrum L_n = Spectrum(0.0f);
	float a, b;
	if (!m_sStorage.getHashGrid().getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	float minT = a = math::clamp(a, tmin, tmax);
	b = math::clamp(b, tmin, tmax);
	float d = 2.0f * m_fCurrentRadiusVol;
	while (a < b)
	{
		float t = a + d / 2.0f;
		Vec3f x = r(t);
		Spectrum L_i(0.0f);
		m_sStorage.ForAll(x - Vec3f(m_fCurrentRadiusVol), x + Vec3f(m_fCurrentRadiusVol), [&](const Vec3u& cell_idx, unsigned int p_idx, const volPhoton& ph)
		{
			Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_idx);
			auto dist2 = distanceSquared(ph_pos, x);
			if (dist2 < math::sqr(m_fCurrentRadiusVol))
			{
				PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
				float p = vol.p(x, pRec);
				L_i += p * ph.getL() / NumEmitted * Kernel::k<3>(math::sqrt(dist2), m_fCurrentRadiusVol);
			}
		});
		L_n += (-Tau - vol.tau(r, a, t)).exp() * L_i * d;
		Tau += vol.tau(r, a, a + d);
		L_n += vol.Lve(x, -r.dir()) * d;
		a += d;
	}
	Tr = (-Tau).exp();
	return L_n;
}

template<bool USE_GLOBAL> Spectrum BeamGrid::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	Spectrum L_n = Spectrum(0.0f);
	TraverseGridRay(r, m_sStorage.getHashGrid(), tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sBeamGridStorage.ForAllCellEntries(cell_pos, [&](unsigned int, entry beam_idx)
		{
			const volPhoton& ph = m_sStorage(beam_idx.getIndex());
			Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_pos);
			float ph_rad1 = ph.getRad1(), ph_rad2 = math::sqr(ph_rad1);
			float l1 = dot(ph_pos - r.ori(), r.dir());
			float isectRadSqr = distanceSquared(ph_pos, r(l1));
			if (isectRadSqr < ph_rad2 && rayT <= l1 && l1 <= cellEndT)
			{
				//transmittance from camera vertex along ray to query point
				Spectrum tauToPhoton = (-Tau - vol.tau(r, rayT, l1)).exp();
				PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
				float p = vol.p(ph_pos, pRec);
				L_n += p * ph.getL() / NumEmitted * tauToPhoton * Kernel::k<2>(math::sqrt(isectRadSqr), ph_rad1);
			}
			/*float t1, t2;
			if (sphere_line_intersection(ph_pos, ph_rad2, r, t1, t2))
			{
				float t = (t1 + t2) / 2;
				auto b = r(t);
				float dist = distance(b, ph_pos);
				auto o_s = vol.sigma_s(b, r.dir()), o_a = vol.sigma_a(b, r.dir()), o_t = Spectrum(o_s + o_a);
				if (dist < ph_rad1 && rayT <= t && t <= cellEndT)
				{
					PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
					float p = vol.p(b, pRec);

					//auto T1 = (-vol.tau(r, 0, t1)).exp(), T2 = (-vol.tau(r, 0, t2)).exp(),
					//	 ta = (t2 - t1) * (T1 + 0.5 * (T2 - T1));
					//L_n += p * ph.getL() / NumEmitted * Kernel::k<3>(dist, ph_rad1) * ta;
					auto Tr_c = (-vol.tau(r, 0, t)).exp();
					L_n += p * ph.getL() / NumEmitted * Kernel::k<3>(dist, ph_rad1) * Tr_c * (t2 - t1);
				}
			}*/
		});
		Tau += vol.tau(r, rayT, cellEndT);
		float localDist = cellEndT - rayT;
		L_n += vol.Lve(r(rayT + localDist / 2), -r.dir()) * localDist;
	});
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMap;
CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMapCaustic;

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum beam_beam_L(const VolHelper<USE_GLOBAL>& vol, const Beam& B, const NormalizedT<Ray>& r, float radius, float beamIsectDist, float queryIsectDist, float beamBeamDistance, float m_uNumEmitted, float sinTheta, float tmin)
{
	Spectrum photon_tau = vol.tau(Ray(B.getPos(), B.getDir()), 0, beamIsectDist);
	Spectrum camera_tau = vol.tau(r, tmin, queryIsectDist);
	Spectrum camera_sc = vol.sigma_s(r(queryIsectDist), r.dir());
	PhaseFunctionSamplingRecord pRec(-r.dir(), B.getDir());
	float p = vol.p(r(queryIsectDist), pRec);
	return B.getL() / m_uNumEmitted * (-photon_tau).exp() * camera_sc * Kernel::k<1>(beamBeamDistance, radius) / sinTheta * (-camera_tau).exp() * 0.5f;//this is not correct; the phase function is missing and the 0.5 is arbirtary scaling
}

template<bool USE_GLOBAL> Spectrum BeamBeamGrid::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	struct BeamIntersectionData
	{
		float beamBeamDistance;
		float sinTheta;
		float beamIsectDist;
		Beam B;
	};

	TraverseGridBeam(r, tmin, tmax, m_sStorage,
		[&](const Vec3u& cell_pos, float rayT, float cellEndT)
	{
		return m_fCurrentRadiusVol;
	},
		[&](const Vec3u& cell_idx, unsigned int ref_element_idx, int beam_idx, float& distAlongRay)
	{
		BeamIntersectionData dat;
		dat.B = this->m_sBeamStorage[beam_idx];
		if(Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, dat.B.getPos(), dat.B.getDir(), 0, dat.B.t, math::sqr(m_fCurrentRadiusVol), dat.beamBeamDistance, dat.sinTheta, distAlongRay, dat.beamIsectDist))
		{
			auto hit_cell = m_sStorage.getHashGrid().Transform(r(distAlongRay));
			if (hit_cell != cell_idx)
				distAlongRay = -1;
		}
		else distAlongRay = -1;
		return dat;
	},
		[&](float rayT, float cellEndT, float minT, float maxT, const Vec3u& cell_idx, unsigned int element_idx, int beam_idx, float distAlongRay, const BeamIntersectionData& dat)
	{
		L_n += beam_beam_L(vol, dat.B, r, m_fCurrentRadiusVol, dat.beamIsectDist, distAlongRay, dat.beamBeamDistance, NumEmitted, dat.sinTheta, tmin);
	}
	);

	/*for (unsigned int i = 0; i < min(m_uBeamIdx, m_sBeamStorage.getLength()); i++)
	{
		const Beam& B = m_sBeamStorage[i];
		float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, B.getPos(), B.getDir(), 0, B.t, math::sqr(m_fCurrentRadiusVol), beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, B, r, m_fCurrentRadiusVol, beamIsectDist, queryIsectDist, beamBeamDistance, NumEmitted, sinTheta, tmin);
	}
	Tr = (-vol.tau(r, tmin, tmax)).exp();*/
	return L_n;
}

template<bool USE_GLOBAL> Spectrum BeamBVHStorage::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const
{
	Spectrum L_n = Spectrum(0.0f);
	iterateBeams(Ray(r(tmin), r.dir()), tmax - tmin, [&](unsigned int ref_idx)
	{
		const BeamRef& R = m_pDeviceRefs[ref_idx];
		//unsigned int beam_idx = R.getIdx();
		const Beam& B = R.beam;// this->m_pDeviceBeams[beam_idx];
		float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, B.getPos(), B.getDir(), R.t_min, R.t_max, math::sqr(m_fCurrentRadiusVol), beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, B, r, m_fCurrentRadiusVol, beamIsectDist, queryIsectDist, beamBeamDistance, NumEmitted, sinTheta, tmin);
	});
	Tr = (-vol.tau(r, tmin, tmax)).exp();
	return L_n;
}

CUDA_CONST unsigned int g_NumPhotonEmittedSurface2, g_NumPhotonEmittedVolume2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax3(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid))];
template<bool F_IS_GLOSSY> CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float r, const Material& mat, unsigned int numPhotonsEmitted, SurfaceMapT* map = 0)
{
	if (!map) map = &g_SurfMap.As();
	Spectrum Lp = Spectrum(0.0f);
	Vec3f a = r*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, c = r*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	map->ForAll(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
	{
		float dist2 = distanceSquared(ph.getPos(map->getHashGrid(), cell_idx), bRec.dg.P);
		Vec3f photonNormal = ph.getNormal();
		float wiDotGeoN = absdot(photonNormal, wi);
		if (dist2 < r * r && dot(photonNormal, bRec.dg.sys.n) > 0.9f && wiDotGeoN > 1e-2f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			float cor_fac = math::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
			float ke = Kernel::k<2>(math::sqrt(dist2), r);
			Spectrum l = ph.getL();
			if(F_IS_GLOSSY)
				l *= mat.bsdf.f(bRec) / Frame::cosTheta(bRec.wo);//bsdf.f returns f * cos(thetha)
			Lp += ke * l;// * cor_fac;
		}
	});
	if(!F_IS_GLOSSY)
	{
		auto wi_l = bRec.wi;
		bRec.wo = bRec.wi = NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f);
		Lp *= mat.bsdf.f(bRec);
		bRec.wi = wi_l;
	}
	return Lp / numPhotonsEmitted;
}

template<bool F_IS_GLOSSY> CUDA_FUNC_IN Spectrum L_SurfaceFinalGathering(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float rad, TraceResult& r2, Sampler& rng, bool DIRECT, unsigned int numPhotonsEmitted)
{
	Spectrum LCaustic = L_Surface<F_IS_GLOSSY>(bRec, wi, rad, r2.getMat(), numPhotonsEmitted, &g_SurfMapCaustic.As());
	Spectrum L(0.0f);
	const int N = 3;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec2(dg);//constantly reloading into bRec and using less registers has about the same performance
	bRec.typeMask = EGlossy | EDiffuse;
	for (int i = 0; i < N; i++)
	{
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		NormalizedT<Ray> r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = traceRay(r);
		if (r3.hasHit())
		{
			r3.getBsdfSample(r, bRec2, ETransportMode::ERadiance);
			bool hasGlossy = r3.getMat().bsdf.hasComponent(EGlossy);
			L += f * (hasGlossy ? L_Surface<true>(bRec2, -r.dir(), rad, r3.getMat(), numPhotonsEmitted) : L_Surface<false>(bRec2, -r.dir(), rad, r3.getMat(), numPhotonsEmitted));
			if (DIRECT)
				L += f * UniformSampleOneLight(bRec2, r3.getMat(), rng);
			else L += f * r3.Le(bRec2.dg.P, bRec2.dg.sys, -r.dir());
		}
	}
	bRec.typeMask = ETypeCombinations::EAll;
	return L / N + LCaustic;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float a_rSurfaceUNUSED, const Material* mat, k_AdaptiveStruct& A, int x, int y, const Spectrum& importance, int iteration, BlockSampleImage& img, SurfaceMapT& surfMap, unsigned int numPhotonsEmittedSurf, float debugScaleVal)
{
	//ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	//ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);

	//float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
	//float ta = (2.0f * math::sqrt(VAR_Psi)) / (PI * float(g_NumPhotonEmittedSurface2) * E_pl * k_22 * E_I * E_I);
	//float q = math::pow(ta, 1.0f / 6.0f) * math::pow(iteration, -1.0f / 6.0f);
	//float p = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	//ent.r = math::clamp(q, A.r_min, A.r_max);

	//Adaptive Progressive Photon Mapping Implementation
	bool hasGlossy = mat->bsdf.hasComponent(EGlossy);
	auto bsdf_diffuse = Spectrum(1);
	if(!hasGlossy)
	{
		auto wi_l = bRec.wi;
		bRec.wo = bRec.wi = NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f);
		bsdf_diffuse = mat->bsdf.f(bRec);
		bRec.wi = wi_l;
	}
	k_AdaptiveEntry ent = A(x, y);
	float r  = iteration <= 1 ? getCurrentRadius(ent.r_std, iteration, 2) :  ent.compute_r(iteration - 1, numPhotonsEmittedSurf, numPhotonsEmittedSurf * (iteration - 1)),
		  rd = iteration <= 1 ? getCurrentRadius(ent.r_std, iteration, 2) : ent.compute_rd(iteration - 1, numPhotonsEmittedSurf, numPhotonsEmittedSurf * (iteration - 1));
	r = math::clamp(r, A.r_min, A.r_max);
	rd = math::clamp(rd, A.r_min, A.r_max);
	rd = a_rSurfaceUNUSED;
	//r = rd = a_rSurfaceUNUSED;
	Vec3f ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	float r_max = max(2 * rd, r);
	Vec3f a = r_max*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r_max*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, 
		  c = r_max*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r_max*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	Spectrum Lp = 0.0f;
	float Sum_psi = 0, Sum_psi2 = 0, n_psi = 0, Sum_DI = 0;
	surfMap.ForAll(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)//CAP = DANGEROUS !!!
	{
		Vec3f ph_pos = ph.getPos(surfMap.getHashGrid(), cell_idx);
		float dist2 = distanceSquared(ph_pos, bRec.dg.P);
		Vec3f photonNormal = ph.getNormal();
		float wiDotGeoN = absdot(photonNormal, wi);
		if (dist2 < math::sqr(r_max) && dot(photonNormal, bRec.dg.sys.n) > 0.9f && wiDotGeoN > 1e-2f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			auto bsdfFactor = hasGlossy ? mat->bsdf.f(bRec) : bsdf_diffuse;
			float psi = Spectrum(importance * bsdfFactor * ph.getL()).getLuminance();
			const Vec3f e_l = bRec.dg.P - ph_pos;
			float k_rd = Kernel::k<2>(e_l, rd);
			float laplu = Kernel::k<2>(e_l + ur, rd) + Kernel::k<2>(e_l - ur, rd) - 2 * k_rd,
				  laplv = Kernel::k<2>(e_l + vr, rd) + Kernel::k<2>(e_l - vr, rd) - 2 * k_rd,
				  lapl = psi / (rd * rd) * (laplu + laplv);
			Sum_DI += lapl;

			if (dist2 < r * r)
			{
				float kri = Kernel::k<2>(math::sqrt(dist2), r);
				Lp += kri * ph.getL() / float(numPhotonsEmittedSurf) * bsdfFactor;
				psi /= numPhotonsEmittedSurf;
				Sum_psi += psi;
				Sum_psi2 += math::sqr(psi);
				ent.Sum_pl += kri;
				n_psi++;
			}
		}
	});
	auto E_DI = Sum_DI / numPhotonsEmittedSurf;
	ent.Sum_DI += E_DI;
	//auto E_DI = ent.Sum_DI / (numPhotonsEmittedSurf * iteration);//TODO : Use the sequence of estimators not the average?
	E_DI = ent.Sum_DI / iteration;
	ent.Sum_E_DI += E_DI;
	ent.Sum_E_DI2 += math::sqr(E_DI);
	if(n_psi != 0)
	{
		ent.Sum_psi += Sum_psi / n_psi;
		ent.Sum_psi2 += Sum_psi2 / n_psi;
	}

	//if (x == 488 && y == 654)
	//	printf("Var[Psi] = %5.5e, Var[Lapl] = %5.5e, E[pl] = %5.5e, E[I] = %5.5e, E[Psi] = %5.5e\n", VAR_Psi, VAR_Lapl, E_pl, E_I, ent.psi / NJ);

	Spectrum qs;
	//float t = (ent.r - A.r_min) / (A.r_max - A.r_min);
	//float t = math::abs(E_I) * 1e-3f / (g_SceneData.m_sBox.Size().length() / 2);
	//float t = ent.DI * 100000000;
	//float t = ent.compute_r(iteration, g_NumPhotonEmittedSurface2, g_NumPhotonEmittedSurface2 * iteration) / (A.r_max * 1 - A.r_min);
	//float t = ent.compute_rd(iteration, numPhotonsEmittedSurf, numPhotonsEmittedSurf * (iteration - 1)) / (A.r_max - A.r_min);
	//float t = (ent.Sum_psi2 / NJ - math::sqr(ent.Sum_psi / NJ)) * debugScaleVal;
	//float t = ent.Sum_pl / NJ * debugScaleVal;
	//float t = math::abs(ent.Sum_DI2 / NJ - math::sqr(ent.Sum_DI / NJ)) * debugScaleVal;
	float t = math::abs(ent.DI) * debugScaleVal;
	qs.fromHSL(2.0f / 3.0f * (1 - math::clamp01(t)), 1, 0.5f);//0 -> 1 : Dark Blue -> Light Blue -> Green -> Yellow -> Red
	//img.Add(x, y, qs);
#ifdef ISCUDA
	A(x, y) = ent;
#endif
	return Lp;
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, k_AdaptiveStruct a_AdpEntries, BlockSampleImage img, bool DIRECT, PPM_Radius_Type Radius_Type, bool finalGathering, float debugScaleVal)
{
	auto rng = g_SamplerData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
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
				dg.computePartials(r, rX, rY);
			if (g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				if (g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
				{
					Spectrum Tr(1.0f);
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), Tr);
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
						Spectrum Li = ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, NormalizedT<Ray>(bRec.dg.P, dRec.d), tmin, tmax, VolHelper<true>(), Tr);
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
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				NormalizedT<Ray> rTrans = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = traceRay(rTrans);
				Spectrum Tr;
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr);
				//throughput = throughput * Tr;
				break;
			}
			bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
				hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasSmooth)
			{
				Spectrum l;
				float rad = math::clamp(getCurrentRadius(a_AdpEntries(pixel.x, pixel.y).r_std, a_PassIndex, 2), a_AdpEntries.r_min, a_AdpEntries.r_max);
				if (Radius_Type == PPM_Radius_Type::Adaptive)
					l = throughput * L_Surface(bRec, -r.dir(), a_rSurface, &r2.getMat(), a_AdpEntries, pixel.x, pixel.y, throughput, a_PassIndex, img, g_SurfMap, g_NumPhotonEmittedSurface2, debugScaleVal);
				else
				{
					float r_i = Radius_Type == PPM_Radius_Type::kNN ? rad : a_rSurface;
					if (hasGlossy)
						l = finalGathering ? L_SurfaceFinalGathering<true>(bRec, -r.dir(), r_i, r2, rng, DIRECT, g_NumPhotonEmittedSurface2) : L_Surface<true>(bRec, -r.dir(), r_i, r2.getMat(), g_NumPhotonEmittedSurface2);
					else l = finalGathering ? L_SurfaceFinalGathering<false>(bRec, -r.dir(), r_i, r2, rng, DIRECT, g_NumPhotonEmittedSurface2) : L_Surface<false>(bRec, -r.dir(), r_i, r2.getMat(), g_NumPhotonEmittedSurface2);
				}
				L += throughput * (hasGlossy ? 0.5f : 1) * l;				
				if (!hasSpecGlossy)
					break;
			}
			if (hasSpecGlossy)
			{
				bRec.sampledType = 0;
				bRec.typeMask = EDelta | EGlossy;
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (!bRec.sampledType)
					break;
				throughput = throughput * t_f * (hasGlossy ? 0.5f : 1);
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
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume((float)g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), Tr);
			L += Tr * throughput * g_SceneData.EvalEnvironment(r);
		}
		img.Add(screenPos.x, screenPos.y, L);
	}
	g_SamplerData(rng);
}

void PPPMTracer::DebugInternal(Image* I, const Vec2i& pixel)
{
	m_sSurfaceMap.Synchronize();
	if (m_sSurfaceMapCaustic)
		m_sSurfaceMapCaustic->Synchronize();
	m_pVolumeEstimator->Synchronize();

	auto ray = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	auto res = traceRay(ray);

	if (g_SceneData.m_sVolume.HasVolumes())
	{
		Spectrum Tr, L;
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			L = ((BeamGrid*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), Tr);
		else if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			L = ((PointStorage*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), Tr);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			L = ((BeamBeamGrid*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), Tr);
	}

	if (res.hasHit())
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		res.getBsdfSample(ray, bRec, ETransportMode::EImportance);

		k_AdaptiveStruct A(r_min, r_max, *m_pAdpBuffer, w, m_uPassesDone);
		k_AdaptiveEntry ent = A(pixel.x, pixel.y);
	}

	/*
	//output some beam specific quantity in a simple csv format along the ray
	float vol_2_rad = getCurrentRadius(2);
	auto& buf = ((BeamGrid*)m_pVolumeEstimator)->m_sStorage;
	std::vector<Vec2f> data;
	for (size_t i = 0; i < min(buf.getNumStoredEntries(), buf.getNumEntries()); i++)
	{
		auto& val = buf(i);
		auto ph_pos = val.getPos(((BeamGrid*)m_pVolumeEstimator)->m_sStorage.getHashGrid(), Vec3u());
		float l1 = dot(ph_pos - ray.ori(), ray.dir());
		float isectRad = distance(ph_pos, ray(l1));
		if (l1 / res.m_fDist > 1 || l1 / res.m_fDist < 0 || isectRad >= 2 * vol_2_rad)
			continue;
		//float k = k_tr<2>(vol_2_rad, isectRad);
		auto l = val.getL() / m_uPhotonEmittedPassVolume;
		auto Tr = g_SceneData.evalTransmittance(ray.ori(), ray(l1));
		PhaseFunctionSamplingRecord pRec(-ray.dir(), val.getWi());
		float k = Spectrum(l * g_SceneData.m_sVolume.p(ray(l1), pRec) * Tr).getLuminance();
		auto e_l = ray(l1) - ph_pos, ur = Vec3f(1, 0, 0)*vol_2_rad, vr = Vec3f(0, 1, 0)*vol_2_rad, wr = Vec3f(0, 0, 1)*vol_2_rad;
		float k_rd = k_tr(vol_2_rad, e_l);
		float laplu = k_tr(vol_2_rad, e_l + ur) + k_tr(vol_2_rad, e_l - ur) - 2 * k_rd,
			laplv = k_tr(vol_2_rad, e_l + vr) + k_tr(vol_2_rad, e_l - vr) - 2 * k_rd,
			laplw = k_tr(vol_2_rad, e_l + wr) + k_tr(vol_2_rad, e_l - wr) - 2 * k_rd;
		//k = k / (vol_2_rad * vol_2_rad * vol_2_rad) * (laplu + laplv + laplw);
		data.push_back(Vec2f(l1 / res.m_fDist, k));
	}
	std::ofstream myfile;
	static int CC = 0;
	myfile.open(format("vol-den-%d-%d-%d.txt", pixel.x, pixel.y, CC++));
	for (auto& v : data)
		myfile << v.x << ", " << v.y << "\n";
	myfile.close();*/
}

void PPPMTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	float radius2 = getCurrentRadius(2, true);

	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	if (m_sSurfaceMapCaustic)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMapCaustic, m_sSurfaceMapCaustic, sizeof(*m_sSurfaceMapCaustic)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedSurface2, &m_uPhotonEmittedPassSurface, sizeof(m_uPhotonEmittedPassSurface)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedVolume2, &m_uPhotonEmittedPassVolume, sizeof(m_uPhotonEmittedPassVolume)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	auto radiusType = m_sParameters.getValue(KEY_RadiiComputationType());
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	if(radiusType != PPM_Radius_Type::Constant)
		m_pAdpBuffer->StartBlock(x, y);
	k_AdaptiveStruct A(r_min, r_max, *m_pAdpBuffer, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	BlockSampleImage img = m_pBlockSampler->getBlockImage();

	if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, radiusType, finalGathering, m_debugScaleVal);
	else if(dynamic_cast<PointStorage*>(m_pVolumeEstimator))
		k_EyePass<PointStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, radiusType, finalGathering, m_debugScaleVal);
	else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamBeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, radiusType, finalGathering, m_debugScaleVal);

	ThrowCudaErrors(cudaThreadSynchronize());
	if (radiusType != PPM_Radius_Type::Constant)
		m_pAdpBuffer->EndBlock();
}

CUDA_DEVICE int g_MaxRad, g_MinRad;
CUDA_FUNC_IN int floatToOrderedInt(float floatVal) {
	int intVal = float_as_int_(floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}
CUDA_FUNC_IN float orderedIntToFloat(int intVal) {
	return int_as_float_((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__global__ void k_PerPixelRadiusEst(Vec2i off, int w, int h, float r_max, float r_1, k_AdaptiveStruct adpt, int k_toFind)
{
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		auto& e = adpt(pixel.x, pixel.y);
		//adaptive progressive intit
		e.Sum_psi = e.Sum_psi2 = e.Sum_E_DI = e.Sum_E_DI2 = e.Sum_pl = e.Sum_DI = 0.0f;
		e.r_std = r_1;

		//initial per pixel rad estimate
		auto rng = g_SamplerData();
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		NormalizedT<Ray> r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
		TraceResult r2 = traceRay(r);
		if (r2.hasHit())
		{
			const float search_rad = r_1;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);
			auto f_t = bRec.dg.sys.t * search_rad, f_s = bRec.dg.sys.s * search_rad;
			Vec3f a = -1.0f * f_t - f_s, b = f_t - f_s, c = -1.0f * f_t + f_s, d = f_t + f_s;
			Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
			int k_found = 0;
#ifdef ISCUDA
			g_SurfMap->ForAll(low, high, [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
			{
				float dist2 = distanceSquared(ph.getPos(g_SurfMap->getHashGrid(), cell_idx), bRec.dg.P);
				if (dist2 < search_rad * search_rad && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
					k_found++;
			});
#endif
			float density = max(k_found, 1) / (PI * search_rad * search_rad);
			e.r_std = math::sqrt(k_toFind / (PI * density));
		}
		else e.r_std = r_1;
		atomicMin(&g_MinRad, floatToOrderedInt(e.r_std));
		atomicMax(&g_MaxRad, floatToOrderedInt(e.r_std));
		g_SamplerData(rng);
	}
}

void PPPMTracer::doPerPixelRadiusEstimation()
{
	int a = floatToOrderedInt(FLT_MAX), b = floatToOrderedInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxRad, &b, sizeof(b)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MinRad, &a, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	float k_Intial = m_sParameters.getValue(KEY_kNN_Neighboor_Num());
	
	IterateAllBlocks(w, h, [&](int x, int y, int, int)
	{
		m_pAdpBuffer->StartBlock(x, y);
		auto A = k_AdaptiveStruct(r_min, r_max, *m_pAdpBuffer, w, m_uPassesDone);//keeps a copy of m_pAdpBuffer!
		k_PerPixelRadiusEst << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(Vec2i(x, y), w, h, r_max * 0.1f, m_fInitialRadiusSurf, A, k_Intial);
		m_pAdpBuffer->EndBlock();
	});

	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
}

}
