#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Engine/Light.h>
#include <Engine/SpatialGridTraversal.h>

#define LOOKUP_NORMAL_THRESH 0.5f

namespace CudaTracerLib {

struct ModelInitializer
{
	float vals[NUM_VOL_MODEL_BINS];
	float nums[NUM_VOL_MODEL_BINS];

	CUDA_FUNC_IN ModelInitializer()
	{
		for (int i = 0; i < NUM_VOL_MODEL_BINS; i++)
			vals[i] = nums[i] = 0.0f;
	}

	CUDA_FUNC_IN void add(float t, float val)
	{
		int n = math::clamp((int)(t * NUM_VOL_MODEL_BINS), 0, NUM_VOL_MODEL_BINS - 1);
		vals[n] += val;
		++nums[n];
	}

	CUDA_FUNC_IN VolumeModel ToModel(float r_std) const
	{
		return VolumeModel([&](int i)
		{
			float r = nums[i] ? vals[i] / nums[i] : r_std;
			return APPM_QueryPointData<3, 3>(0, 0, DerivativeCollection<3>(), 0, 0, 0, r);
		});
	}
};

template<bool USE_GLOBAL> Spectrum PointStorage::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, VolumeModel& model, PPM_Radius_Type radType, Spectrum& Tr)
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

void PointStorage::Compute_kNN_radii(float numEmitted, float rad, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, VolumeModel& model)
{
	ModelInitializer mInit;
	float a, b;
	if (!m_sStorage.getHashGrid().getAABB().Intersect(r, &a, &b))
		return;//that would be dumb
	float minT = a = math::clamp(a, tmin, tmax);
	b = math::clamp(b, tmin, tmax);
	float d = 2.0f * rad;
	while (a < b)
	{
		float t = a + d / 2.0f;
		Vec3f x = r(t);
		float density = 0;
		m_sStorage.ForAll(x - Vec3f(rad), x + Vec3f(rad), [&](const Vec3u& cell_idx, unsigned int p_idx, const volPhoton& ph)
		{
			Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_idx);
			float dist2 = distanceSquared(ph_pos, x);
			if (dist2 <= math::sqr(rad))
				density += Kernel::k<3>(math::sqrt(dist2), rad);
		});
		auto t_m = model_t(t, tmin, tmax);
		if(density > 0.0f)
			mInit.add(t_m, math::pow(kToFind / (4.0f / 3.0f * PI * density), 1.0f / 3.0f));
		a += d;
	}
	model = mInit.ToModel(rad);
}

template<bool USE_GLOBAL> Spectrum BeamGrid::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, VolumeModel& model, PPM_Radius_Type radType, Spectrum& Tr)
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

void BeamGrid::Compute_kNN_radii(float numEmitted, float rad, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, VolumeModel& model)
{
	ModelInitializer mInit;
	float density = 0;
	TraverseGridBeamExt(r, tmin, tmax, m_sStorage,
		[&](const Vec3u& cell_pos, float rayT, float cellEndT)
	{
		density = 0;
		return rad;
	},
		[&](const Vec3u& cell_idx, unsigned int element_idx, const volPhoton& element, float& distAlongRay)
	{
		return CudaTracerLib::sqrDistanceToRay(r, element.getPos(m_sStorage.getHashGrid(), cell_idx), distAlongRay);
	},
		[&](float rayT, float cellEndT, float minT, float maxT, const Vec3u& cell_idx, unsigned int element_idx, const volPhoton& element, float distAlongRay, float distRay2)
	{
		if (distRay2 < math::sqr(m_fCurrentRadiusVol))
		{
			auto ph_pos = element.getPos(m_sStorage.getHashGrid(), cell_idx);
			auto dist2 = distanceSquared(ph_pos, r(distAlongRay));
			if (dist2 <= math::sqr(rad))
				density += Kernel::k<3>(math::sqrt(dist2), rad);
		}
	},
		[&](float rayT, float cellEndT, float minT, float maxT, const Vec3u& cell_idx)
	{
		auto t_m = model_t((rayT + cellEndT) / 2.0f, tmin, tmax);
		if (density > 0.0f)
			mInit.add(t_m, math::sqrt(kToFind / (PI * density)));
	}
	);
	model = mInit.ToModel(rad);
}

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum beam_beam_L(const VolHelper<USE_GLOBAL>& vol, const Beam& B, const NormalizedT<Ray>& r, float radius, float beamIsectDist, float queryIsectDist, float beamBeamDistance, float m_uNumEmitted, float sinTheta, float tmin)
{
	Spectrum photon_tau = vol.tau(Ray(B.getPos(), B.getDir()), 0, beamIsectDist);
	Spectrum camera_tau = vol.tau(r, tmin, queryIsectDist);
	Spectrum camera_sc = vol.sigma_s(r(queryIsectDist), r.dir());
	PhaseFunctionSamplingRecord pRec(-r.dir(), B.getDir());
	float p = vol.p(r(queryIsectDist), pRec);
	return B.getL() / m_uNumEmitted * (-photon_tau).exp() * camera_sc * Kernel::k<1>(beamBeamDistance, radius) / sinTheta * (-camera_tau).exp() * 0.5f;//this is not correct; the phase function is missing and the 0.5 is arbirtary scaling
}

struct BeamIntersectionData
{
	float beamBeamDistance;
	float sinTheta;
	float beamIsectDist;
	Beam B;
};
template<bool USE_GLOBAL> Spectrum BeamBeamGrid::L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, VolumeModel& model, PPM_Radius_Type radType, Spectrum& Tr)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	TraverseGridBeam(r, tmin, tmax, m_sStorage,
		[&](const Vec3u& cell_pos, float rayT, float cellEndT)
	{
		return m_fCurrentRadiusVol;
	},
		[&](const Vec3u& cell_idx, unsigned int ref_element_idx, int beam_idx, float& distAlongRay)
	{
		BeamIntersectionData dat;
		dat.B = this->m_sBeamStorage[beam_idx];
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, dat.B.getPos(), dat.B.getDir(), 0, dat.B.t, math::sqr(m_fCurrentRadiusVol), dat.beamBeamDistance, dat.sinTheta, distAlongRay, dat.beamIsectDist))
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

void BeamBeamGrid::Compute_kNN_radii(float numEmitted, float rad, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, VolumeModel& model)
{
	ModelInitializer mInit;
	float density = 0;
	TraverseGridBeamExt(r, tmin, tmax, m_sStorage,
		[&](const Vec3u& cell_pos, float rayT, float cellEndT)
	{
		density = 0;
		return rad;
	},
		[&](const Vec3u& cell_idx, unsigned int ref_element_idx, int beam_idx, float& distAlongRay)
	{
		BeamIntersectionData dat;
		dat.B = this->m_sBeamStorage[beam_idx];
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, dat.B.getPos(), dat.B.getDir(), 0, dat.B.t, math::sqr(rad), dat.beamBeamDistance, dat.sinTheta, distAlongRay, dat.beamIsectDist))
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
		density += Kernel::k<3>(dat.beamBeamDistance, rad);
	},
		[&](float rayT, float cellEndT, float minT, float maxT, const Vec3u& cell_idx)
	{
		auto t_m = model_t((rayT + cellEndT) / 2.0f, tmin, tmax);
		if (density > 0.0f)
			mInit.add(t_m, kToFind / (2.0f * density));
	}
	);
	model = mInit.ToModel(rad);
}

CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMap;
CUDA_CONST CudaStaticWrapper<SurfaceMapT> g_SurfMapCaustic;
CUDA_CONST unsigned int g_NumPhotonEmittedSurface2, g_NumPhotonEmittedVolume2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax3(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid))];

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float r, const Material& mat, unsigned int numPhotonsEmitted, SurfaceMapT* map = 0)
{
	if (!map) map = &g_SurfMap.As();
	bool hasGlossy = mat.bsdf.hasComponent(EGlossy);
	Spectrum Lp = Spectrum(0.0f);
	Vec3f a = r*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, c = r*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	map->ForAll(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
	{
		float dist2 = distanceSquared(ph.getPos(map->getHashGrid(), cell_idx), bRec.dg.P);
		Vec3f photonNormal = ph.getNormal();
		float wiDotGeoN = absdot(photonNormal, wi);
		if (dist2 < r * r && dot(photonNormal, bRec.dg.sys.n) > LOOKUP_NORMAL_THRESH && wiDotGeoN > 1e-2f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			float cor_fac = math::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
			float ke = Kernel::k<2>(math::sqrt(dist2), r);
			Spectrum l = ph.getL();
			if(hasGlossy)
				l *= mat.bsdf.f(bRec) / Frame::cosTheta(bRec.wo);//bsdf.f returns f * cos(thetha)
			Lp += ke * l;// * cor_fac;
		}
	});
	if(!hasGlossy)
	{
		auto wi_l = bRec.wi;
		bRec.wo = bRec.wi = NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f);
		Lp *= mat.bsdf.f(bRec);
		bRec.wi = wi_l;
	}
	return Lp / (float)numPhotonsEmitted;
}

CUDA_FUNC_IN Spectrum L_SurfaceFinalGathering(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float rad, TraceResult& r2, Sampler& rng, bool DIRECT, unsigned int numPhotonsEmitted)
{
	Spectrum LCaustic = L_Surface(bRec, wi, rad, r2.getMat(), numPhotonsEmitted, &g_SurfMapCaustic.As());
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
			L += f * L_Surface(bRec2, -r.dir(), rad, r3.getMat(), numPhotonsEmitted);
			if (DIRECT)
				L += f * UniformSampleOneLight(bRec2, r3.getMat(), rng);
			//do not account for emission because such photons will be in the specular photon map, hopefulyl with lower variance
		}
	}
	bRec.typeMask = ETypeCombinations::EAll;
	return L / (float)N + LCaustic;
}

CUDA_DEVICE float g_Sum_Photons_Touched;
CUDA_DEVICE float g_Sum_Photons_Touched2;
CUDA_DEVICE unsigned int g_NumRadEstimates;
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
	auto ent = A(x, y).m_surfaceData;
	float r = iteration <= 1 ? A.r_max / 5.0f : ent.compute_r<2>(iteration - 1, numPhotonsEmittedSurf, numPhotonsEmittedSurf * (iteration - 1), [](const DerivativeCollection<1>& gr) {return Lapl(gr); }),
		  rd = iteration <= 1 ? a_rSurfaceUNUSED : ent.compute_rd(iteration - 1, numPhotonsEmittedSurf, numPhotonsEmittedSurf * (iteration - 1));

	r = math::clamp(r != -1.0f ? r : a_rSurfaceUNUSED, A.r_min, A.r_max);
	rd = math::clamp(rd != -1.0f ? rd : a_rSurfaceUNUSED, A.r_min, A.r_max);
	rd = a_rSurfaceUNUSED;
	//r = rd = a_rSurfaceUNUSED;
	Vec3f ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	float r_max = max(2 * rd, r);
	Vec3f a = r_max*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r_max*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, 
		  c = r_max*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r_max*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;

	Spectrum Lp = 0.0f;
	float Sum_psi = 0, Sum_psi2 = 0, n_psi = 0, Sum_DI = 0, numPhotonsTouched = 0;
	surfMap.ForAll(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)//CAP = DANGEROUS !!!
	{
		Vec3f ph_pos = ph.getPos(surfMap.getHashGrid(), cell_idx);
		float dist2 = distanceSquared(ph_pos, bRec.dg.P);
		Vec3f photonNormal = ph.getNormal();
		float wiDotGeoN = absdot(photonNormal, wi);
		if (dot(photonNormal, bRec.dg.sys.n) > LOOKUP_NORMAL_THRESH && wiDotGeoN > 1e-2f)
		{
			numPhotonsTouched += dist2 <= math::sqr(r_max);
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			auto bsdfFactor = hasGlossy ? mat->bsdf.f(bRec) / Frame::cosTheta(bRec.wo) : bsdf_diffuse;
			float psi = Spectrum(importance * bsdfFactor * ph.getL()).getLuminance();
			const Vec3f e_l = bRec.dg.P - ph_pos;
			float k_rd = Kernel::k<2>(e_l, rd);
			float laplu = Kernel::k<2>(e_l + ur, rd) + Kernel::k<2>(e_l - ur, rd) - 2 * k_rd,
				  laplv = Kernel::k<2>(e_l + vr, rd) + Kernel::k<2>(e_l - vr, rd) - 2 * k_rd,
				  lapl = psi / (rd * rd) * (laplu + laplv);
			Sum_DI += lapl;

			if (dist2 < math::sqr(r))
			{
				float kri = Kernel::k<2>(math::sqrt(dist2), r);
				Lp += kri * ph.getL() / float(numPhotonsEmittedSurf) * bsdfFactor;
				Sum_psi += psi;
				Sum_psi2 += math::sqr(psi);
				ent.Sum_pl += kri;
				n_psi++;
			}
		}
	});
#ifdef ISCUDA
	atomicAdd(&g_Sum_Photons_Touched, numPhotonsTouched);
	atomicAdd(&g_Sum_Photons_Touched2, math::sqr(numPhotonsTouched));
	atomicInc(&g_NumRadEstimates, 0xffffffff);
#endif
	ent.Sum_DI.df_di[0] += Sum_DI / numPhotonsEmittedSurf;
	auto E_DI = Lapl(ent.Sum_DI) / (float)iteration;
	ent.Sum_E_DI += E_DI;
	ent.Sum_E_DI2 += math::sqr(E_DI);
	if(n_psi != 0)
	{
		ent.Sum_psi += Sum_psi / n_psi;
		ent.Sum_psi2 += Sum_psi2 / n_psi;
	}

#ifdef ISCUDA
	A(x, y).m_surfaceData = ent;
#endif
	return Lp;
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, k_AdaptiveStruct a_AdpEntries, BlockSampleImage img, bool DIRECT, PPM_Radius_Type Radius_Type, bool finalGathering, float debugScaleVal)
{
	auto rng = g_SamplerData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	auto adp_ent = a_AdpEntries(pixel.x, pixel.y);
	if (pixel.x < w && pixel.y < h)
	{
		Vec2f screenPos = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		NormalizedT<Ray> r, rX, rY;
		Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());
		TraceResult r2;
		r2.Init();
		int depth = -1;
		Spectrum L(0.0f);
		bool deltaChain = true;
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
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), adp_ent.m_volumeModel, Radius_Type, Tr);
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
						Spectrum Li = ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, NormalizedT<Ray>(bRec.dg.P, dRec.d), tmin, tmax, VolHelper<true>(), adp_ent.m_volumeModel, Radius_Type, Tr);
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
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), adp_ent.m_volumeModel, Radius_Type, Tr);
				//throughput = throughput * Tr;
				break;
			}
			bool hasDiffuse = r2.getMat().bsdf.hasComponent(EDiffuse),
				hasSpec = r2.getMat().bsdf.hasComponent(EDelta),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasDiffuse)
			{
				Spectrum L_r;//reflected radiance computed by querying photon map
				if (Radius_Type == PPM_Radius_Type::Adaptive && deltaChain)
					L_r = L_Surface(bRec, -r.dir(), a_rSurface, &r2.getMat(), a_AdpEntries, pixel.x, pixel.y, throughput, a_PassIndex, img, g_SurfMap, g_NumPhotonEmittedSurface2, debugScaleVal);
				else
				{
					float rad = math::clamp(getCurrentRadius(adp_ent.m_surfaceData.r_std, a_PassIndex, 2), a_AdpEntries.r_min, a_AdpEntries.r_max);
					float r_i = Radius_Type == PPM_Radius_Type::kNN ? rad : a_rSurface;
					L_r = finalGathering ? L_SurfaceFinalGathering(bRec, -r.dir(), r_i, r2, rng, DIRECT, g_NumPhotonEmittedSurface2) : 
										   L_Surface(bRec, -r.dir(), r_i, r2.getMat(), g_NumPhotonEmittedSurface2);
				}
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
				deltaChain &= (bRec.sampledType & EGlossy) == 0;
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
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume((float)g_NumPhotonEmittedVolume2, r, tmin, tmax, VolHelper<true>(), adp_ent.m_volumeModel, Radius_Type, Tr);
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

	auto radiusType = m_sParameters.getValue(KEY_RadiiComputationType());
	auto ray = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	auto res = traceRay(ray);
	k_AdaptiveStruct A(r_min, r_max, *m_pAdpBuffer, w, m_uPassesDone);
	auto pixelInfo = A(pixel.x, pixel.y);

	float tmin, tmax;
	if (g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(ray, 0.0f, FLT_MAX, &tmin, &tmax))
	{
		float k_Intial = m_sParameters.getValue(KEY_kNN_Neighboor_Num_Vol());
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			((BeamGrid*)m_pVolumeEstimator)->Compute_kNN_radii((float)m_uPhotonEmittedPassVolume, m_fInitialRadiusVol, k_Intial, ray, tmin, tmax, pixelInfo.m_volumeModel);
		if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			((PointStorage*)m_pVolumeEstimator)->Compute_kNN_radii((float)m_uPhotonEmittedPassVolume, m_fInitialRadiusVol, k_Intial, ray, tmin, tmax, pixelInfo.m_volumeModel);
		if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			((BeamBeamGrid*)m_pVolumeEstimator)->Compute_kNN_radii((float)m_uPhotonEmittedPassVolume, m_fInitialRadiusVol, k_Intial, ray, tmin, tmax, pixelInfo.m_volumeModel);

		Spectrum Tr, L;
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			L = ((BeamGrid*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), pixelInfo.m_volumeModel, radiusType, Tr);
		else if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			L = ((PointStorage*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), pixelInfo.m_volumeModel, radiusType, Tr);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			L = ((BeamBeamGrid*)m_pVolumeEstimator)->L_Volume((float)m_uPhotonEmittedPassVolume, ray, 0.0f, res.m_fDist, VolHelper<true>(), pixelInfo.m_volumeModel, radiusType, Tr);
	}

	static float Sum_lapl_N = 0;
	static float Sum_lapl_N2 = 0;
	static float sum_lapl = 0;
	if (res.hasHit())
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		res.getBsdfSample(ray, bRec, ETransportMode::EImportance);

		/*int iteration = m_uPassesDone;
		float lapl_it_sum = 0;
		auto rd = iteration <= 1 ? getCurrentRadius(2, true) : ent.compute_rd(iteration - 1, m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassSurface * (iteration - 1));
		rd = getCurrentRadius(2, true);
		auto ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
		m_sSurfaceMap.ForAll(Vec3u(0), Vec3u(m_sSurfaceMap.getHashGrid().m_gridDim), [&](const Vec3u& cell_idx, unsigned int e_idx, const PPPMPhoton& p)
		{
			Vec3f photonNormal = p.getNormal();
			float wiDotGeoN = absdot(photonNormal, -ray.dir());
			if (dot(p.getNormal(), bRec.dg.sys.n) < 0.9f && wiDotGeoN > 1e-2f)
				return;
			bRec.wo = bRec.dg.toLocal(p.getWi());
			float psi = Spectrum(p.getL()).getLuminance();
			auto ph_pos = p.getPos(m_sSurfaceMap.getHashGrid(), cell_idx);
			const Vec3f e_l = bRec.dg.P - ph_pos;
			float k_rd = Kernel::k<2>(e_l, rd);
			float laplu = Kernel::k<2>(e_l + ur, rd) + Kernel::k<2>(e_l - ur, rd) - 2 * k_rd,
				laplv = Kernel::k<2>(e_l + vr, rd) + Kernel::k<2>(e_l - vr, rd) - 2 * k_rd,
				lapl = psi / (rd * rd) * (laplu + laplv);
			lapl_it_sum += lapl;
		});
		float lapl_i = lapl_it_sum / m_uPhotonEmittedPassSurface;
		sum_lapl += lapl_i;
		float lapl_N = sum_lapl / iteration;
		Sum_lapl_N += lapl_N;
		Sum_lapl_N2 += lapl_N * lapl_N;*/

		L_Surface(bRec, -ray.dir(), getCurrentRadius(2), &res.getMat(), A, pixel.x, pixel.y, Spectrum(1.0f), m_uPassesDone, m_pBlockSampler->getBlockImage(), m_sSurfaceMap, m_uPhotonEmittedPassSurface, m_debugScaleVal);
		//m_adpBuffer->setOnCPU();
		//m_adpBuffer->Synchronize();
	}
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
	ZeroSymbol(g_Sum_Photons_Touched);
	ZeroSymbol(g_Sum_Photons_Touched2);
	ZeroSymbol(g_NumRadEstimates);

	auto radiusType = m_sParameters.getValue(KEY_RadiiComputationType());
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	//not starting the block will lead to (correct) warnings due to no a
	m_pAdpBuffer->StartBlock(x, y, radiusType != PPM_Radius_Type::Constant);
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
template<typename VolEstimator> __global__ void k_PerPixelRadiusEst(Vec2i off, int w, int h, float r_max, float r_surface, float r_volume, float numEmitted, k_AdaptiveStruct adpt, float k_toFindSurf, float k_toFindVol)
{
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	auto& pixleInfo = adpt(pixel.x, pixel.y);
	if (pixel.x < w && pixel.y < h)
	{
		NormalizedT<Ray> r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);

		//adaptive progressive intit
		pixleInfo.Initialize(r_surface, r.dir());

		//initial per pixel rad estimate
		auto rng = g_SamplerData();
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		TraceResult r2 = traceRay(r);
		if (r2.hasHit())
		{
			//compute surface query radius based on kNN search
			const float search_rad_surf = r_surface;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);
			auto f_t = bRec.dg.sys.t * search_rad_surf, f_s = bRec.dg.sys.s * search_rad_surf;
			Vec3f a = -1.0f * f_t - f_s, b = f_t - f_s, c = -1.0f * f_t + f_s, d = f_t + f_s;
			Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
			float density = 0;
#ifdef ISCUDA
			g_SurfMap->ForAll(low, high, [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
			{
				float dist2 = distanceSquared(ph.getPos(g_SurfMap->getHashGrid(), cell_idx), bRec.dg.P);
				if (dist2 < search_rad_surf * search_rad_surf && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
					density += Kernel::k<2>(math::sqrt(dist2), search_rad_surf);
			});
#endif
			pixleInfo.m_surfaceData.r_std = math::sqrt(k_toFindSurf / (PI * density));

			//compute volume query radii
			float tmin, tmax;
			if(g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(r, 0.0f, FLT_MAX, &tmin, &tmax))
			{
				((VolEstimator*)g_VolEstimator2)->Compute_kNN_radii(numEmitted, r_volume, k_toFindVol, r, tmin, tmax, pixleInfo.m_volumeModel);
			}

			//traverse on specular manifold until a subsurface scattering object is hit, if so compute kNN based volumetric radii
			int depth = 0;
			while(r2.hasHit() && depth++ < 3)
			{
				const VolumeRegion* vReg;
				if (r2.getMat().GetBSSRDF(dg, &vReg))
				{
					Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());//materials with subsurface scattering should use some (nearly) delta bsdf
					bRec.wo.z *= -1.0f;
					NormalizedT<Ray> rTrans = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
					TraceResult r3 = traceRay(rTrans);
					((VolEstimator*)g_VolEstimator2)->Compute_kNN_radii(numEmitted, r_volume * 10, k_toFindVol, rTrans, 0.0f, r3.m_fDist, pixleInfo.m_volumeModel);

					break;//only compute radii for first object
				}
				else
				{
					r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
					r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
					r2 = traceRay(r);
				}
			}

			
		}
		atomicMin(&g_MinRad, floatToOrderedInt(pixleInfo.m_surfaceData.r_std));
		atomicMax(&g_MaxRad, floatToOrderedInt(pixleInfo.m_surfaceData.r_std));
		g_SamplerData(rng);
	}
}

void PPPMTracer::doPerPixelRadiusEstimation()
{
	int a = floatToOrderedInt(FLT_MAX), b = floatToOrderedInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxRad, &b, sizeof(b)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MinRad, &a, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));
	float k_toFindSurf = m_sParameters.getValue(KEY_kNN_Neighboor_Num_Surf()),
		  k_toFindVol = m_sParameters.getValue(KEY_kNN_Neighboor_Num_Vol());
	
	IterateAllBlocks(w, h, [&](int x, int y, int, int)
	{
		m_pAdpBuffer->StartBlock(x, y);
		auto A = k_AdaptiveStruct(r_min, r_max, *m_pAdpBuffer, w, m_uPassesDone);//keeps a copy of m_pAdpBuffer!
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			k_PerPixelRadiusEst<BeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(Vec2i(x, y), w, h, r_max * 0.1f, m_fInitialRadiusSurf, m_fInitialRadiusVol, (float)m_uPhotonEmittedPassVolume, A, k_toFindSurf, k_toFindVol);
		else if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			k_PerPixelRadiusEst<PointStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(Vec2i(x, y), w, h, r_max * 0.1f, m_fInitialRadiusSurf, m_fInitialRadiusVol, (float)m_uPhotonEmittedPassVolume, A, k_toFindSurf, k_toFindVol);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			k_PerPixelRadiusEst<BeamBeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(Vec2i(x, y), w, h, r_max * 0.1f, m_fInitialRadiusSurf, m_fInitialRadiusVol, (float)m_uPhotonEmittedPassVolume, A, k_toFindSurf, k_toFindVol);
		m_pAdpBuffer->EndBlock();
	});

	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
}

}
