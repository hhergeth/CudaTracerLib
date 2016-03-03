#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Engine/Light.h>

namespace CudaTracerLib {
CUDA_FUNC_IN bool sphere_line_intersection(const Vec3f& p, float radSqr, const Ray& r, float& t_min, float& t_max)
{
	auto d = r.dir();
	Vec3f oc = r.ori() - p;
	float f = dot(d, oc);
	float w = f * f - oc.lenSqr() + radSqr;
	if (w < 0)
		return false;
	t_min = -f - math::sqrt(w);
	t_max = -f + math::sqrt(w);
	return true;
}

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum PointStorage::L_Volume(float NumEmitted, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	float Vs = 1.0f / (4.0f / 3.0f * PI * m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_fCurrentRadiusVol * NumEmitted);
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
			if (distanceSquared(ph_pos, x) < m_fCurrentRadiusVol * m_fCurrentRadiusVol)
			{
				float p = vol.p(x, -r.dir(), ph.getWi(), rng);
				L_i += p * ph.getL() * Vs;
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

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum BeamGrid::L_Volume(float NumEmitted, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	Spectrum L_n = Spectrum(0.0f);
	TraverseGrid(r, m_sStorage.getHashGrid(), tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
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
				float p = vol.p(ph_pos, r.dir(), ph.getWi(), rng);
				L_n += p * ph.getL() * tauToPhoton / (PI * NumEmitted * ph_rad2);
			}
			/*float t1, t2;
			if (sphere_line_intersection(ph_pos, ph_rad2, r, t1, t2))
			{
				float Vs = 1.0f / (4.0f / 3.0f * PI * ph_rad2 * ph_rad1 * NumEmitted);
				float p = vol.p(r((t1 + t2) / 2), r.dir(), ph.getWi(), rng);
				auto o_t = vol.sigma_t(r((t1 + t2) / 2), r.dir());
				auto ta = ((-o_t * t1).exp() - (-o_t * t2).exp()) / o_t;
				//auto f_ta = (-vol.tau(r, tmin, t1)).exp(), f_tb = (-vol.tau(r, tmin, t2)).exp();
				//auto ta = (t2 - t1) * (f_ta + 0.5f * (f_tb - f_ta));
				L_n += p * ph.getL() * Vs * ta;
			}*/
		});
		Tau += vol.tau(r, rayT, cellEndT);
		float localDist = cellEndT - rayT;
		L_n += vol.Lve(r(rayT + localDist / 2), -r.dir()) * localDist;
	});
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_CONST SurfaceMapT g_SurfMap;
CUDA_CONST SurfaceMapT g_SurfMapCaustic;
CUDA_CONST unsigned int g_NumPhotonEmittedSurface2, g_NumPhotonEmittedVolume2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax4(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid), sizeof(BeamBVHStorage))];

template<bool USE_GLOBAL> CUDA_ONLY_FUNC Spectrum beam_beam_L(const VolHelper<USE_GLOBAL>& vol, CudaRNG& rng, const Beam& B, const NormalizedT<Ray>& r, float radius, float beamIsectDist, float queryIsectDist, float beamBeamDistance, int m_uNumEmitted, float sinTheta, float tmin)
{
	Spectrum photon_tau = vol.tau(Ray(B.getPos(), B.getDir()), 0, beamIsectDist);
	Spectrum camera_tau = vol.tau(r, tmin, queryIsectDist);
	Spectrum camera_sc = vol.sigma_s(r(queryIsectDist), r.dir());
	float p = vol.p(r(queryIsectDist), r.dir(), B.getDir(), rng);
	//return B.getL() / float(m_uNumEmitted) * camera_sc * (-photon_tau).exp() * (-camera_tau).exp() * p * (1 - beamBeamDistance * beamBeamDistance / (radius * radius)) * 3 / (4 * radius*sinTheta);
	return camera_sc * p * B.getL() / m_uNumEmitted * (-photon_tau).exp() * (-camera_tau).exp() / sinTheta * k_tr<1>(radius, beamBeamDistance);
}

template<bool USE_GLOBAL> CUDA_ONLY_FUNC Spectrum BeamBeamGrid::L_Volume(float NumEmitted, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	/*TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sStorage.ForAll(cell_pos, [&](unsigned int, int beam_idx)
		{
			const Beam& B = m_pDeviceBeams[beam_idx];
			float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
			if (Beam::testIntersectionBeamBeam(r.origin, r.direction, tmin, tmax, B.pos, B.dir, 0, B.t, radius * radius, beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist)
				 && m_sStorage.hashMap.Transform(B.pos + B.dir * beamIsectDist) == cell_pos)
				L_n += beam_beam_L(vol, rng, B, r, radius, beamIsectDist, queryIsectDist, beamBeamDistance, m_uNumEmitted, sinTheta, tmin);
		});
		float localDist = cellEndT - rayT;
		Spectrum tauD = vol.tau(r, rayT, cellEndT);
		Tau += tauD;
		L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
	});
	Tr = (-Tau).exp();*/
	for (unsigned int i = 0; i < min(m_uBeamIdx, m_uBeamLength); i++)
	{
		const Beam& B = m_pDeviceBeams[i];
		float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, B.getPos(), B.getDir(), 0, B.t, math::sqr(m_fCurrentRadiusVol), beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, rng, B, r, m_fCurrentRadiusVol, beamIsectDist, queryIsectDist, beamBeamDistance, NumEmitted, sinTheta, tmin);
	}
	Tr = (-vol.tau(r, tmin, tmax)).exp();
	return L_n;
}

template<bool USE_GLOBAL> Spectrum BeamBVHStorage::L_Volume(float NumEmitted, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const
{
	Spectrum L_n = Spectrum(0.0f);
	iterateBeams(Ray(r(tmin), r.dir()), tmax - tmin, [&](unsigned int ref_idx)
	{
		const BeamRef& R = m_pDeviceRefs[ref_idx];
		//unsigned int beam_idx = R.getIdx();
		const Beam& B = R.beam;// this->m_pDeviceBeams[beam_idx];
		float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
		if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, B.getPos(), B.getDir(), R.t_min, R.t_max, math::sqr(m_fCurrentRadiusVol), beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, rng, B, r, m_fCurrentRadiusVol, beamIsectDist, queryIsectDist, beamBeamDistance, NumEmitted, sinTheta, tmin);
	});
	Tr = (-vol.tau(r, tmin, tmax)).exp();
	return L_n;
}

template<bool F_IS_GLOSSY> CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, const Vec3f& wi, float r, const Material& mat, SurfaceMapT* map = 0)
{
	if (!map) map = &g_SurfMap;
	Spectrum Lp = Spectrum(0.0f);
	Vec3f a = r*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, c = r*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
#ifdef ISCUDA
	map->ForAll<200>(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
	{
		float dist2 = distanceSquared(ph.getPos(map->getHashGrid(), cell_idx), bRec.dg.P);
		Vec3f photonNormal = ph.getNormal();
		float wiDotGeoN = absdot(photonNormal, wi);
		if (dist2 < r * r && dot(photonNormal, bRec.dg.sys.n) > 0.1f && wiDotGeoN > 1e-2f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			float cor_fac = math::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
			float ke = k_tr<2>(r, math::sqrt(dist2));
			Spectrum l = ph.getL();
			if(F_IS_GLOSSY)
				l *= mat.bsdf.f(bRec) / Frame::cosTheta(bRec.wo);//bsdf.f returns f * cos(thetha)
			Lp += ke * l;// * cor_fac;
		}
	});
	if(!F_IS_GLOSSY)
		Lp *= mat.bsdf.f(bRec) / Frame::cosTheta(bRec.wo);
	return Lp / g_NumPhotonEmittedSurface2;
#else
	return 1.0f;
#endif
}

template<bool F_IS_GLOSSY> CUDA_FUNC_IN Spectrum L_SurfaceFinalGathering(BSDFSamplingRecord& bRec, const Vec3f& wi, float rad, TraceResult& r2, CudaRNG& rng, bool DIRECT)
{
	Spectrum LCaustic = L_Surface<F_IS_GLOSSY>(bRec, wi, rad, r2.getMat(), &g_SurfMapCaustic);
	Spectrum L(0.0f);
	const int N = 3;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec2(dg);//constantly reloading into bRec and using less registers has about the same performance
	for (int i = 0; i < N; i++)
	{
		bRec.typeMask = EGlossy | EDiffuse;
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		NormalizedT<Ray> r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = traceRay(r);
		if (r3.hasHit())
		{
			r3.getBsdfSample(r, bRec2, ETransportMode::ERadiance, &rng);
			bool hasGlossy = r3.getMat().bsdf.hasComponent(EGlossy);
			L += f * (hasGlossy ? L_Surface<true>(bRec2, -r.dir(), rad, r3.getMat()) : L_Surface<false>(bRec2, -r.dir(), rad, r3.getMat()));
			if (DIRECT)
				L += f * UniformSampleOneLight(bRec2, r3.getMat(), rng);
			else L += f * r3.Le(bRec2.dg.P, bRec2.dg.sys, -r.dir());
		}
	}
	bRec.typeMask = ETypeCombinations::EAll;
	return L / N + LCaustic;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const Material* mat, k_AdaptiveStruct& A, int x, int y,
	const Spectrum& importance, int iteration, BlockSampleImage& img)
{
#ifdef ISCUDA
	//ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	//ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);

	//float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
	//float ta = (2.0f * math::sqrt(VAR_Psi)) / (PI * float(g_NumPhotonEmittedSurface2) * E_pl * k_22 * E_I * E_I);
	//float q = math::pow(ta, 1.0f / 6.0f) * math::pow(iteration, -1.0f / 6.0f);
	//float p = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	//ent.r = math::clamp(q, A.r_min, A.r_max);

	//Adaptive Progressive Photon Mapping Implementation
	float NJ = iteration * g_NumPhotonEmittedSurface2, scaleA = (NJ - 1) / NJ, scaleB = 1.0f / NJ;
	float r = a_rSurfaceUNUSED, rd = a_rSurfaceUNUSED;
	k_AdaptiveEntry ent = A(x, y);
	Vec3f ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	float r_max = max(2 * rd, r);
	Vec3f a = r_max*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r_max*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P,
		  c = r_max*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r_max*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	Spectrum Lp = 0.0f;

	g_SurfMap.ForAll<200>(min(a, b, c, d), max(a, b, c, d), [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
	{
		Vec3f ph_pos = ph.getPos(g_SurfMap.getHashGrid(), cell_idx);
		float dist2 = distanceSquared(ph_pos, bRec.dg.P);
		//if (dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			Spectrum bsdfFactor = mat->bsdf.f(bRec);
			float psi = Spectrum(importance * bsdfFactor * ph.getL() / float(g_NumPhotonEmittedSurface2)).getLuminance();//this 1/J has nothing to do with E[X], it is scaling for radiance distribution
			const Vec3f e_l = bRec.dg.P - ph_pos;
			float k_rd = k_tr(rd, e_l);
			float laplu = k_tr(rd, e_l + ur) + k_tr(rd, e_l - ur) - 2 * k_rd,
				  laplv = k_tr(rd, e_l + vr) + k_tr(rd, e_l - vr) - 2 * k_rd,
				  lapl = psi / (rd * rd) * (laplu + laplv);
			ent.DI = ent.DI * scaleA + lapl * scaleB;
			ent.E_DI = ent.E_DI * scaleA + psi * scaleB;
			ent.E_DI2 = ent.E_DI2 * scaleA + psi * psi * scaleB;
			if (dist2 < r * r)
			{
				float kri = k_tr<2>(r, math::sqrt(dist2));
				Lp += kri * ph.getL() / float(g_NumPhotonEmittedSurface2) * bsdfFactor;
				ent.E_psi = ent.E_psi * scaleA + psi * scaleB;
				ent.E_psi2 = ent.E_psi2 * scaleA + psi * psi * scaleB;
				ent.pl += kri;
			}
		}
	});
	//if (x == 488 && y == 654)
	//	printf("Var[Psi] = %5.5e, Var[Lapl] = %5.5e, E[pl] = %5.5e, E[I] = %5.5e, E[Psi] = %5.5e\n", VAR_Psi, VAR_Lapl, E_pl, E_I, ent.psi / NJ);

	Spectrum qs;
	//float t = E_pl / (PPM_MaxRecursion / (PI * 4 * math::sqr(g_SceneData.m_sBox.Size().length() / 2)));
	//float t = (ent.r - A.r_min) / (A.r_max - A.r_min);
	//float t = math::abs(E_I) * 1e-3f / (g_SceneData.m_sBox.Size().length() / 2);
	//float t = ent.DI * 100000000;
	float t = ent.compute_r(iteration, g_NumPhotonEmittedSurface2, g_NumPhotonEmittedSurface2 * iteration) / (A.r_max * 1 - A.r_min);
	//float t = ent.compute_rd(iteration) / (A.r_max - A.r_min) * 1000;
	qs.fromHSL(2.0f / 3.0f * (1 - math::clamp01(t)), 1, 0.5f);//0 -> 1 : Dark Blue -> Light Blue -> Green -> Yellow -> Red
	img.Add(x, y, qs);

	A(x, y) = ent;
	return Lp;
#else
	return 0.0f;
#endif
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, k_AdaptiveStruct a_AdpEntries, BlockSampleImage img, bool DIRECT, bool USE_PerPixelRadius, bool finalGathering)
{
	CudaRNG rng = g_RNGData();
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
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			if (depth == 0)
				dg.computePartials(r, rX, rY);
			if (g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				if (g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
				{
					Spectrum Tr(1.0f);
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rng, r, tmin, tmax, VolHelper<true>(), Tr);
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
						Spectrum Li = ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rng, NormalizedT<Ray>(bRec.dg.P, dRec.d), tmin, tmax, VolHelper<true>(), Tr);
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
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rng, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr);
				//throughput = throughput * Tr;
				break;
			}
			bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
				hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasSmooth)
			{
				float rad = math::clamp(getCurrentRadius(a_AdpEntries(pixel.x, pixel.y).r_std, a_PassIndex, 2), a_AdpEntries.r_min, a_AdpEntries.r_max);
				float r_i = USE_PerPixelRadius ? rad : a_rSurface;
				Spectrum l;
				if (hasGlossy)
					l = finalGathering ? L_SurfaceFinalGathering<true>(bRec, -r.dir(), r_i, r2, rng, DIRECT) : L_Surface<true>(bRec, -r.dir(), r_i, r2.getMat());
				else l = finalGathering ? L_SurfaceFinalGathering<false>(bRec, -r.dir(), r_i, r2, rng, DIRECT) : L_Surface<false>(bRec, -r.dir(), r_i, r2.getMat());
				L += throughput * (hasGlossy ? 0.5f : 1) * l;
				//L += throughput * L_Surface(bRec, a_rSurface, &r2.getMat(), a_AdpEntries, pixel.x, pixel.y, throughput, a_PassIndex, img);
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
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(g_NumPhotonEmittedVolume2, rng, r, tmin, tmax, VolHelper<true>(), Tr);
			L += Tr * throughput * g_SceneData.EvalEnvironment(r);
		}
		img.Add(screenPos.x, screenPos.y, L);
	}
	g_RNGData(rng);
}

void PPPMTracer::Debug(Image* I, const Vec2i& pixel)
{
	
}

void PPPMTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	float radius2 = getCurrentRadius(2, true);

	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMapCaustic, &m_sSurfaceMapCaustic, sizeof(m_sSurfaceMapCaustic)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedSurface2, &m_uPhotonEmittedPassSurface, sizeof(m_uPhotonEmittedPassSurface)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmittedVolume2, &m_uPhotonEmittedPassVolume, sizeof(m_uPhotonEmittedPassVolume)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	bool perPixelRad = m_sParameters.getValue(KEY_PerPixelRadius());
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	k_AdaptiveStruct A(r_min, r_max, m_pEntries, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	BlockSampleImage img = m_pBlockSampler->getBlockImage();

	if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if(dynamic_cast<PointStorage*>(m_pVolumeEstimator))
		k_EyePass<PointStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamBeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if (dynamic_cast<BeamBVHStorage*>(m_pVolumeEstimator))
		k_EyePass<BeamBVHStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, (float)m_uPassesDone, radius2, A, img, m_useDirectLighting, perPixelRad, finalGathering);

	ThrowCudaErrors(cudaThreadSynchronize());
}

CUDA_DEVICE int g_MaxRad, g_MinRad;
CUDA_FUNC_IN int floatToOrderedInt(float floatVal) {
	int intVal = float_as_int_(floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}
CUDA_FUNC_IN float orderedIntToFloat(int intVal) {
	return int_as_float_((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__global__ void k_PerPixelRadiusEst(int w, int h, float r_max, float r_1, k_AdaptiveStruct adpt, int k_toFind)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		auto& e = adpt(x, y);
		//adaptive progressive intit
		e.E_psi = e.E_psi2 = e.E_DI = e.E_DI2 = e.DI = 0.0f;

		//initial per pixel rad estimate
		CudaRNG rng = g_RNGData();
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		NormalizedT<Ray> r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = traceRay(r);
		if (r2.hasHit())
		{
			const float search_rad = r_1;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			auto f_t = bRec.dg.sys.t * search_rad, f_s = bRec.dg.sys.s * search_rad;
			Vec3f a = -1.0f * f_t - f_s, b = f_t - f_s, c = -1.0f * f_t + f_s, d = f_t + f_s;
			Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
			int k_found = 0;
#ifdef ISCUDA
			g_SurfMap.ForAll(low, high, [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
			{
				float dist2 = distanceSquared(ph.getPos(g_SurfMap.getHashGrid(), cell_idx), bRec.dg.P);
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
		g_RNGData(rng);
	}
}

void PPPMTracer::doPerPixelRadiusEstimation()
{
	int a = floatToOrderedInt(FLT_MAX), b = floatToOrderedInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxRad, &b, sizeof(b)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MinRad, &a, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	int p = 32;
	if (m_pEntries)
		k_PerPixelRadiusEst << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(w, h, r_max * 0.1f, m_fInitialRadiusSurf, k_AdaptiveStruct(r_min, r_max, m_pEntries, w, m_uPassesDone), k_Intial);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
	std::cout << "m_fIntitalRadMin = " << m_fIntitalRadMin << ", m_fIntitalRadMax = " << m_fIntitalRadMax << "\n";
}

}
