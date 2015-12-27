#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>

namespace CudaTracerLib {

CUDA_FUNC_IN bool sphere_line_intersection(const Vec3f& p, float rad, const Ray& r, float& t_min, float& t_max)
{
	Vec3f d = r.direction.normalized();
	Vec3f oc = r.origin - p;
	float f = dot(d, oc);
	float w = f * f - oc.lenSqr() + rad * rad;
	if (w < 0)
		return false;
	t_min = -f - math::sqrt(w);
	t_max = -f + math::sqrt(w);
	return true;
}

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum PointStorage::L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	float Vs = 1.0f / (4.0f / 3.0f * PI * m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_uNumEmitted);
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
				float p = vol.p(x, -r.direction, ph.getWi(), rng);
				L_i += p * ph.getL() * Vs;
			}
		});
		L_n += (-Tau - vol.tau(r, a, t)).exp() * L_i * d;
		Tau += vol.tau(r, a, a + d);
		L_n += vol.Lve(x, -1.0f * r.direction) * d;
		a += d;
	}
	Tr = (-Tau).exp();
	return L_n;
}

template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum BeamGrid::L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum Tau = Spectrum(0.0f);
	Spectrum L_n = Spectrum(0.0f);
	TraverseGrid(r, m_sStorage.getHashGrid(), tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sBeamGridStorage.ForAllCellEntries(cell_pos, [&](unsigned int, entry beam_idx)
		{
			const volPhoton& ph = m_sStorage(beam_idx.i & ~(1 << 31));
			Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_pos);
			float l1 = dot(ph_pos - r.origin, r.direction);
			float isectRadSqr = distanceSquared(ph_pos, r(l1));
			if (isectRadSqr < ph.getRad() && l1 >= 0)
			{
				float p = vol.p(ph_pos, r.direction, ph.getWi(), rng);
				Spectrum tauToPhoton = (-Tau - (l1 >= rayT ? vol.tau(r, rayT, l1) : -vol.tau(r, max(minT, l1), rayT))).exp();//corner case : the photon lies in the cell but the projected distance is before the cell

				//Spectrum tauToPhotonC = (-vol.tau(r, tmin, l1)).exp();
				//if ((tauToPhotonC - tauToPhoton).abs().max() > 1e-3f)
				//	printf("{%f, %f, %f}, {%f, %f, %f}\n", tauToPhotonC[0], tauToPhotonC[1], tauToPhotonC[2], tauToPhoton[0], tauToPhoton[1], tauToPhoton[2]);

				L_n += p * ph.getL() / m_uNumEmitted * tauToPhoton * 
					(1 - isectRadSqr / ph.getRad()) / (ph.getRad() * PI * 0.5f);
			}
			/*float t1, t2;
			if (sphere_line_intersection(ph_pos, m_fCurrentRadiusVol, r, t1, t2))
			{
				//transmittance from camera vertex along ray to query point
				Spectrum tauToPhoton = (-Tau - (l1 >= rayT ? vol.tau(r, rayT, t1) : -vol.tau(r, max(minT, t1), rayT))).exp();//corner case : the photon lies in the cell but the projected distance is before the cell
				float p = vol.p(r((t1 + t2) / 2), r.direction, ph.getWi(), rng);
				float Vs = 1.0f / (4.0f / 3.0f * PI * m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_uNumEmitted);
				L_n += p * ph.getL() * Vs * (-vol.tau(r, t1, t2)).exp();
			}*/
		});
		Tau += vol.tau(r, rayT, cellEndT);
		float localDist = cellEndT - rayT;
		L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
	});
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_CONST SurfaceMapT g_SurfMap;
CUDA_CONST SurfaceMapT g_SurfMapCaustic;
CUDA_CONST unsigned int g_NumPhotonEmitted2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax4(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid), sizeof(BeamBVHStorage))];

template<bool USE_GLOBAL> CUDA_ONLY_FUNC Spectrum beam_beam_L(const VolHelper<USE_GLOBAL>& vol, CudaRNG& rng, const Beam& B, const Ray& r, float radius, float beamIsectDist, float queryIsectDist, float beamBeamDistance, int m_uNumEmitted, float sinTheta, float tmin)
{
	Spectrum photon_tau = vol.tau(Ray(B.getPos(), B.getDir()), 0, beamIsectDist);
	Spectrum camera_tau = vol.tau(r, tmin, queryIsectDist);
	Spectrum camera_sc = vol.sigma_s(r(queryIsectDist), r.direction);
	float p = vol.p(r(queryIsectDist), r.direction, B.getDir(), rng);
	//return B.Phi / float(m_uNumEmitted) * camera_sc * (-photon_tau).exp() * (-camera_tau).exp() * p * (1 - beamBeamDistance * beamBeamDistance / (radius * radius)) * 3 / (4 * radius*sinTheta);
	float t = math::clamp01(beamBeamDistance / radius), k = 1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f);
	return camera_sc / radius * p * B.getL() / m_uNumEmitted * (-photon_tau).exp() * (-camera_tau).exp() / sinTheta * k;
}

template<bool USE_GLOBAL> CUDA_ONLY_FUNC Spectrum BeamBeamGrid::L_Volume(float, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	float radius = m_fCurrentRadiusVol;
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
		if (Beam::testIntersectionBeamBeam(r.origin, r.direction, tmin, tmax, B.getPos(), B.getDir(), 0, B.t, radius * radius, beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, rng, B, r, radius, beamIsectDist, queryIsectDist, beamBeamDistance, m_uNumEmitted, sinTheta, tmin);
	}
	Tr = (-vol.tau(r, tmin, tmax)).exp();
	return L_n;
}

template<bool USE_GLOBAL> Spectrum BeamBVHStorage::L_Volume(float radius, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const
{
	Spectrum L_n = Spectrum(0.0f);
	iterateBeams(Ray(r(tmin), r.direction), tmax - tmin, [&](unsigned int ref_idx)
	{
		const BeamRef& R = m_pDeviceRefs[ref_idx];
		//unsigned int beam_idx = R.getIdx();
		const Beam& B = R.beam;// this->m_pDeviceBeams[beam_idx];
		float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
		if (Beam::testIntersectionBeamBeam(r.origin, r.direction, tmin, tmax, B.getPos(), B.getDir(), R.t_min, R.t_max, radius * radius, beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			L_n += beam_beam_L(vol, rng, B, r, radius, beamIsectDist, queryIsectDist, beamBeamDistance, m_uNumEmitted, sinTheta, tmin);
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
			float ke = k_tr(r, math::sqrt(dist2));
			Spectrum l = ph.getL();
			if(F_IS_GLOSSY)
				l *= mat.bsdf.f(bRec);
			Lp += ke * l * cor_fac;
		}
	});
	if(!F_IS_GLOSSY)
		Lp *= mat.bsdf.f(bRec);
	return Lp / g_NumPhotonEmitted2;
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
		Ray r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = traceRay(r);
		if (r3.hasHit())
		{
			r3.getBsdfSample(r, bRec2, ETransportMode::ERadiance, &rng);
			bool hasGlossy = r3.getMat().bsdf.hasComponent(EGlossy);
			L += f * (hasGlossy ? L_Surface<true>(bRec2, -r.direction, rad, r3.getMat()) : L_Surface<false>(bRec2, -r.direction, rad, r3.getMat()));
			if (DIRECT)
				L += f * UniformSampleOneLight(bRec2, r3.getMat(), rng);
			else L += f * r3.Le(bRec2.dg.P, bRec2.dg.sys, -r.direction);
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
	//float ta = (2.0f * math::sqrt(VAR_Psi)) / (PI * float(g_NumPhotonEmitted2) * E_pl * k_22 * E_I * E_I);
	//float q = math::pow(ta, 1.0f / 6.0f) * math::pow(iteration, -1.0f / 6.0f);
	//float p = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	//ent.r = math::clamp(q, A.r_min, A.r_max);

	//Adaptive Progressive Photon Mapping Implementation
	float NJ = iteration * g_NumPhotonEmitted2, scaleA = (NJ - 1) / NJ, scaleB = 1.0f / NJ;
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
			float psi = Spectrum(importance * bsdfFactor * ph.getL() / float(g_NumPhotonEmitted2)).getLuminance();//this 1/J has nothing to do with E[X], it is scaling for radiance distribution
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
				float kri = k_tr(r, math::sqrt(dist2));
				Lp += kri * ph.getL() / float(g_NumPhotonEmitted2) * bsdfFactor;
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
	float t = ent.compute_r(iteration, g_NumPhotonEmitted2, g_NumPhotonEmitted2 * iteration) / (A.r_max * 10 - A.r_min) / 0.05f;
	//float t = ent.compute_rd(iteration) / (A.r_max - A.r_min) * 1000;
	qs.fromHSL(2.0f / 3.0f * (1 - math::clamp01(t)), 1, 0.5f);//0 -> 1 : Dark Blue -> Light Blue -> Green -> Yellow -> Red
	img.Add(x, y, qs);

	A(x, y) = ent;
	return Lp;
#else
	return 0.0f;
#endif
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct a_AdpEntries, BlockSampleImage img, bool DIRECT, bool USE_PerPixelRadius, bool finalGathering)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		Vec2f screenPos = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		Ray r, rX, rY;
		Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());
		TraceResult r2;
		r2.Init();
		int depth = -1;
		Spectrum L(0.0f);
		while (traceRay(r.direction, r.origin, &r2) && depth++ < 5)
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
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, r, tmin, tmax, VolHelper<true>(), Tr);
					throughput = throughput * Tr;
				}
			}
			if (DIRECT)
				L += throughput * UniformSampleOneLight(bRec, r2.getMat(), rng);
			L += throughput * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);//either it's the first bounce or it's a specular reflection
			const VolumeRegion* bssrdf;
			if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = traceRay(rTrans);
				Spectrum Tr;
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr);
				//throughput = throughput * Tr;//break;
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
					l = finalGathering ? L_SurfaceFinalGathering<true>(bRec, -r.direction, r_i, r2, rng, DIRECT) : L_Surface<true>(bRec, -r.direction, r_i, r2.getMat());
				else l = finalGathering ? L_SurfaceFinalGathering<false>(bRec, -r.direction, r_i, r2, rng, DIRECT) : L_Surface<false>(bRec, -r.direction, r_i, r2.getMat());
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
				r = Ray(bRec.dg.P, bRec.getOutgoing());
				r2.Init();
			}
			else break;
		}

		if (!r2.hasHit())
		{
			Spectrum Tr(1);
			float tmin, tmax;
			if (g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, r, tmin, tmax, VolHelper<true>(), Tr);
			L += Tr * throughput * g_SceneData.EvalEnvironment(r);
		}
		img.Add((int)screenPos.x, (int)screenPos.y, L);
	}
	g_RNGData(rng);
}

__global__ void k_EyePass2(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct A, BlockSampleImage img, float rMax, float rMin)
{
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	/*Ray r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	BeamBeamGrid* grid = (BeamBeamGrid*)g_VolEstimator2;
	int n = 0;
	#ifdef ISCUDA
	TraverseGrid(r, grid->m_sStorage.hashMap, 0, FLT_MAX, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
	grid->m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
	{
	n += beam_idx == 1234;
	//if(pixel.x == 200 && pixel.y == 200)
	//	printf("(%u, %d), ", ABC, beam_idx);
	});
	});
	#endif
	img.Add(pixel.x, pixel.y, Spectrum(n!=0));*/
	float rq = (getCurrentRadius(A(pixel.x, pixel.y).r_std, a_PassIndex, 2) - a_rSurface) / getCurrentRadius(rMax, a_PassIndex, 2);
	img.Add(pixel.x, pixel.y, Spectrum(rq > 0 ? rq : 0, rq < 0 ? -rq : 0, 0));
	//float ab = getCurrentRadius(A(pixel.x, pixel.y).r, a_PassIndex, 2) < a_rSurface;
	//img.Add(pixel.x, pixel.y, Spectrum(ab));
}

void PPPMTracer::Debug(Image* I, const Vec2i& pixel)
{
	
}

void PPPMTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	float radius2 = getCurrentRadius(2);
	float radius3 = getCurrentRadius(3);

	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMapCaustic, &m_sSurfaceMapCaustic, sizeof(m_sSurfaceMapCaustic)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmitted2, &m_uPhotonEmittedPass, sizeof(m_uPhotonEmittedPass)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	bool perPixelRad = m_sParameters.getValue(KEY_PerPixelRadius());
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	k_AdaptiveStruct A(r_min, r_max, m_pEntries, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	BlockSampleImage img = m_pBlockSampler->getBlockImage();

	if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if(dynamic_cast<PointStorage*>(m_pVolumeEstimator))
		k_EyePass<PointStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamBeamGrid> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	else if (dynamic_cast<BeamBVHStorage*>(m_pVolumeEstimator))
		k_EyePass<BeamBVHStorage> << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_useDirectLighting, perPixelRad, finalGathering);
	//k_EyePass2 << <BLOCK_SAMPLER_LAUNCH_CONFIG >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_fIntitalRadMin, m_fIntitalRadMax);

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
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = traceRay(r);
		if (r2.hasHit())
		{
			const float search_rad = r_1;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			Frame sys = Frame(bRec.dg.n);
			sys.t *= search_rad;
			sys.s *= search_rad;
			Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
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
		k_PerPixelRadiusEst << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(w, h, r_max * 0.1f, m_fInitialRadius, k_AdaptiveStruct(r_min, r_max, m_pEntries, w, m_uPassesDone), k_Intial);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
	std::cout << "m_fIntitalRadMin = " << m_fIntitalRadMin << ", m_fIntitalRadMax = " << m_fIntitalRadMax << "\n";
}

}
