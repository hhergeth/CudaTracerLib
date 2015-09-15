#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

#define MAX_PHOTONS_PER_CELL 30
texture<uint4, 1> t_Photons;
texture<int2, 1> t_Beams;
texture<int2, 1> t_PhotonBeams;
CUDA_DEVICE k_BeamGrid g_PhotonBeamGrid;
CUDA_CONST k_pGridEntry* g_SurfaceEntries2;

template<bool HAS_MULTIPLE_MAPS, typename PHOTON> CUDA_FUNC_IN PHOTON loadPhoton(unsigned int idx, const k_PhotonMapCollection<HAS_MULTIPLE_MAPS, PHOTON>& map)
{
	PHOTON e = map.m_pPhotons[idx];
	//uint4 dat = tex1Dfetch(t_Photons, idx);
	//PHOTON e = *(k_pPpmPhoton*)&dat;
	return e;
}

template<bool VOL> CUDA_DEVICE Spectrum L_Volume1(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, unsigned int a_NodeIndex = 0xffffffff)
{
	const k_PhotonMapReg& map = photonMap.m_sVolumeMap;
	Spectrum Tau = Spectrum(0.0f);
	float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * photonMap.m_uPhotonNumEmitted), r2 = a_r * a_r;
	Spectrum L_n = Spectrum(0.0f);
	float a, b;
	if (!map.m_sHash.getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	float minT = a = math::clamp(a, tmin, tmax);
	b = math::clamp(b, tmin, tmax);
	float d = 2.0f * a_r;
	while (a < b)
	{
		float t = a + d / 2.0f;
		Vec3f x = r(t);
		uint3 lo = map.m_sHash.Transform(x - Vec3f(a_r)), hi = map.m_sHash.Transform(x + Vec3f(a_r));
		for (unsigned int ac = lo.x; ac <= hi.x; ac++)
			for (unsigned int bc = lo.y; bc <= hi.y; bc++)
				for (unsigned int cc = lo.z; cc <= hi.z; cc++)
				{
					unsigned int i0 = map.m_sHash.Hash(Vec3u(ac, bc, cc)), i = map.m_pDeviceHashGrid[i0];
					while (i != 0xffffffff && i != 0xffffff)
					{
						k_pPpmPhoton e = loadPhoton(i, photonMap);
						Vec3f wi = e.getWi(), P = e.getPos(map.m_sHash, Vec3u(ac, bc, cc));
						Spectrum l = e.getL();
						if (distanceSquared(P, x) < r2)
						{
							float p = VOL ? g_SceneData.m_sVolume.p(x, r.direction, wi, rng, a_NodeIndex) : Warp::squareToUniformSpherePdf();
							float l1 = dot(P - r.origin, r.direction) / dot(r.direction, r.direction);
							Spectrum tauToPhoton = VOL ? (-Tau - g_SceneData.m_sVolume.tau(r, a, l1)).exp() : (-sigt * (l1 - minT)).exp();
							L_n += p * l * Vs * tauToPhoton * d;
						}
						i = e.getNext();
					}
				}
		Spectrum tauDelta = VOL ? g_SceneData.m_sVolume.tau(r, a, a + d, a_NodeIndex) : sigt * d;
		Tau += tauDelta;
		if (VOL)
			L_n += g_SceneData.m_sVolume.Lve(x, -1.0f * r.direction, a_NodeIndex) * d;
		a += d;
	}
	Tr = (-Tau).exp();
	return L_n;
}

template<bool VOL> CUDA_DEVICE Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, unsigned int a_NodeIndex = 0xffffffff)
{
	Spectrum Tau = Spectrum(0.0f);
	float r2 = a_r * a_r;
	Spectrum L_n = Spectrum(0.0f);
	TraverseGrid(r, photonMap.m_sVolumeMap.m_sHash, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
#ifdef ISCUDA
		int2 beam;
		beam.y = photonMap.m_sVolumeMap.m_sHash.Hash(cell_pos);
		while (beam.y != -1)
		{
			beam = tex1Dfetch(t_Beams, beam.y);
			if (beam.x != -1)
			{
				k_pPpmPhoton e = loadPhoton(beam.x, photonMap);
				Vec3f wi = e.getWi(), P = e.getPos(photonMap.m_sVolumeMap.m_sHash, cell_pos);
				float r_p2 = half(e.accessNormalStorage()).ToFloat();
				float l1 = dot(P - r.origin, r.direction) / dot(r.direction, r.direction);
				if (distanceSquared(P, r(l1)) < r_p2 && rayT <= l1 && l1 <= cellEndT)
				{
					float p = VOL ? g_SceneData.m_sVolume.p(P, r.direction, wi, rng, a_NodeIndex) : Warp::squareToUniformSpherePdf();
					Spectrum tauToPhoton = VOL ? (-Tau - g_SceneData.m_sVolume.tau(r, rayT, l1)).exp() : (-sigt * (l1 - minT)).exp();
					L_n += p * e.getL() / (PI * photonMap.m_uPhotonNumEmitted * r_p2) * tauToPhoton;
				}
			}
		}
		float localDist = cellEndT - rayT;
		Spectrum tauD = VOL ? g_SceneData.m_sVolume.tau(r, rayT, cellEndT) : sigt * localDist;
		Tau += tauD;
		if (VOL)
			L_n += g_SceneData.m_sVolume.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
#endif
	});
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_FUNC_IN void skew_lines(const Ray& r, const Ray& r2, float& t1, float& t2)
{
	float v1dotv2 = dot(r.direction, r2.direction), v1p2 = r.direction.lenSqr(), v2p2 = r2.direction.lenSqr();
	float x = dot(r2.origin - r.origin, r.direction), y = dot(r2.origin - r.origin, r2.direction);
	float dc = 1.0f / (v1dotv2 * v1dotv2 - v1p2 * v2p2);
	t1 = dc * (-v2p2 * x + v1dotv2 * y);
	t2 = dc * (-v1dotv2 * x + v1p2 * y);
}
template<bool VOL> CUDA_DEVICE Spectrum L_Volume3(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, unsigned int a_NodeIndex = 0xffffffff)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	TraverseGrid(r, photonMap.m_sVolumeMap.m_sHash, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
#ifdef ISCUDA
		int2 beam;
		beam.y = photonMap.m_sVolumeMap.m_sHash.Hash(cell_pos);
		while (beam.y != -1)
		{
			beam = tex1Dfetch(t_PhotonBeams, beam.y);
			if (beam.x != -1)
			{
				k_Beam B = g_PhotonBeamGrid.m_pDeviceBeams[beam.x];
				float t1, t2;
				skew_lines(r, Ray(B.pos, B.dir), t1, t2);
				Vec3f p_b = B.pos + t2 * B.dir, p_c = r.origin + t1 * r.direction;//photonMap.m_sVolumeMap.m_sHash.Transform(p_b) == cell_pos && 
				if (t1 > 0 && t2 > 0 && t2 < B.t && t1 < cellEndT && distanceSquared(p_c, p_b) < a_r * a_r && photonMap.m_sVolumeMap.m_sHash.Transform(p_c) == cell_pos)
				{
					Spectrum tauC = Tau + (VOL ? g_SceneData.m_sVolume.tau(r, rayT, t1) : sigt * (t1 - rayT));
					Spectrum tauP = VOL ? g_SceneData.m_sVolume.tau(Ray(B.pos, B.dir), 0, t2) : sigt * (t2 - 0);
					float p = VOL ? g_SceneData.m_sVolume.p(p_b, (p_c - p_b).normalized(), -B.dir, rng, a_NodeIndex) : Warp::squareToUniformSpherePdf();
					float sin_theta_b = math::sqrt(1 - math::sqr(dot(r.direction, B.dir) / (r.direction.length() * B.dir.length())));
					L_n += 1.0f / (photonMap.m_uPhotonNumEmitted * a_r) * p * B.Phi * (-tauC - tauP).exp() / sin_theta_b;
				}
			}
		}
		float localDist = cellEndT - rayT;
		Spectrum tauD = VOL ? g_SceneData.m_sVolume.tau(r, rayT, cellEndT) : sigt * localDist;
		Tau += tauD;
		if (VOL)
			L_n += g_SceneData.m_sVolume.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
#endif
	});
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_FUNC_IN float calculatePlaneAreaInCell(const AABB& planeBox)
{
	return length(Vec3f(planeBox.minV.x, planeBox.maxV.y, planeBox.maxV.z) - planeBox.minV) * length(Vec3f(planeBox.maxV.x, planeBox.minV.y, planeBox.minV.z) - planeBox.minV);
}
CUDA_FUNC_IN AABB calculatePlaneAABBInCell(const AABB& cell, const Vec3f& p, const Vec3f& n, float r)
{
	Vec3f down_plane = normalize(cross(Vec3f(1,0,0), n));
	Vec3f left_plane = normalize(cross(down_plane, n));
	AABB res = AABB::Identity();
	res = res.Extend(cell.Clamp(p + r * down_plane));
	res = res.Extend(cell.Clamp(p + r * left_plane));
	res = res.Extend(cell.Clamp(p - r * down_plane));
	res = res.Extend(cell.Clamp(p - r * left_plane));
	return res;
}
CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, const k_PhotonMapReg& map)
{
	Spectrum Lp = Spectrum(0.0f);
	const float r2 = a_rSurfaceUNUSED * a_rSurfaceUNUSED;
	Frame sys = Frame(bRec.dg.n);
	sys.t *= a_rSurfaceUNUSED;
	sys.s *= a_rSurfaceUNUSED;
	Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
	Vec3u lo = map.m_sHash.Transform(low), hi = map.m_sHash.Transform(high);
	for(unsigned int a = lo.x; a <= hi.x; a++)
		for(unsigned int b = lo.y; b <= hi.y; b++)
			for(unsigned int c = lo.z; c <= hi.z; c++)
			{
				unsigned int i0 = map.m_sHash.Hash(Vec3u(a, b, c)), i = map.m_pDeviceHashGrid[i0];
				while (i != 0xffffffff && i != 0xffffff)
				{
					k_pPpmPhoton e = loadPhoton(i, photonMap);
					Vec3f n = e.getNormal(), wi = e.getWi(), P = e.getPos(map.m_sHash, Vec3u(a,b,c));
					Spectrum l = e.getL();
					float dist2 = distanceSquared(P, bRec.dg.P);
					if (dist2 < r2 )//&& dot(n, bRec.dg.sys.n) > 0.8f
					{
						bRec.wo = bRec.dg.toLocal(wi);
						Spectrum bsdfFactor = mat->bsdf.f(bRec);
						float ke = k_tr(a_rSurfaceUNUSED, math::sqrt(dist2));
						Lp += PI * ke * l * bsdfFactor / Frame::cosTheta(bRec.wo);
					}
					i = e.getNext();
				}
				/*unsigned int cell_idx = map.m_sHash.Hash(Vec3u(a, b, c));
				if (cell_idx < map.m_uGridLength)
				{
					AABB cell_world = map.m_sHash.getCell(Vec3u(a, b, c));
					AABB disk_aabb_in_box = calculatePlaneAABBInCell(cell_world, bRec.dg.P, bRec.dg.sys.n, a_rSurfaceUNUSED);
					AABB plane_aabb_in_box = calculatePlaneAABBInCell(cell_world, bRec.dg.P, bRec.dg.sys.n, map.m_sHash.m_vCellSize.length() * 5);
					float scale = (calculatePlaneAreaInCell(disk_aabb_in_box)) / calculatePlaneAreaInCell(plane_aabb_in_box);
					auto& e = g_SurfaceEntries2[cell_idx];
					//float X = disk_aabb_in_box.maxV.x - disk_aabb_in_box.minV.x, X2 = disk_aabb_in_box.maxV.x * disk_aabb_in_box.maxV.
					Spectrum m_sValue = e.m_sValues[0];
					Lp += mat->bsdf.As<diffuse>()->m_reflectance.Evaluate(bRec.dg) * m_sValue / (PI * r2);
				}*/
			}
	return Lp / float(photonMap.m_uPhotonNumEmitted);
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, k_AdaptiveStruct& A, int x, int y,
	const Spectrum& importance, int a_PassIndex, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap)
{
	//Adaptive Progressive Photon Mapping Implementation
	k_AdaptiveEntry ent = A(x,y);
	float r2 = ent.r * ent.r, maxr = max(ent.r, ent.rd), rd2 = ent.rd * ent.rd, rd = ent.rd, r = ent.r;
	Frame sys = bRec.dg.sys;
	sys.t *= maxr;
	sys.s *= maxr;
	sys.n *= maxr;
	Vec3f ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
	const k_PhotonMapReg& map = photonMap.m_sSurfaceMap;
	Vec3u lo = map.m_sHash.Transform(low), hi = map.m_sHash.Transform(high);
	Spectrum Lp = 0.0f;
	for (unsigned int a = lo.x; a <= hi.x; a++)
	for (unsigned int b = lo.y; b <= hi.y; b++)
	for (unsigned int c = lo.z; c <= hi.z; c++)
	{
		unsigned int i0 = map.m_sHash.Hash(Vec3u(a, b, c)), i = map.m_pDeviceHashGrid[i0];
		while (i != 0xffffffff && i != 0xffffff)
		{
			k_pPpmPhoton e = loadPhoton(i, photonMap);
			Vec3f nor = e.getNormal(), wi = e.getWi(), P = e.getPos(map.m_sHash, Vec3u(a,b,c));
			Spectrum l = e.getL();
			float dist2 = distanceSquared(P, bRec.dg.P);
			if (dot(nor, bRec.dg.sys.n) > 0.95f)
			{
				bRec.wo = bRec.dg.toLocal(wi);
				Spectrum bsdfFactor = mat->bsdf.f(bRec);
				float psi = Spectrum(importance * bsdfFactor * l).getLuminance();
				if (dist2 < rd2)
				{
					const Vec3f e_l = bRec.dg.P - P;
					float cc = k_tr(rd, e_l);
					float laplu = k_tr(rd, e_l + ur) + k_tr(rd, e_l - ur) - 2.0f * cc,
						  laplv = k_tr(rd, e_l + vr) + k_tr(rd, e_l - vr) - 2.0f * cc,
						  lapl = psi / rd2 * (laplu + laplv);
					ent.I += lapl;
					ent.I2 += lapl * lapl;
				}
				if (dist2 < r2)
				{
					float kri = k_tr(r, math::sqrt(dist2));
					Lp += PI * kri * l * bsdfFactor / Frame::cosTheta(bRec.wo);
					ent.psi += psi;
					ent.psi2 += psi * psi;
					ent.pl += kri;
				}
			}
			i = e.getNext();
		}
	}
	float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / a_PassIndex;

	if (VAR_Lapl)
	{
		ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(a_PassIndex, -1.0f / 8.0f);
		ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);
	}

	if (VAR_Lapl && VAR_Phi)
	{
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float ta = (2.0f * math::sqrt(VAR_Phi)) / (PI * float(photonMap.m_uPhotonNumEmitted) * E_pl * k_22 * E_I * E_I);
		ent.r = math::pow(ta, 1.0f / 6.0f) * math::pow(a_PassIndex, -1.0f / 6.0f);
		ent.r = math::clamp(ent.r, A.r_min, A.r_max);
	}
	A(x,y) = ent;
	//return 0.0f;
	return Lp / float(photonMap.m_uPhotonNumEmitted);
	//return L_Surface(bRec, ent.r, mat, photonMap, photonMap.m_sSurfaceMap);
}

template<bool DIRECT> CUDA_FUNC_IN Spectrum L_FinalGathering(TraceResult& r2, BSDFSamplingRecord& bRec, CudaRNG& rng, float a_rSurfaceUNUSED, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap)
{
	Spectrum LCaustic = L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), photonMap, photonMap.m_sCausticMap);
	Spectrum L(0.0f);
	const int N = 10;
	for (int i = 0; i < N; i++)
	{
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		Ray r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = k_TraceRay(r);
		if (r3.hasHit())
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec2(dg);
			r3.getBsdfSample(r, bRec2, ETransportMode::ERadiance, &rng);
			L += f * L_Surface(bRec2, a_rSurfaceUNUSED, &r3.getMat(), photonMap, photonMap.m_sSurfaceMap);
			if (DIRECT)
				L += f * UniformSampleAllLights(bRec2, r3.getMat(), 1, rng);
			else L += f * r3.Le(bRec2.dg.P, bRec2.dg.sys, -r.direction);
		}
	}
	return L / float(N) + LCaustic;
}

template<bool DIRECT, bool FINAL_GATHER> CUDA_FUNC_IN void k_EyePassF(int x, int y, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct a_AdpEntries, k_BlockSampleImage& img, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2f screenPos = Vec2f(x, y) + rng.randomFloat2();
	Ray r, rX, rY;
	Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2()), importance = throughput;
	TraceResult r2;
	r2.Init();
	int depth = -1;
	Spectrum L(0.0f);
	while(k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
	{
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
		if (depth == 0)
			dg.computePartials(r, rX, rY);
		if(g_SceneData.m_sVolume.HasVolumes())
		{
			float tmin, tmax;
			if (g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
			{
				Spectrum Tr;
#ifdef ISCUDA
				L += throughput * L_Volume<true>(a_rVolume, rng, r, tmin, tmax, Spectrum(0.0f), Spectrum(0.0f), Tr, photonMap);
#endif
				throughput = throughput * Tr;
			}
		}
		if(DIRECT)
			L += throughput * UniformSampleOneLight(bRec, r2.getMat(), rng);
		L += throughput * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);//either it's the first bounce -> account or it's a specular reflection -> ...
		const e_KernelBSSRDF* bssrdf;
		if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
		{
			Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			bRec.wo.z *= -1.0f;
			Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
			TraceResult r3 = k_TraceRay(rTrans);
			Spectrum Tr;
#ifdef ISCUDA
			L += throughput * L_Volume<false>(a_rVolume, rng, rTrans, 0, r3.m_fDist, bssrdf->sigp_s + bssrdf->sig_a, bssrdf->sigp_s, Tr, photonMap, r2.getNodeIndex());
#endif
			//break;
		}
		bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
			 hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
			 hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
		if (hasSmooth)
		{
			if (FINAL_GATHER)
				L += throughput * (hasGlossy ? 0.5f : 1) * L_FinalGathering<DIRECT>(r2, bRec, rng, a_rSurfaceUNUSED, photonMap);
			else L += throughput * (hasGlossy ? 0.5f : 1) * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), photonMap, photonMap.m_sSurfaceMap);
			//L += throughput * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), a_AdpEntries, x, y, importance, a_PassIndex, photonMap);
			if(!hasSpecGlossy)
				break;
		}
		if (hasSpecGlossy)
		{
			bRec.sampledType = 0;
			bRec.typeMask = EDelta | EGlossy;
			Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			if(!bRec.sampledType)
				break;
			throughput = throughput * t_f * (hasGlossy ? 0.5f : 1);
			importance = t_f;
			r = Ray(bRec.dg.P, bRec.getOutgoing());
			r2.Init();
		}
		else break;
	}
	if(!r2.hasHit())
	{
		if(g_SceneData.m_sVolume.HasVolumes())
		{
			float tmin, tmax;
			g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax);
			Spectrum Tr;
#ifdef ISCUDA
			L += throughput * L_Volume<true>(a_rVolume, rng, r, tmin, tmax, Spectrum(0.0f), Spectrum(0.0f), Tr, photonMap);
#endif
		}
		L += throughput * g_SceneData.EvalEnvironment(r);
	}
	img.Add(screenPos.x, screenPos.y, L);
	//Spectrum qs;
	//float t = A.E[y * w + x].r / a_rSurfaceUNUSED;
	//t = (A.E[y * w + x].r - A.r_min) / (A.r_max - A.r_min);
	//qs.fromHSL(1.0f / 3.0f - t / 3.0f, 1, 0.5f);
	//g_Image.AddSample(screenPos.x, screenPos.y, qs);
	//auto& ent = a_AdpEntries(x, y);
	/*float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / NJ;
	g_Image.AddSample(screenPos.x, screenPos.y, Spectrum(VAR_Phi*100));*/
	//float v = (ent.I2 - (ent.I * ent.I) / a_PassIndex) / a_PassIndex * 1e-10f;
	//float v = (ent.rd - a_AdpEntries.r_min) / (a_AdpEntries.r_max - a_AdpEntries.r_min);
	//img.Add(x, y, Spectrum(math::abs(v)));
	g_RNGData(rng);
}

template<bool DIRECT, bool FINAL_GATHER> __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage img, k_PhotonMapCollection<true, k_pPpmPhoton> photonMap)
{
	Vec2i pixel = k_TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		k_EyePassF<DIRECT, FINAL_GATHER>(pixel.x, pixel.y, w, h, a_PassIndex, a_rSurfaceUNUSED, a_rVolume, A, img, photonMap);
		//img.img.AddSample(pixel.x, pixel.y, Spectrum(A.E[pixel.y * w + pixel.x].r / a_rSurfaceUNUSED));
	}
}

void k_sPpmTracer::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	cudaChannelFormatDesc cdu4 = cudaCreateChannelDesc<uint4>(), cdi2 = cudaCreateChannelDesc<int2>();
	size_t offset = 0;
	ThrowCudaErrors(cudaBindTexture(&offset, &t_Photons, m_sMaps.m_pPhotons, &cdu4, m_sMaps.m_uPhotonNumStored * sizeof(uint4)));
	if (m_sBeams.m_pDeviceData)
		ThrowCudaErrors(cudaBindTexture(&offset, &t_Beams, m_sBeams.m_pDeviceData, &cdi2, m_sBeams.m_uNumEntries * sizeof(int2)));
	if (m_sPhotonBeams.m_pGrid)
		ThrowCudaErrors(cudaBindTexture(&offset, &t_PhotonBeams, m_sPhotonBeams.m_pGrid, &cdi2, m_sPhotonBeams.m_uGridLength * sizeof(int2)));
	//ThrowCudaErrors(cudaMemcpyToSymbol(g_PhotonBeamGrid, &m_sPhotonBeams, sizeof(m_sPhotonBeams)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceEntries2, &m_pSurfaceValues, sizeof(m_pSurfaceValues)));

	float radius2 = getCurrentRadius2(2);
	float radius3 = getCurrentRadius2(3);

	//radius2 = radius3 = m_fInitialRadius * 10;
	k_AdaptiveStruct A(r_min, r_max, m_pEntries, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	k_BlockSampleImage img = m_pBlockSampler->getBlockImage();
	if (m_bDirect)
	{
		if (m_bFinalGather)
			k_EyePass<true, true> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_sMaps);
		else k_EyePass<true, false> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_sMaps);
	}
	else
	{
		if (m_bFinalGather)
			k_EyePass<false, true> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_sMaps);
		else k_EyePass<false, false> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_sMaps);
	}
	ThrowCudaErrors(cudaThreadSynchronize());
}

__global__ void k_StartPass(int w, int h, float r, float rd, k_AdaptiveEntry* E)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * w + x;
	if(x < w && y < h)
	{
		E[i].r = r;
		E[i].rd = rd;
		E[i].psi = E[i].psi2 = E[i].I = E[i].I2 = E[i].pl = 0.0f;
	}
}

void k_sPpmTracer::doStartPass(float r, float rd)
{
	int p = 32;
	if (m_pEntries)
		k_StartPass<<<dim3(w / p + 1, h / p + 1, 1), dim3(p,p,1)>>>(w, h, r, rd, m_pEntries);
}