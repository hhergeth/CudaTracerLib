#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

#define MAX_PHOTONS_PER_CELL 30
texture<uint4, 1> t_Photons;
texture<int2, 1> t_Beams;

template<bool HAS_MULTIPLE_MAPS, typename PHOTON> CUDA_FUNC_IN PHOTON loadPhoton(unsigned int idx, const k_PhotonMapCollection<HAS_MULTIPLE_MAPS, PHOTON>& map)
{
	PHOTON e = map.m_pPhotons[idx];
	//uint4 dat = tex1Dfetch(t_Photons, idx);
	//PHOTON e = *(k_pPpmPhoton*)&dat;
	return e;
}

template<bool VOL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, unsigned int a_NodeIndex = 0xffffffff)
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
							L_n += p * l * Vs * tauToPhoton*d;
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

template<typename V, typename T> CUDA_FUNC_IN V sign(T f)
{
	return f > T(0) ? V(1) : (f < T(0) ? V(-1) : V(0));
}
template<bool VOL> CUDA_FUNC_IN Spectrum L_Volume2(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap, unsigned int a_NodeIndex = 0xffffffff)
{
	const k_PhotonMapReg& map = photonMap.m_sVolumeMap;
	Spectrum Tau = Spectrum(0.0f);
	float r2 = a_r * a_r;
	Spectrum L_n = Spectrum(0.0f);
	float rayT, maxT;
	AABB box = map.m_sHash.getAABB();
	if (!box.Intersect(r, &rayT, &maxT))
		return 0.0f;
	float minT = rayT = math::clamp(rayT, tmin, tmax);
	maxT = math::clamp(maxT, tmin, tmax);
	//pbrt grid accellerator copy! (slightly streamlined for SIMD)
	Vec3u Pos = map.m_sHash.Transform(r(rayT));
	Vec3i Step(sign<int>(r.direction.x), sign<int>(r.direction.y), sign<int>(r.direction.z));
	Vec3f inv_d = r.direction;
	const float ooeps = math::exp2(-80.0f);
	inv_d.x = 1.0f / (math::abs(inv_d.x) > ooeps ? inv_d.x : copysignf(ooeps, inv_d.x));
	inv_d.y = 1.0f / (math::abs(inv_d.y) > ooeps ? inv_d.y : copysignf(ooeps, inv_d.y));
	inv_d.z = 1.0f / (math::abs(inv_d.z) > ooeps ? inv_d.z : copysignf(ooeps, inv_d.z));
	Vec3f NextCrossingT = Vec3f(rayT) + (map.m_sHash.m_sBox.minV + (Vec3f(Pos.x, Pos.y, Pos.z) + max(Vec3f(0.0f), sign(r.direction))) * map.m_sHash.m_vCellSize - r(rayT)) * inv_d,
		  DeltaT = abs(map.m_sHash.m_vCellSize * inv_d);

	for (;;)
	{
		int bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) + ((NextCrossingT[0] < NextCrossingT[2]) << 1) + ((NextCrossingT[1] < NextCrossingT[2]));
		const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
		int stepAxis = cmpToAxis[bits];
		float localDist = NextCrossingT[stepAxis] - rayT;
#ifdef ISCUDA
		int beam_idx = map.m_sHash.Hash(Pos);
		if (beam_idx != -1)
			do
			{
				int2 beam = tex1Dfetch(t_Beams, beam_idx);
				beam_idx = beam.y;
				if (beam.x != -1)
				{
					k_pPpmPhoton e = loadPhoton(beam.x, photonMap);
					Vec3f wi = e.getWi(), P = e.getPos(map.m_sHash, Pos);
					//float r_p = half(e.accessNormalStorage()).ToFloat();
					float l1 = dot(P - r.origin, r.direction) / dot(r.direction, r.direction);
					if (distanceSquared(P, r(l1)) < r2 && rayT <= l1 && l1 <= NextCrossingT[stepAxis])
					{
						float p = VOL ? g_SceneData.m_sVolume.p(P, r.direction, wi, rng, a_NodeIndex) : Warp::squareToUniformSpherePdf();
						Spectrum tauToPhoton = VOL ? (-Tau - g_SceneData.m_sVolume.tau(r, rayT, l1)).exp() : (-sigt * (l1 - minT)).exp();
						L_n += p * e.getL() / (PI * photonMap.m_uPhotonNumEmitted * r2) * tauToPhoton;
					}
				}
			} while (beam_idx != -1);
		Spectrum tauD = VOL ? g_SceneData.m_sVolume.tau(r, rayT, NextCrossingT[stepAxis]) : sigt * localDist;
		Tau += tauD;
		if (VOL)
			L_n += g_SceneData.m_sVolume.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
#endif
		Pos[stepAxis] += Step[stepAxis];
		if (Pos[stepAxis] > map.m_sHash.m_fGridSize || maxT < NextCrossingT[stepAxis])
			break;
		rayT = NextCrossingT[stepAxis];
		NextCrossingT[stepAxis] += DeltaT[stepAxis];
	}
	Tr = (-Tau).exp();
	return L_n;
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
				unsigned int i0 = map.m_sHash.Hash(Vec3u(a, b, c)), i = map.m_pDeviceHashGrid[i0], NUM = 0;
				while (i != 0xffffffff && i != 0xffffff && NUM++ < MAX_PHOTONS_PER_CELL)
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
			}
	/*unsigned int hash_idx = map.m_sHash.Hash(bRec.dg.P);
	unsigned int list_idx = map.m_pDeviceHashGrid[hash_idx];
	while (list_idx != 0xffffffff)
	{
		uint2 list_entry = map.m_pDeviceLinkedList[list_idx];
		k_pPpmPhoton e = photonMap.m_pPhotons[list_entry.x];
		list_idx = list_entry.y;

		float3 n = e.getNormal(), wi = e.getWi(), P = e.getPos();
		Spectrum l = e.getL();
		float dist2 = DistanceSquared(P, bRec.dg.P);
		if (dist2 < r2 && dot(n, bRec.dg.sys.n) > 0.8f)
		{
			float ke = k_tr(a_rSurfaceUNUSED, math::sqrt(dist2));
			float dA = PI * r2;
			Lp += PI * ke * l / dA;
		}
	}*/
	return Lp / float(photonMap.m_uPhotonNumEmitted);
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, k_AdaptiveStruct& A, int idx,
	const Spectrum& importance, int a_PassIndex, float scale0, float scale1, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap)
{
	//Adaptive Progressive Photon Mapping Implementation
	k_AdaptiveEntry ent = A.E[idx];
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
	Spectrum Lp = 0.0f, gamma = mat->bsdf.f(bRec)	* INV_PI;//only diffuse //BUG f
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
				float psi = Spectrum(importance * gamma * l).getLuminance();
				if (dist2 < rd2)
				{
					const Vec3f e_l = bRec.dg.P - P;
					float aa = k_tr(rd, e_l + ur), ab = k_tr(rd, e_l - ur);
					float ba = k_tr(rd, e_l + vr), bb = k_tr(rd, e_l - vr);
					float cc = k_tr(rd, e_l);
					float laplu = psi / rd2 * (aa + ab - 2.0f * cc);
					float laplv = psi / rd2 * (ba + bb - 2.0f * cc);
					ent.I += laplu + laplv;
					ent.I2 += (laplu + laplv) * (laplu + laplv);
				}
				if (dist2 < r2)
				{
					float kri = k_tr(r, math::sqrt(dist2));
					Lp += kri * l * PI;
					ent.psi += psi;
					ent.psi2 += psi * psi;
					ent.pl += kri;
				}
			}
			i = e.getNext();
		}
	}
	/*
#define UPD(tar, val, pow) tar = scale0 * tar + scale1 * (pow == 1 ? val : val * val);
	UPD(ent.I, I_tmp, 1)
	UPD(ent.I2, I_tmp, 2)
	UPD(ent.psi, psi_tmp, 1)
	UPD(ent.psi2, psi_tmp, 2)
	UPD(ent.pl, pl_tmp, 1)
#undef UPD
	float VAR_Lapl = ent.I2 - ent.I * ent.I;
	float VAR_Phi = ent.psi2 - ent.psi * ent.psi;*/
	float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / NJ;

	if (VAR_Lapl)
	{
		ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(a_PassIndex, -1.0f / 8.0f);
		ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);
	}

	if (VAR_Lapl && VAR_Phi)
	{
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float ta = (2.0f * math::sqrt(VAR_Phi / float(photonMap.m_uPhotonNumEmitted))) / (PI * float(photonMap.m_uPhotonNumEmitted) * E_pl * k_22 * E_I * E_I);
		ent.r = math::pow(ta, 1.0f / 6.0f) * math::pow(a_PassIndex, -1.0f / 6.0f);
		ent.r = math::clamp(ent.r, A.r_min, A.r_max);
	}
	A.E[idx] = ent;

	//return Lp / (a_rSurfaceUNUSED * a_rSurfaceUNUSED);
	return L_Surface(bRec, ent.r, mat, photonMap, photonMap.m_sSurfaceMap);
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

template<bool DIRECT, bool FINAL_GATHER> CUDA_FUNC_IN void k_EyePassF(int x, int y, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage& img, const k_PhotonMapCollection<true, k_pPpmPhoton>& photonMap)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2f screenPos = Vec2f(x, y) + rng.randomFloat2();
	Ray r, rX, rY;
	Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());
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
				L += throughput * L_Volume<true>(a_rVolume, rng, r, tmin, tmax, Spectrum(0.0f), Spectrum(0.0f), Tr, photonMap);
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
			/*if (r2.getMat().bsdf.m_bReflectDirAtSurface)
				L += throughput * L_Volume<true>(a_rVolume, rng, rTrans, 0, r3.m_fDist, r2.getNodeIndex(), Spectrum(0.0f), Spectrum(0.0f), Tr, photonMap);
				else */L += throughput * L_Volume<false>(a_rVolume, rng, rTrans, 0, r3.m_fDist, bssrdf->sigp_s + bssrdf->sig_a, bssrdf->sigp_s, Tr, photonMap, r2.getNodeIndex());
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
			//L += throughput * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), A, y * w + x, throughput, a_PassIndex, scale0, scale1, photonMap);
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
			L += throughput * L_Volume<true>(a_rVolume, rng, r, tmin, tmax, Spectrum(0.0f), Spectrum(0.0f), Tr, photonMap);
		}
		L += throughput * g_SceneData.EvalEnvironment(r);
	}
	img.Add(screenPos.x, screenPos.y, L);
	//Spectrum qs;
	//float t = A.E[y * w + x].r / a_rSurfaceUNUSED;
	//t = (A.E[y * w + x].r - A.r_min) / (A.r_max - A.r_min);
	//qs.fromHSL(1.0f / 3.0f - t / 3.0f, 1, 0.5f);
	//g_Image.AddSample(screenPos.x, screenPos.y, qs);
	/*auto ent = A.E[y * w + x];
	float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / NJ;
	g_Image.AddSample(screenPos.x, screenPos.y, Spectrum(VAR_Phi*100));*/
	g_RNGData(rng);
}

template<bool DIRECT, bool FINAL_GATHER> __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage img, k_PhotonMapCollection<true, k_pPpmPhoton> photonMap)
{
	Vec2i pixel = k_TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		float radius2 = math::pow(math::pow(A.E[pixel.y * w + pixel.x].r, 2.0f) / math::pow(float(a_PassIndex), 0.5f * (1 - ALPHA)), 0.5f);
		k_EyePassF<DIRECT, FINAL_GATHER>(pixel.x, pixel.y, w, h, a_PassIndex, a_rSurfaceUNUSED, a_rVolume, A, img, photonMap);
		//img.img.AddSample(pixel.x, pixel.y, Spectrum(A.E[pixel.y * w + pixel.x].r / a_rSurfaceUNUSED));
	}
}

#define TN(r) (r * math::pow(float(m_uPassesDone), -1.0f/6.0f))
void k_sPpmTracer::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	cudaChannelFormatDesc cdu4 = cudaCreateChannelDesc<uint4>(), cdi2 = cudaCreateChannelDesc<int2>();
	size_t offset = 0;
	cudaError_t r = cudaBindTexture(&offset, &t_Photons, m_sMaps.m_pPhotons, &cdu4, m_sMaps.m_uPhotonNumStored * sizeof(uint4));
	r = cudaBindTexture(&offset, &t_Beams, m_sBeams.m_pDeviceData, &cdi2, m_sBeams.m_uNumEntries * sizeof(int2));

	float radius2 = getCurrentRadius2(2);
	float radius3 = getCurrentRadius2(3);

	//radius2 = radius3 = m_fInitialRadius * 10;
	k_AdaptiveStruct A(TN(r_min), TN(r_max), m_pEntries);
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
}

#include <Engine/e_DynamicScene.h>
void k_sPpmTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	StartNewTrace(I);
	L_Volume<true>(getCurrentRadius2(3), g_RNGData(), m_pScene->getCamera()->GenRay(pixel.x, pixel.y), 0, FLT_MAX, Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f), m_sMaps);
	//L_Volume<true>(getCurrentRadius2(3), g_RNGData(), Ray(Vec3f(39.592178f,41.220329f,3.317543f), Vec3f(-0.147530f,-0.893036f,0.425113f)), 0, 32.386997f, Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f), m_sMaps);
	//for (int i = 0; i < w; i++)
	//	for (int j = 0; j < h; j++)
	//		L_Volume<true>(r, g_RNGData(), m_pScene->getCamera()->GenRay(i, j), 0, FLT_MAX, Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f), m_sMaps);
	/*if(m_uPhotonsEmitted == (unsigned long long)-1)
		return;
	static k_AdaptiveEntry* hostEntries = 0;
	if (hostEntries == 0)
		hostEntries = new k_AdaptiveEntry[w * h];
	cudaMemcpy(hostEntries, m_pEntries, w * h * sizeof(k_AdaptiveEntry), cudaMemcpyDeviceToHost);
	k_AdaptiveStruct A(TN(r_min), TN(r_max), hostEntries);
	k_INITIALIZE(m_pScene, g_sRngs);
	k_PhotonMapCollection<true> map = m_sMaps;
	k_PhotonMapReg& map2 = map.m_sSurfaceMap;
	static k_pPpmPhoton* hostPhotons = 0;
	static unsigned int* hostGrid = 0;
	if (hostPhotons == 0)
	{
		hostPhotons = new k_pPpmPhoton[map.m_uPhotonBufferLength];
		hostGrid = new unsigned int[map2.m_uGridLength];
	}
	cudaMemcpy(hostPhotons, map.m_pPhotons, sizeof(k_pPpmPhoton)* map.m_uPhotonBufferLength, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostGrid, map2.m_pDeviceHashGrid, sizeof(unsigned int)* map2.m_uGridLength, cudaMemcpyDeviceToHost);
	map.m_pPhotons = hostPhotons;
	map2.m_pDeviceHashGrid = hostGrid;
	if (m_bDirect)
	{
		if (m_bFinalGather)
			k_EyePassF<true, true>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
		else k_EyePassF<true, false>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
	}
	else
	{
		if (m_bFinalGather)
			k_EyePassF<false, true>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
		else k_EyePassF<false, false>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
	}*/
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
	k_StartPass<<<dim3(w / p + 1, h / p + 1, 1), dim3(p,p,1)>>>(w, h, r, rd, m_pEntries);
}