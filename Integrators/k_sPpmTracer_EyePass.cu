#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_FUNC_IN float k(float t)
{
	//float t2 = t * t;
	//return 1.0f + t2 * t * (-6.0f * t2 + 15.0f * t - 10.0f);
	return 1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f);
}

CUDA_FUNC_IN float k_tr(float r , float t)
{
	//if (t > r)
	//	printf("t : %f, r : %f", t, r);
	return k(t / r);
}

CUDA_FUNC_IN float k_tr(float r, const float3& t)
{
	return k_tr(r, length(t));
}

template<bool VOL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt, const Spectrum& sigs, Spectrum& Tr, const k_PhotonMapCollection& photonMap)
{
	Spectrum Tau = Spectrum(0.0f);
	float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * photonMap.m_uPhotonNumEmitted), r2 = a_r * a_r;
	Spectrum L_n = Spectrum(0.0f);
	float a,b;
	if (!photonMap.m_sVolumeMap.m_sHash.getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	a = clamp(a, tmin, tmax);
	b = clamp(b, tmin, tmax);
	float d = 2.0f * a_r;
	b -= d / 2.0f;
	while(b > a)
	{
		Spectrum L = Spectrum(0.0f);
		float3 x = r(b);
		uint3 lo = photonMap.m_sVolumeMap.m_sHash.Transform(x - make_float3(a_r)), hi = photonMap.m_sVolumeMap.m_sHash.Transform(x + make_float3(a_r));
		for(unsigned int ac = lo.x; ac <= hi.x; ac++)
			for(unsigned int bc = lo.y; bc <= hi.y; bc++)
				for(unsigned int cc = lo.z; cc <= hi.z; cc++)
				{
					unsigned int i0 = photonMap.m_sVolumeMap.m_sHash.Hash(make_uint3(ac, bc, cc)), i = photonMap.m_sVolumeMap.m_pDeviceHashGrid[i0];
					while(i != 0xffffffff)
					{
						k_pPpmPhoton e = photonMap.m_pPhotons[i];
						float3 wi = e.getWi(), P = e.getPos();
						Spectrum l = e.getL();
						if(DistanceSquared(P, x) < r2)
						{
							float p = VOL ? g_SceneData.m_sVolume.p(x, r.direction, wi, rng) : Warp::squareToUniformSpherePdf();
							L += p * l * Vs;
						}
						i = e.next;
					}
				}
		Spectrum tauDelta = VOL ? g_SceneData.m_sVolume.tau(r, b - d, b) : sigt * d;
		Tau += tauDelta;
		Spectrum o_s = VOL ? g_SceneData.m_sVolume.sigma_s(x, -r.direction) : sigs;
		L_n = L * d + L_n * (-tauDelta).exp() + (VOL ? g_SceneData.m_sVolume.Lve(x, -1.0f * r.direction) * d : Spectrum(0.0f));
		b -= d;
	}
	Tr = (-Tau).exp();
	return L_n;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, const k_PhotonMapCollection& photonMap)
{
	Frame sys = bRec.dg.sys;
	sys.t *= a_rSurfaceUNUSED;
	sys.s *= a_rSurfaceUNUSED;
	sys.n *= a_rSurfaceUNUSED;
	float3 a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	float3 low = fminf(fminf(a, b), fminf(c, d)) + bRec.dg.P, high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + bRec.dg.P;
	Spectrum Lp = Spectrum(0.0f);
	uint3 lo = photonMap.m_sSurfaceMap.m_sHash.Transform(low), hi = photonMap.m_sSurfaceMap.m_sHash.Transform(high);
	const float r2 = a_rSurfaceUNUSED * a_rSurfaceUNUSED;
	for(unsigned int a = lo.x; a <= hi.x; a++)
		for(unsigned int b = lo.y; b <= hi.y; b++)
			for(unsigned int c = lo.z; c <= hi.z; c++)
			{
				unsigned int i0 = photonMap.m_sSurfaceMap.m_sHash.Hash(make_uint3(a, b, c)), i = photonMap.m_sSurfaceMap.m_pDeviceHashGrid[i0];
				while (i != 0xffffffff)
				{
					k_pPpmPhoton e = photonMap.m_sSurfaceMap.m_pDevicePhotons[i];
					float3 n = e.getNormal(), wi = e.getWi(), P = e.getPos();
					Spectrum l = e.getL();
					float dist2 = DistanceSquared(P, bRec.dg.P);
					if (dist2 < r2 && dot(n, bRec.dg.sys.n) > 0.8f)
					{
						float ke = k_tr(a_rSurfaceUNUSED, sqrtf(dist2));
						float dA = PI * r2;
						Lp += PI * ke * l / dA;
					}
					i = e.next;
				}
			}
	return Lp / float(photonMap.m_uPhotonNumEmitted) * mat->bsdf.getDiffuseReflectance(bRec) * INV_PI;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, k_AdaptiveStruct& A, int idx,
	const Spectrum& importance, int a_PassIndex, float scale0, float scale1, const k_PhotonMapCollection& photonMap)
{
	//Adaptive Progressive Photon Mapping Implementation
	k_AdaptiveEntry ent = A.E[idx];
	float r2 = ent.r * ent.r, maxr = MAX(ent.r, ent.rd), rd2 = ent.rd * ent.rd, rd = ent.rd;
	Frame sys = bRec.dg.sys;
	sys.t *= maxr;
	sys.s *= maxr;
	sys.n *= maxr;
	float3 ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	float3 a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	float3 low = fminf(fminf(a, b), fminf(c, d)) + bRec.dg.P, high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + bRec.dg.P;
	uint3 lo = photonMap.m_sSurfaceMap.m_sHash.Transform(low), hi = photonMap.m_sSurfaceMap.m_sHash.Transform(high);
	Spectrum Lp = make_float3(0), gamma = mat->bsdf.getDiffuseReflectance(bRec)	* INV_PI;//only diffuse
	for (int a = lo.x; a <= hi.x; a++)
	for (int b = lo.y; b <= hi.y; b++)
	for (int c = lo.z; c <= hi.z; c++)
	{
		unsigned int i0 = photonMap.m_sSurfaceMap.m_sHash.Hash(make_uint3(a, b, c)), i = photonMap.m_sSurfaceMap.m_pDeviceHashGrid[i0];
		while (i != 0xffffffff)
		{
			k_pPpmPhoton e = photonMap.m_pPhotons[i];
			float3 nor = e.getNormal(), wi = e.getWi(), P = e.getPos();
			Spectrum l = e.getL();
			float dist2 = DistanceSquared(P, bRec.dg.P);
			if (dot(nor, bRec.dg.sys.n) > 0.95f)
			{
				bRec.wo = bRec.dg.toLocal(wi);
				float psi = Spectrum(importance * gamma * l).getLuminance();
				if (dist2 < rd2)
				{
					const float3 e_l = bRec.dg.P - P;
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
					float kri = k_tr(ent.r, sqrtf(dist2));
					Lp += kri * l;
					ent.psi += kri * psi;
					ent.psi2 += (kri * psi) * (kri * psi);
					ent.pl += kri;
				}
			}
			i = e.next;
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

	/*if (VAR_Lapl)
	{
		ent.rd = 1.9635f * sqrtf(VAR_Lapl) * powf(a_PassIndex, -1.0f / 8.0f);
		ent.rd = clamp(ent.rd, A.r_min, A.r_max);
	}

	if (VAR_Lapl && VAR_Phi)
	{
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float ta = (2.0f * sqrtf(VAR_Phi)) / (PI * float(photonMap.m_uPhotonNumEmitted) * E_pl * k_22 * E_I * E_I);
		ent.r = powf(ta, 1.0f / 6.0f) * powf(a_PassIndex, -1.0f / 6.0f);
		ent.r = clamp(ent.r, A.r_min, A.r_max);
	}*/
	A.E[idx] = ent;

	//return Lp / (float(photonMap.m_uPhotonNumEmitted) * a_rSurfaceUNUSED * a_rSurfaceUNUSED);
	return L_Surface(bRec, ent.r, mat, photonMap);
}

CUDA_FUNC_IN Spectrum L_FinalGathering(TraceResult& r2, BSDFSamplingRecord& bRec, CudaRNG& rng, float a_rSurfaceUNUSED, const k_PhotonMapCollection& photonMap)
{
	Spectrum L(0.0f);
	const int N = 3;
	for (int i = 0; i < N; i++)
	{
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		Ray r(bRec.dg.P, bRec.getOutgoing());
		TraceResult r3 = k_TraceRay(r);
		if (r3.hasHit())
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec2(dg);
			r3.getBsdfSample(r, rng, &bRec2);
			Spectrum dif = r3.getMat().bsdf.getDiffuseReflectance(bRec2) / PI;
			L += f * L_Surface(bRec, a_rSurfaceUNUSED, &r3.getMat(), photonMap) * dif;
		}
	}
	return L / float(N);
}

template<bool DIRECT> CUDA_FUNC_IN void k_EyePassF(int x, int y, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, float scale0, float scale1, e_Image g_Image, const k_PhotonMapCollection& photonMap)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	float2 screenPos = make_float2(x, y) + rng.randomFloat2();
	Ray r, rX, rY;
	Spectrum importance = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());
	TraceResult r2;
	r2.Init();
	int depth = -1;
	Spectrum L(0.0f), throughput(1.0f);
	while(k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
	{
		r2.getBsdfSample(r, rng, &bRec);
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
			L += throughput * UniformSampleAllLights(bRec, r2.getMat(), 1);
		L += throughput * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);//either it's the first bounce -> account or it's a specular reflection -> ...
		const e_KernelBSSRDF* bssrdf;
		if(r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
		{
			Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			bRec.wo.z *= -1.0f;
			Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
			TraceResult r3 = k_TraceRay(rTrans);
			Spectrum Tr;
			L += throughput * L_Volume<false>(a_rVolume, rng, rTrans, 0, r3.m_fDist, bssrdf->sigp_s + bssrdf->sig_a, bssrdf->sigp_s, Tr, photonMap);
			break;
		}
		bool hasDiffuse = r2.getMat().bsdf.hasComponent(EDiffuse), hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy);
		if(hasDiffuse)
		{
			L += throughput * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), photonMap);
			//L += throughput * L_FinalGathering(r2, bRec, rng, a_rSurfaceUNUSED, photonMap);
			//L += throughput * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), A, y * w + x, throughput, a_PassIndex, scale0, scale1, photonMap);
			if(!hasSpecGlossy)
				break;
		}
		if(hasSpecGlossy)
		{
			bRec.sampledType = 0;
			bRec.typeMask = EDelta | EGlossy;
			Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			if(!bRec.sampledType)
				break;
			throughput = throughput * t_f;
			r = Ray(bRec.dg.P, bRec.getOutgoing());
			r2.Init();
		}
		//else break;
	}
	if(!r2.hasHit())
	{
		/*if(g_SceneData.m_sVolume.HasVolumes())
		{
			float tmin, tmax;
			g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax);
			L += throughput * g_Map2.L<true>(a_rVolume, rng, r, tmin, tmax, make_float3(0));
		}*/
		L += throughput * g_SceneData.EvalEnvironment(r);
	}
	g_Image.AddSample(screenPos.x, screenPos.y, importance * L);
	//Spectrum qs;
	//float t = A.E[y * w + x].r / a_rSurfaceUNUSED;
	//t = (A.E[y * w + x].r - A.r_min) / (A.r_max - A.r_min);
	//qs.fromHSL(1.0f / 3.0f - t / 3.0f, 1, 0.5f);
	/*auto ent = A.E[y * w + x];
	float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / NJ;
	g_Image.SetSample(x, y, Spectrum(VAR_Lapl*0.0001f).toRGBCOL());*/
	g_RNGData(rng);
}

template<bool DIRECT> __global__ void k_EyePass(int2 off, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, float scale0, float scale1, e_Image g_Image, k_PhotonMapCollection photonMap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	x += off.x;
	y += off.y;
	if (x < w && y < h)
		k_EyePassF<DIRECT>(x, y, w, h, a_PassIndex, a_rSurfaceUNUSED, a_rVolume, A, scale0, scale1, g_Image, photonMap);
}

#define TN(r) (r * powf(float(m_uPassesDone), -1.0f/6.0f))
void k_sPpmTracer::doEyePass(e_Image* I)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	float s1 = float(m_uPassesDone - 1) / float(m_uPassesDone), s2 = 1.0f / float(m_uPassesDone);
	k_AdaptiveStruct A(TN(r_min), TN(r_max), m_pEntries);
	if(m_pScene->getVolumes().getLength() || m_bLongRunning || w * h > 800 * 800)
	{
		unsigned int q = 8, p = 16, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				if(m_bDirect)
					k_EyePass<true> << <dim3(q, q, 1), dim3(p, p, 1) >> >(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, m_sMaps);
				else k_EyePass<false> << <dim3(q, q, 1), dim3(p, p, 1) >> >(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, m_sMaps);
	}
	else
	{
		const unsigned int p = 16;
		if(m_bDirect)
			k_EyePass<true> << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(make_int2(0, 0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, m_sMaps);
		else k_EyePass<false> << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(make_int2(0, 0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, m_sMaps);
	}
	//Debug(I, make_int2(269, 158));
	I->DoUpdateDisplay(0);
}

void k_sPpmTracer::Debug(e_Image* I, int2 pixel)
{
	if(m_uPhotonsEmitted == (unsigned long long)-1)
		return;
	static k_AdaptiveEntry* hostEntries = 0;
	if (hostEntries == 0)
		hostEntries = new k_AdaptiveEntry[w * h];
	cudaMemcpy(hostEntries, m_pEntries, w * h * sizeof(k_AdaptiveEntry), cudaMemcpyDeviceToHost);
	k_AdaptiveStruct A(TN(r_min), TN(r_max), hostEntries);
	float s1 = float(m_uPassesDone - 1) / float(m_uPassesDone), s2 = 1.0f / float(m_uPassesDone);
	k_INITIALIZE(m_pScene, g_sRngs);
	k_PhotonMapCollection map = m_sMaps;
	static k_pPpmPhoton* hostPhotons = 0;
	static unsigned int* hostGrid = 0;
	if (hostPhotons == 0)
	{
		hostPhotons = new k_pPpmPhoton[map.m_uPhotonBufferLength];
		hostGrid = new unsigned int[map.m_sSurfaceMap.m_uGridLength];
	}
	cudaMemcpy(hostPhotons, map.m_pPhotons, sizeof(k_pPpmPhoton)* map.m_uPhotonBufferLength, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostGrid, map.m_sSurfaceMap.m_pDeviceHashGrid, sizeof(unsigned int)* map.m_sSurfaceMap.m_uGridLength, cudaMemcpyDeviceToHost);
	map.m_pPhotons = hostPhotons;
	map.m_sSurfaceMap.m_pDeviceHashGrid = hostGrid;
	map.m_sSurfaceMap.m_pDevicePhotons = hostPhotons;
	if (m_bDirect)
		k_EyePassF<true>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
	else k_EyePassF<false>(pixel.x, pixel.y, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, s1, s2, *I, map);
}

__global__ void k_StartPass(int w, int h, float r, float rd, k_AdaptiveEntry* E)
{
	int i = threadId, x = i % w, y = i / w;
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