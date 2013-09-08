#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"

/*{
					k_AdaptiveEntry ent = A.E[y * w + x];
					float r2 = ent.r * ent.r, maxr = MAX(ent.r, ent.rd), rd2 = ent.rd * ent.rd, rd = ent.rd;
					Frame sys = bsdf.sys;
					sys.t *= maxr;
					sys.s *= maxr;
					sys.n *= maxr;
					float3 ud = bsdf.sys.t * rd, vd = bsdf.sys.s * rd;
					float3 a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
					float3 low = fminf(fminf(a, b), fminf(c, d)) + p, high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + p;
					uint3 lo = g_Map2.m_sSurfaceMap.m_sHash.Transform(low), hi = g_Map2.m_sSurfaceMap.m_sHash.Transform(high);
					float3 Lp = make_float3(0);
					for(int a = lo.x; a <= hi.x; a++)
						for(int b = lo.y; b <= hi.y; b++)
							for(int c = lo.z; c <= hi.z; c++)
							{
								unsigned int i0 = g_Map2.m_sSurfaceMap.m_sHash.Hash(make_uint3(a,b,c)), i = g_Map2.m_sSurfaceMap.m_pDeviceHashGrid[i0], q = 0;
								while(i != -1)
								{
									k_pPpmPhoton e = g_Map2.m_pPhotons[i];
									float3 nor = e.getNormal(), wi = e.getWi(), l = e.getL(), P = e.Pos;
									float dist2 = dot(P - p, P - p);
									if(dot(nor, bsdf.sys.n) > 0.95f)
									{
										float3 gamma = bsdf.f(-s.r.direction, wi);
										float psi = ::y(gamma * l);
										if(dist2 < rd2)
										{
											float aa = k_tr(rd, p - P + ud), ab = k_tr(rd, p - P - ud);
											float ba = k_tr(rd, p - P + vd), bb = k_tr(rd, P - P - vd);
											float cc = k_tr(rd, p - P);
											float laplu = psi / rd2 * (aa + ab - 2.0f * cc);
											float laplv = psi / rd2 * (ba + bb - 2.0f * cc);
											ent.I += laplu + laplv;
											ent.I2 += (laplu + laplv) * (laplu + laplv);
										}
										if(dist2 < r2)
										{
											float kri = k_tr(ent.r, sqrtf(dist2));
											Lp += kri * gamma * l;
											ent.psi += kri * psi;
											ent.psi2 += (kri * psi) * (kri * psi);
											ent.pl += kri;
										}
									}
									i = e.next;
								}
							}

					float VAR_Lapl = (ent.I / NumAllPhotons * ent.I / NumAllPhotons - ent.I2 / NumAllPhotons);
					if(VAR_Lapl) {
						ent.rd = 1.9635f * sqrtf(VAR_Lapl) * powf(a_PassIndex, -1.0f / 8.0f);
						ent.rd = clamp(ent.rd, A.r_min, A.r_max);
					}
					float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
					float VAR_Phi = sqrtf(ent.psi / NumAllPhotons * ent.psi / NumAllPhotons - ent.psi2 / NumAllPhotons);
					float ta = (2.0f * VAR_Phi) / (PI * float(g_Map2.m_uPhotonNumEmitted) * ent.pl / NumAllPhotons * k_22 * ent.I2 / NumAllPhotons);
					if(VAR_Lapl && VAR_Phi) {
						ent.r = powf(ta, 1.0f / 6.0f) * powf(a_PassIndex, -1.0f / 6.0f);
						ent.r = clamp(ent.r, A.r_min, A.r_max);	
					}
					A.E[y * w + x] = ent;
					Lp = ent.r != A.r_max ? make_float3(1,0,0) : make_float3(0);

					L += s.fs * Lp / float(g_Map2.m_uPhotonNumEmitted);
				}*/

CUDA_DEVICE k_PhotonMapCollection g_Map2;

CUDA_FUNC_IN float k(float t)
{
	float t2 = t * t;
	return 1.0f + t2 * t * (-6.0f * t2 + 15.0f * t - 10.0f);
}

CUDA_FUNC_IN float k_tr(float r , float t)
{
	float ny = 1.0f / (PI * r * r);
	return ny * k(t / r);
}

CUDA_FUNC_IN float k_tr(float r, const float3& t)
{
	return k_tr(r, length(t));
}

template<typename HASH> template<bool VOL> CUDA_ONLY_FUNC Spectrum k_PhotonMap<HASH>::L_Volume(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const Ray& r, float tmin, float tmax, const Spectrum& sigt) const
{
	float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * a_NumPhotonEmitted), r2 = a_r * a_r;
	Spectrum L_n = Spectrum(0.0f);
	float a,b;
	if(!m_sHash.getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	a = clamp(a, tmin, tmax);
	b = clamp(b, tmin, tmax);
	float d = 8.0f * a_r;
	while(b > a)
	{
		Spectrum L = Spectrum(0.0f);
		float3 x = r(b);
		uint3 lo = m_sHash.Transform(x - make_float3(a_r)), hi = m_sHash.Transform(x + make_float3(a_r));
		for(unsigned int ac = lo.x; ac <= hi.x; ac++)
			for(unsigned int bc = lo.y; bc <= hi.y; bc++)
				for(unsigned int cc = lo.z; cc <= hi.z; cc++)
				{
					unsigned int i0 = m_sHash.Hash(make_uint3(ac,bc,cc)), i = m_pDeviceHashGrid[i0];
					while(i != -1)
					{
						k_pPpmPhoton e = m_pDevicePhotons[i];
						float3 wi = e.getWi(), P = e.Pos;
						Spectrum l = e.getL();
						if(dot(P - x, P - x) < r2)
						{
							float p;
							if(VOL)
								p = g_SceneData.m_sVolume.p(x, -1.0f * r.direction, r.direction, rng);
							else p = 1.f / (4.f * PI);
							L += p * l * Vs;
						}
						i = e.next;
					}
				}
		if(VOL)
			L_n = L * d + L_n * (-g_SceneData.m_sVolume.tau(r, b - d, b)).exp() + g_SceneData.m_sVolume.Lve(x, -1.0f * r.direction) * d;
		else L_n = L * d + L_n * (sigt * -d).exp();
		b -= d;
	}
	return L_n;
}

template<bool DIRECT> __global__ void k_EyePass(int2 off, int w, int h, float a_PassIndex, float a_rSurfaceUNUSED, float a_rVolume, k_AdaptiveStruct A, float NumAllPhotons, e_Image g_Image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	x += off.x; y += off.y;
	BSDFSamplingRecord bRec;
	if(x < w && y < h)
	{
		CameraSample samp = nextSample(x, y, rng);
		Ray r = g_CameraData.GenRay(samp, w, h);
		TraceResult r2;
		r2.Init(true);
		int depth = -1;
		Spectrum L(0.0f), throughput(1.0f);
		while(depth++ < 7)
		{
			r2.Init();
			if(k_TraceRay(r.direction, r.origin, &r2))
			{
				float3 p = r(r2.m_fDist);
				r2.getBsdfSample(r, rng, &bRec);
				if(g_SceneData.m_sVolume.HasVolumes())
				{
					float tmin, tmax;
					g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax);
					L += throughput * g_Map2.L<true>(a_rVolume, rng, r, tmin, tmax, make_float3(0));
					throughput = throughput * (-g_SceneData.m_sVolume.tau(r, tmin, tmax)).exp();
				}
				if(DIRECT)
					L += throughput * UniformSampleAllLights(bRec, r2.getMat(), 4);
				L += throughput * r2.Le(p, bRec.map.sys.n, -r.direction);
				e_KernelBSSRDF* bssrdf;
				if(r2.getMat().GetBSSRDF(bRec.map, &bssrdf))
				{
					float3 dir = VectorMath::refract(r.direction, bRec.map.sys.n, 1.0f / bssrdf->e);
					TraceResult r3;
					r3.Init();
					k_TraceRay(dir, p, &r3);
					L += throughput * g_Map2.L<false>(a_rVolume, rng, Ray(p, dir), 0, r3.m_fDist, bssrdf->sigp_s + bssrdf->sig_a);
				}
				if(r2.getMat().bsdf.hasComponent(EDiffuse))
				{
					Frame sys = bRec.map.sys;
					sys.t *= a_rSurfaceUNUSED;
					sys.s *= a_rSurfaceUNUSED;
					sys.n *= a_rSurfaceUNUSED;
					float3 a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
					float3 low = fminf(fminf(a, b), fminf(c, d)) + p, high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + p;
					Spectrum Lp = Spectrum(0.0f);
					uint3 lo = g_Map2.m_sSurfaceMap.m_sHash.Transform(low), hi = g_Map2.m_sSurfaceMap.m_sHash.Transform(high);
					const bool glossy = false;
					for(int a = lo.x; a <= hi.x; a++)
						for(int b = lo.y; b <= hi.y; b++)
							for(int c = lo.z; c <= hi.z; c++)
							{
								unsigned int i0 = g_Map2.m_sSurfaceMap.m_sHash.Hash(make_uint3(a,b,c)), i = g_Map2.m_sSurfaceMap.m_pDeviceHashGrid[i0];
								while(i != -1)
								{
									k_pPpmPhoton e = g_Map2.m_sSurfaceMap.m_pDevicePhotons[i];
									float3 nor = e.getNormal(), wi = e.getWi(), P = e.Pos;
									Spectrum l = e.getL();
									float dist2 = dot(P - p, P - p);
									if(dist2 < a_rSurfaceUNUSED * a_rSurfaceUNUSED && dot(nor, bRec.map.sys.n) > 0.95f)
									{
										bRec.wo = bRec.map.sys.toLocal(wi);
										Spectrum gamma = r2.getMat().bsdf.f(bRec);
										Lp += k_tr(a_rSurfaceUNUSED, sqrtf(dist2)) * gamma * l;
									}
									i = e.next;
								}
							}
					L += throughput * Lp / float(g_Map2.m_uPhotonNumEmitted);
					break;
				}
				bRec.typeMask = EDelta | EGlossy;
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if(bRec.sampledComponent == -1)
					break;
				throughput = throughput * t_f;
				r = Ray(p, bRec.getOutgoing());
			}
			else if(g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax);
				L += throughput * g_Map2.L<true>(a_rVolume, rng, r, tmin, tmax, make_float3(0));
				if(g_SceneData.m_sEnvMap.CanSample())
					L += throughput * g_SceneData.m_sEnvMap.Sample(r);
			}
			else if(g_SceneData.m_sEnvMap.CanSample())
					L += throughput * g_SceneData.m_sEnvMap.Sample(r);
		}
		g_Image.AddSample(samp, L);
	}
	g_RNGData(rng);
}

#define TN(r) (r * powf(float(m_uPassesDone), -1.0f/6.0f))
void k_sPpmTracer::doEyePass(e_Image* I)
{
	k_AdaptiveStruct A(TN(r_min), TN(r_max), m_pEntries);
	cudaMemcpyToSymbol(g_Map2, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	if(m_pScene->getVolumes().getLength() || m_bLongRunning)
	{
		unsigned int p = 16, q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				if(m_bDirect)
					k_EyePass<true><<<dim3( q, q, 1), dim3(p, p, 1)>>>(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, m_uPhotonsEmitted, *I);
				else k_EyePass<false><<<dim3( q, q, 1), dim3(p, p, 1)>>>(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, m_uPhotonsEmitted, *I);
	}
	else
	{
		const unsigned int p = 16;
		if(m_bDirect)
			k_EyePass<true><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, m_uPhotonsEmitted, *I);
		else k_EyePass<false><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, m_uPhotonsEmitted, *I);
	}
}

void k_sPpmTracer::Debug(int2 pixel)
{
	k_AdaptiveStruct A(TN(r_min), TN(r_max), m_pEntries);
	cudaMemcpyToSymbol(g_Map2, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	k_EyePass<true><<<1, 1>>>(pixel, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3), A, m_uPhotonsEmitted, e_Image());
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