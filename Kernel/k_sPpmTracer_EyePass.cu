#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"

CUDA_DEVICE k_PhotonMapCollection g_Map;

template<typename HASH> CUDA_ONLY_FUNC float3 k_PhotonMap<HASH>::L_Surface(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo) const
{
	Onb sys = bsdf->sys;
	sys.m_tangent *= a_r;
	sys.m_binormal *= a_r;
	sys.m_normal *= a_r;
	float3 a = -1.0f * sys.m_tangent - sys.m_binormal, b = sys.m_tangent - sys.m_binormal, c = -1.0f * sys.m_tangent + sys.m_binormal, d = sys.m_tangent + sys.m_binormal;
	float3 low = fminf(fminf(a, b), fminf(c, d)) + p, high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + p;
	//float3 a = p - sys.m_tangent, b = p + sys
	//float3 f0 = p - sys.m_tangent + sys.m_binormal, f1 = p + sys.m_tangent - sys.m_binormal;
	//float3 low = fminf(f0, f1 + sys.m_normal), high = fmaxf(f0, f1 + sys.m_normal);
	const float r2 = a_r * a_r, r3 = 1.0f / (r2 * a_NumPhotonEmitted), r4 = 1.0f / r2;
	float3 L = make_float3(0), Lr = make_float3(0), Lt = make_float3(0);
	//uint3 lo = m_sHash.Transform(p - make_float3(a_r)), hi = m_sHash.Transform(p + make_float3(a_r));
	uint3 lo = m_sHash.Transform(low), hi = m_sHash.Transform(high);
	//const bool glossy = bsdf->NumComponents(BxDFType(BSDF_ALL_TRANSMISSION | BSDF_ALL_REFLECTION | BSDF_GLOSSY));
	const bool glossy = false;
	for(int a = lo.x; a <= hi.x; a++)
		for(int b = lo.y; b <= hi.y; b++)
			for(int c = lo.z; c <= hi.z; c++)
			{
				unsigned int i0 = m_sHash.Hash(make_uint3(a,b,c)), i = m_pDeviceHashGrid[i0], q = 0;
				while(i != -1 && q++ < 1000)
				{
					k_pPpmPhoton e = m_pDevicePhotons[i];
					float3 nor = e.getNormal(), wi = e.getWi(), l = e.getL(), P = e.Pos;
					float dist2 = dot(P - p, P - p);
					if(dist2 < r2 && dot(nor, n) > 0.95f)
					{
						float s = 1.0f - dist2 * r4, k = 3.0f * INV_PI * s * s * r3;
						if(glossy)
							L += bsdf->f(wo, wi) * k * l;
						else if(dot(n, wi) > 0.0f)
							Lr += k * l;
						else Lt += k * l;
					}
					i = e.next;
				}
			}
	float buf[6 * 6 * 2];
	L += Lr * bsdf->rho(wo, rng, (unsigned char*)&buf, BSDF_ALL_REFLECTION)   * INV_PI +
		 Lt * bsdf->rho(wo, rng, (unsigned char*)&buf, BSDF_ALL_TRANSMISSION) * INV_PI;
	return L;
}

template<typename HASH> template<bool VOL> CUDA_ONLY_FUNC float3 k_PhotonMap<HASH>::L_Volume(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& sigt) const
{
	float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * a_NumPhotonEmitted), r2 = a_r * a_r;
	float3 L_n = make_float3(0);
	float a,b;
	if(!m_sHash.getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	a = clamp(a, tmin, tmax);
	b = clamp(b, tmin, tmax);
	float d = 8.0f * a_r;
	while(b > a)
	{
		float3 L = make_float3(0);
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
						float3 wi = e.getWi(), l = e.getL(), P = e.Pos;
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
			L_n = L * d + L_n * exp(-g_SceneData.m_sVolume.tau(r, b - d, b)) + g_SceneData.m_sVolume.Lve(x, -1.0f * r.direction) * d;
		else L_n = L * d + L_n * exp(sigt * -d);
		b -= d;
	}
	return L_n;
}

template<bool DIRECT> __global__ void k_EyePass(int2 off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	x += off.x; y += off.y;
	e_KernelBSDF bsdf;
	if(x < w && y < h)
	{
		CameraSample s = nextSample(x, y, rng);
		Ray ro = g_CameraData.GenRay(s, w, h);

		struct stackEntry
		{
			Ray r;
			float3 fs;
			unsigned int d;
			CUDA_FUNC_IN stackEntry(){}
			CUDA_FUNC_IN stackEntry(Ray _r, float3 _fs, unsigned int _d)
			{
				r = _r;
				fs = _fs;
				d = _d;
			}
		};
		float3 L = make_float3(0);
		const unsigned int stackN = 16;
		stackEntry stack[stackN];
		stack[0] = stackEntry(ro, make_float3(1), 0);
		unsigned int stackPos = 1;
		while(stackPos)
		{
			stackEntry s = stack[--stackPos];
			TraceResult r2;
			r2.Init();
			if(k_TraceRay<true>(s.r.direction, s.r.origin, &r2))
			{
				float3 p = s.r(r2.m_fDist);
				r2.GetBSDF(p, &bsdf);
				
				if(g_SceneData.m_sVolume.HasVolumes())
				{
					float tmin, tmax;
					g_SceneData.m_sVolume.IntersectP(s.r, 0, r2.m_fDist, &tmin, &tmax);
					L += s.fs * g_Map.L<true>(a_rVolume, rng, s.r, tmin, tmax, make_float3(0));
					s.fs = s.fs * exp(-g_SceneData.m_sVolume.tau(s.r, tmin, tmax));
				}
				
				if(DIRECT)
					L += s.fs * UniformSampleAllLights(p, bsdf.sys.m_normal, -s.r.direction, &bsdf, rng, 4);
				L += s.fs * r2.Le(p, bsdf.sys.m_normal, -s.r.direction);
				e_KernelBSSRDF bssrdf;
				if(r2.m_pTri->GetBSSRDF(p, r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset, &bssrdf))
				{
					float3 dir = refract(s.r.direction, bsdf.sys.m_normal, 1.0f / bssrdf.e);
					TraceResult r3;
					r3.Init();
					k_TraceRay<true>(dir, p, &r3);
					L += s.fs * g_Map.L<false>(a_rVolume, rng, Ray(p, dir), 0, r3.m_fDist, bssrdf.sigp_s + bssrdf.sig_a);
				}
				if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)))
					L += s.fs * g_Map.L(a_rSurface, rng, &bsdf, bsdf.sys.m_normal, p, -s.r.direction);
					//L += s.fs * UniformSampleAllLights(p, bsdf.sys.m_normal, -s.r.direction, &bsdf, rng, 16);
				if(s.d < 5 && stackPos < stackN - 1)
				{
					float3 r_wi;
					float r_pdf;
					float3 r_f = bsdf.Sample_f(-s.r.direction, &r_wi, BSDFSample(rng), &r_pdf, BxDFType(BSDF_REFLECTION | BSDF_SPECULAR | BSDF_GLOSSY));
					if(r_pdf && !ISBLACK(r_f))
						stack[stackPos++] = stackEntry(Ray(p, r_wi), bsdf.IntegratePdf(r_f, r_pdf, r_wi) * s.fs, s.d + 1);
					float3 t_wi;
					float t_pdf;
					float3 t_f = bsdf.Sample_f(-s.r.direction, &t_wi, BSDFSample(rng), &t_pdf, BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR | BSDF_GLOSSY));
					if(t_pdf && !ISBLACK(t_f))
						stack[stackPos++] = stackEntry(Ray(p, t_wi), bsdf.IntegratePdf(t_f, t_pdf, t_wi) * s.fs, s.d + 1);
				}
			}
			else if(g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				g_SceneData.m_sVolume.IntersectP(s.r, 0, r2.m_fDist, &tmin, &tmax);
				L += s.fs * g_Map.L<true>(a_rVolume, rng, s.r, tmin, tmax, make_float3(0));
				if(g_SceneData.m_sEnvMap.CanSample() && !s.d)
					L += s.fs * g_SceneData.m_sEnvMap.Sample(s.r);
			}
			else if(g_SceneData.m_sEnvMap.CanSample() && s.d == 0)
					L += s.fs * g_SceneData.m_sEnvMap.Sample(s.r);
		}
		g_Image.AddSample(s, L);
	}
	g_RNGData(rng);
}

void k_sPpmTracer::doEyePass(e_Image* I)
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASSI(m_pScene, m_pCamera, m_sRngs, *I);
	if(m_pScene->getVolumes().getLength() || m_bLongRunning)
	{
		unsigned int p = 16, q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				if(m_bDirect)
					k_EyePass<true><<<dim3( q, q, 1), dim3(p, p, 1)>>>(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
				else k_EyePass<false><<<dim3( q, q, 1), dim3(p, p, 1)>>>(make_int2(pq * i, pq * j), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
	}
	else
	{
		const unsigned int p = 16;
		if(m_bDirect)
			k_EyePass<true><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
		else k_EyePass<false><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
	}
}

void k_sPpmTracer::Debug(int2 pixel)
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	const unsigned int p = 16;
	k_EyePass<true><<<1, 1>>>(pixel, w, h, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
}