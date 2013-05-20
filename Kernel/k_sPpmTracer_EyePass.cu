#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"

//texture<uint4, 1> t_PhotonTex;
//texture<unsigned int, 1> t_HashTex;

void k_PhotonMapCollection::StartNewRendering(const AABB& sbox, const AABB& vbox, float a_R)
{
	//cudaChannelFormatDesc cd0 = cudaCreateChannelDesc<uint4>();
	//cudaBindTexture(0, &t_PhotonTex, m_pPhotons, &cd0, m_uPhotonBufferLength * sizeof(k_pPpmPhoton));
	m_sVolumeMap.StartNewRendering(vbox, a_R);
	m_sSurfaceMap.StartNewRendering(sbox, a_R);
}

template<typename HASH> void k_PhotonMap<HASH>::StartNewRendering(const AABB& box, float a_InitRadius)
{
	m_sHash = HASH(box, a_InitRadius, m_uGridLength);
	cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int) * m_uGridLength);

	//cudaChannelFormatDesc cd1 = cudaCreateChannelDesc<unsigned int>();		
	//cudaBindTexture(0, &t_HashTex, m_pDeviceHashGrid, &cd1, m_uGridLength * sizeof(unsigned int));
}

template<typename HASH> CUDA_ONLY_FUNC float3 k_PhotonMap<HASH>::L_Surface(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo) const
{
	Onb sys(n);
	sys.m_tangent *= a_r;
	sys.m_binormal *= a_r;
	float3 low = fminf(p - sys.m_tangent + sys.m_binormal, p + sys.m_tangent - sys.m_binormal), high = fmaxf(p - sys.m_tangent + sys.m_binormal, p + sys.m_tangent - sys.m_binormal);
	const float r2 = a_r * a_r, r3 = 1.0f / (r2 * a_NumPhotonEmitted), r4 = 1.0f / r2;
	float3 L = make_float3(0), Lr = make_float3(0), Lt = make_float3(0);
	//uint3 lo = m_sHash.Transform(p - make_float3(a_r)), hi = m_sHash.Transform(p + make_float3(a_r));
	uint3 lo = m_sHash.Transform(low), hi = m_sHash.Transform(high);
	const bool glossy = bsdf->NumComponents(BxDFType(BSDF_ALL_TRANSMISSION | BSDF_ALL_REFLECTION | BSDF_GLOSSY));
	for(int a = lo.x; a <= hi.x; a++)
		for(int b = lo.y; b <= hi.y; b++)
			for(int c = lo.z; c <= hi.z; c++)
			{
				unsigned int i0 = m_sHash.Hash(make_uint3(a,b,c)), i = m_pDeviceHashGrid[i0], q = 0;//tex1Dfetch(t_HashTex, i0)
				while(i != -1 && q++ < 1000)
				{
					k_pPpmPhoton e = m_pDevicePhotons[i];
					//k_pPpmPhoton e(tex1Dfetch(t_PhotonTex, i));
					float3 nor = e.getNormal(), wi = e.getWi(), l = e.getL(), P = e.Pos;//m_sHash.DecodePos(e.Pos, make_uint3(a,b,c))
					float dist2 = dot(P - p, P - p);
					if(dist2 < r2 && AbsDot(nor, n) > 0.95f)//
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

CUDA_DEVICE k_PhotonMapCollection g_Map;

template<typename HASH> template<bool VOL> CUDA_ONLY_FUNC float3 k_PhotonMap<HASH>::L_Volume(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& sigt) const
{
	float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * a_NumPhotonEmitted), r2 = a_r * a_r;
	float3 L_n = make_float3(0);
	float a,b;
	if(!m_sHash.getAABB().Intersect(r, &a, &b))
		return L_n;//that would be dumb
	a = clamp(a, tmin , tmax );
	b = clamp(b, tmin , tmax );
	float d = 2.0f * a_r;
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
					while(i != -1 && i < m_uMaxPhotonCount)
					{
						k_pPpmPhoton e = m_pDevicePhotons[i];
						float3 wi = e.getWi(), l = e.getL(), P = e.Pos;
						if(dot(P - x, P - x) < r2)
						{
							float p;
							if(VOL)
								p = g_SceneData.m_sVolume.p(x, -wi, r.direction);
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

template<bool DIRECT> __global__ void k_EyePass(int2 off, int w, int h, RGBCOL* a_Target, k_sPpmPixel* a_Pixels, float a_PassIndex, float a_rSurface, float a_rVolume)
{
	if(off.x)
		a_PassIndex++;
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	x += off.x; y += off.y;
	if(x < w && y < h)
	{
		Ray ro = g_CameraData.GenRay(x, y, w, h, rng.randomFloat(), rng.randomFloat());

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
				e_KernelBSDF bsdf = r2.m_pTri->GetBSDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);

				if(g_SceneData.m_sVolume.HasVolumes())
				{
					float tmin, tmax;
					g_SceneData.m_sVolume.IntersectP(s.r, 0, r2.m_fDist, &tmin, &tmax);
					L += s.fs * g_Map.L<true>(a_rVolume, rng, s.r, tmin, tmax, make_float3(0));
					s.fs = s.fs * exp(-g_SceneData.m_sVolume.tau(s.r, tmin, tmax));
				}

				float3 p = s.r(r2.m_fDist);
				if(DIRECT)
					L += s.fs * UniformSampleAllLights(p, bsdf.ng, -s.r.direction, &bsdf, rng, 4);
				L += s.fs * Le(s.r(r2.m_fDist), bsdf.ng, -s.r.direction, r2, g_SceneData);
				e_KernelBSSRDF bssrdf;
				if(r2.m_pTri->GetBSSRDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset, &bssrdf))
				{
					float3 dir = refract(s.r.direction, bsdf.sys.m_normal, 1.0f / bssrdf.e);
					TraceResult r3;
					r3.Init();
					k_TraceRay<true>(dir, p, &r3);
					L += s.fs * g_Map.L<false>(a_rVolume, rng, Ray(p, dir), 0, r3.m_fDist, bssrdf.sigp_s + bssrdf.sig_a);
				}
				if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)))
					L += s.fs * g_Map.L(a_rSurface, rng, &bsdf, bsdf.sys.m_normal, p, -s.r.direction);
				if(s.d < 5 && stackPos < stackN - 1)
				{/*
					if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_GLOSSY)))
					{
						float3 wi;
						float pdf;
						const int N = clamp(2u, 0u, stackN - stackPos);
						for(int i = 0; i < N; i++)
						{
							float3 f = bsdf.Sample_f(-s.r.direction, &wi, BSDFSample(rng), &pdf, BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_GLOSSY));
							if(pdf && fsumf(f) != 0)
								stack[stackPos++] = stackEntry(Ray(p, wi), bsdf.IntegratePdf(f, pdf, wi) * s.fs / float(N), s.d + 1);
						}
					}
					else*/
					{
						float3 r_wi;
						float r_pdf;
						float3 r_f = bsdf.Sample_f(-s.r.direction, &r_wi, BSDFSample(rng), &r_pdf, BxDFType(BSDF_REFLECTION | BSDF_SPECULAR | BSDF_GLOSSY));
						if(r_pdf && fsumf(r_f) != 0)
							stack[stackPos++] = stackEntry(Ray(p, r_wi), bsdf.IntegratePdf(r_f, r_pdf, r_wi) * s.fs, s.d + 1);
						float3 t_wi;
						float t_pdf;
						float3 t_f = bsdf.Sample_f(-s.r.direction, &t_wi, BSDFSample(rng), &t_pdf, BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR | BSDF_GLOSSY));
						if(t_pdf && fsumf(t_f) != 0)
							stack[stackPos++] = stackEntry(Ray(p, t_wi), bsdf.IntegratePdf(t_f, t_pdf, t_wi) * s.fs, s.d + 1);
					}
				}
			}
			else if(g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				g_SceneData.m_sVolume.IntersectP(s.r, 0, r2.m_fDist, &tmin, &tmax);
				L += s.fs * g_Map.L<true>(a_rVolume, rng, s.r, tmin, tmax, make_float3(0));
			}
		}
		a_Pixels[y * w + x].m_vPixelColor += L;
		RGBCOL c = Float3ToCOLORREF(a_Pixels[y * w + x].m_vPixelColor / a_PassIndex);
		unsigned int i2 = y * w + x;
		a_Target[i2] = c;
	}
	g_RNGData(rng);
}

/*
__global__ void k_EyePass(int2 off, int w, int h, RGBCOL* a_Target, k_sPpmPixel* a_Pixels, float a_PassIndex, float a_rSurface, float a_rVolume)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	curand_init(1234, y * w + x, a_PassIndex, (curandState*)&rng);
	x += off.x; y += off.y;
	if(x < w && y < h)
	{
		Ray r = g_CameraData.GenRay(x, y, w, h, rng.randomFloat(), rng.randomFloat());

		float3 L = make_float3(0), throughput = make_float3(1);
		TraceResult r2;
		r2.Init(); int d = 0;
		while(k_TraceRay<true>(r.direction, r.origin, &r2) && d++ < 10)
		{
			float3 p = r(r2.m_fDist);
			e_KernelBSDF bsdf = r2.m_pTri->GetBSDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);
			L += throughput * r2.m_pTri->Le(r2.m_fUV, bsdf.ng, -r.direction, g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);
			if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)))
			{
				L += throughput * g_Map.L(a_rSurface, rng, &bsdf, bsdf.sys.m_normal, p, -r.direction);
				break;
			}
			else
			{
				float3 wi;
				float pdf;
				BxDFType sampledType;
				float3 f = bsdf.Sample_f(-r.direction, &wi, BSDFSample(rng), &pdf, BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR | BSDF_GLOSSY), &sampledType);
				if(pdf > 0 && fsumf(f) != 0)
				{
					throughput = throughput * f * AbsDot(wi, bsdf.sys.m_normal) / pdf;
					r = Ray(p, wi);
				}
				else break;
			}
			r2.Init();
		}

		a_Pixels[y * w + x].m_vPixelColor += L;
		RGBCOL c = Float3ToCOLORREF(a_Pixels[y * w + x].m_vPixelColor / a_PassIndex);
		unsigned int i2 = y * w + x;
		a_Target[i2] = c;
	}
	g_RNGData(rng);
}
*/
void k_sPpmTracer::doEyePass(RGBCOL* a_Buf)
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	const unsigned int p = 16;
	if(m_bDirect)
		k_EyePass<true><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, a_Buf, m_pDevicePixels, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
	else k_EyePass<false><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, a_Buf, m_pDevicePixels, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
}

void k_sPpmTracer::Debug(int2 pixel)
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	const unsigned int p = 16;
	k_EyePass<true><<<1, 1>>>(pixel, w, h, (RGBCOL*)m_pDevicePixels, m_pDevicePixels, m_uPassesDone, getCurrentRadius(2), getCurrentRadius(3));
}