#include "k_IrradianceCache.h"
#include "k_TraceAlgorithms.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"

CUDA_DEVICE k_HashGrid_Irreg g_sHash;
CUDA_DEVICE unsigned int g_sEntryCount;

template<bool DIRECT> CUDA_DEVICE float3 E1(Ray& r, TraceResult& r2, CudaRNG& rng, e_KernelBSDF* bsdf, float* dist, float3* awi = 0)
{
	float3 wo = -r.direction;
	float pdf;
	float3 wi;
	bsdf->Sample_f(wo, &wi, BSDFSample(rng), &pdf, BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE));
	if(awi)
		*awi = wi;
	return PathTrace<DIRECT>(wi, r(r2.m_fDist), rng, dist);
}

template<bool DIRECT, int N> CUDA_DEVICE void E2(Ray& r, TraceResult& r2, CudaRNG& rng, e_KernelBSDF* bsdf, float3* ae, float* ar, float3* awi = 0)
{
	float3 e = make_float3(0);
	float air = 0;
	float3 w = make_float3(0);
	for(int i = 0; i < N; i++)
	{
		float q;
		e += E1<DIRECT>(r, r2, rng, bsdf, &q, &w);
		air += 1.0f / q;
	}
	*ar = float(N) / air;
	if(awi)
		*awi = w / float(N);
	*ae = e / float(N) * PI;
}

template<bool DIRECT, int N> CUDA_DEVICE float3 E(Ray& r, TraceResult& r2, CudaRNG& rng, e_KernelBSDF* bsdf, k_IrrEntry* entries, unsigned int entryNum, unsigned int* grid, float rScale, float3* awi = 0)
{/*
		float3 ae;
		float ar;
		float3 wi;
		E2<DIRECT, N>(r, r2, rng, bsdf, &ae, &ar, &wi);
		if(awi)
			*awi = wi;
		return ae;*/

	uint3 i0 = g_sHash.Transform(r(r2.m_fDist));
	unsigned int i = g_sHash.Hash(i0);
	unsigned int j = atomicInc(&g_sEntryCount, -1);
	if(j < entryNum)
	{
		unsigned int k = atomicExch(grid + i, j);
		float3 ae;
		float ar;
		float3 wi;
		E2<DIRECT, N>(r, r2, rng, bsdf, &ae, &ar, &wi);
		entries[j] = k_IrrEntry(r(r2.m_fDist), ae, bsdf->sys.m_normal, ar * rScale, k, wi);
		if(awi)
			*awi = wi;
		return ae;
	}
	return make_float3(0);
}

template<bool DIRECT, int N> __global__ void kFirstPass(int w, int h, k_IrrEntry* entries, unsigned int entryNum, unsigned int* grid, float rScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
	{
		Ray r = g_CameraData.GenRay(x, y, w, h, rng.randomFloat(), rng.randomFloat());
		TraceResult r2 = k_TraceRay(r);
		if(r2)
		{
			e_KernelBSDF bsdf = r2.m_pTri->GetBSDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);
			if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)) && g_sHash.IsValidHash(r(r2.m_fDist)))
				E<DIRECT, N>(r, r2, rng, &bsdf, entries, entryNum, grid, rScale);
		}
	}
}

template<bool DIRECT, int N, int M, int O> __global__ void kScndPass(int w, int h, RGBCOL* a_Target, k_IrrEntry* entries, unsigned int entryNum, unsigned int* grid, unsigned int gridLength, float rScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
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
			TraceResult r2 = k_TraceRay(s.r);
			if(r2)
			{
				e_KernelBSDF bsdf = r2.m_pTri->GetBSDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);

				float3 p = s.r(r2.m_fDist);
				L += s.fs * Le(p, bsdf.ng, -s.r.direction, r2, g_SceneData);
				if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)) && g_sHash.IsValidHash(p))
				{
					uint3 i0 = g_sHash.Transform(p);
					int num = 0, si = -1;
					float3 EAcc = make_float3(0), wiAcc = make_float3(0);
					float wAcc = 0;
					while(num < M && si++ < O)
					{
						uint3 low = i0 - make_uint3(si), high = i0 + make_uint3(si);
#define ITERATE(fixedCoord, a, b, c) \
	for(int i = low.a; i < high.a; i++) \
		for(int j = low.b; j < high.b; j++) \
		{ \
			uint3 co; \
			co.a = i; \
			co.b = j; \
			co.c = fixedCoord.c; \
			unsigned int hi = g_sHash.Hash(co), e = grid[hi]; \
			while(e != -1 && e < entryNum) \
			{ \
				k_IrrEntry e2 = entries[e]; \
				e = e2.next; \
				float d = dot(e2.nor, bsdf.ng), d2 = DistanceSquared(e2.pos, p); \
				float perr = Distance(p, e2.pos) / e2.rad; \
				float nerr = sqrtf((1.f - dot(e2.nor, bsdf.ng)) / (1.f - cosf(Radians(30)))); \
				if(perr < 1 && dot(e2.nor, bsdf.ng) > 0.99f) \
				{ \
					float wi = 1 ; \
					EAcc += wi * e2.E; \
					wiAcc += wi * e2.wi; \
					wAcc += wi; \
					num++; \
				} \
			} \
		}
						ITERATE(low, x, z, y)
						if(si)
						{
							ITERATE(high, x, z, y)

							ITERATE(low, x, y, z)
							ITERATE(high, x, y, z)						

							ITERATE(low, z, y, x)
							ITERATE(high, z, y, x)
						}
#undef ITERATE
					}
					float3 e, wi;
					if(num >= M)
					{
						e = EAcc / wAcc;
						wi = normalize(wiAcc / wAcc);
						L = make_float3(0,0,0);
						break;
					}
					else
					{
						e = E<DIRECT, N>(s.r, r2, rng, &bsdf, entries, entryNum, grid, rScale, &wi);
						L = make_float3(1,0,0);
						break;
					}
					L += s.fs * bsdf.f(-s.r.direction, wi, BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE)) * e;
				}
				if(s.d < 5 && stackPos < stackN - 1)
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

		unsigned int i2 = y * w + x;
		a_Target[i2] = Float3ToCOLORREF(L);
	}
	g_RNGData(rng);
}

void k_IrradianceCache::DoRender(RGBCOL* a_Buf)
{	
	m_uPassesDone++;
}

void k_IrradianceCache::StartNewTrace(RGBCOL* a_Buf)
{
	unsigned int abc = 0;
	cudaMemcpyToSymbol(g_sEntryCount, &abc, 4);
	cudaMemset(m_pGrid, -1, sizeof(unsigned int) * m_uGridLength);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	AABB m_sEyeBox = GetEyeHitPointBox();
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / w;
	m_sGrid = k_HashGrid_Irreg(m_sEyeBox, r, m_uGridLength);
	cudaMemcpyToSymbol(g_sHash, &m_sGrid, sizeof(m_sGrid));
	int p = 16, p2 = 64;
	//kFirstPass<false, 4><<<dim3( p2 / p, p2 / p, 1), dim3(p, p, 1)>>>(p2, p2, m_pEntries, m_uEntryNum, m_pGrid, rScale);
	//cudaThreadSynchronize();
	kScndPass<false, 16, 4, 10><<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, a_Buf, m_pEntries, m_uEntryNum, m_pGrid, m_uGridLength, rScale);
	cudaThreadSynchronize();
}

void k_IrradianceCache::Resize(unsigned int _w, unsigned int _h)
{
	k_TracerBase::Resize(_w, _h);

}

void k_IrradianceCache::Debug(int2 pixel)
{

}