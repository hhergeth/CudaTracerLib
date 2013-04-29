#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"

CUDA_FUNC_IN unsigned int FloatToUInt(float f2)
{
	unsigned int f = *(unsigned int*)&f2;
	unsigned int mask = -int(f >> 31) | 0x80000000;
	return f ^ mask;
}

CUDA_FUNC_IN float UIntToFloat(float f2)
{
	unsigned int f = *(unsigned int*)&f2;
	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	unsigned int i = f ^ mask;
	return *(float*)&i;
}

CUDA_DEVICE uint3 g_EyeHitBoxMin;
CUDA_DEVICE uint3 g_EyeHitBoxMax;
__global__ void k_GuessPass(int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	CudaRNG localState = g_RNGData();
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		Ray r;
		g_CameraData.GenRay(x2, y2, &r.direction, &r.origin,  localState.randomFloat(), localState.randomFloat());
		TraceResult r2;
		r2.Init();
		int d = -1;
		while(k_TraceRay<true>(r.direction, r.origin, &r2) && ++d < 10)
		{
			float2 uv = r2.m_pTri->lerpUV(r2.m_fUV);
			Onb sys = r2.m_pTri->lerpOnb(r2.m_fUV);
			float3 inc;
			float pdf;
			e_KernelMaterial& mat = g_SceneData.m_sMatData[r2.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
			float3 col = mat.m_sBSDF.Sample_f(-1.0f * r.direction, &inc, localState.randomFloat(), localState.randomFloat(), localState.randomFloat(), &pdf, sys);
			float3 p = r(r2.m_fDist);
			r = CalcNextRay(r, r2, inc, sys.m_normal);
			uint3 pu = make_uint3(FloatToUInt(p.x), FloatToUInt(p.y), FloatToUInt(p.z));
			atomicMin(&g_EyeHitBoxMin.x, pu.x);
			atomicMin(&g_EyeHitBoxMin.y, pu.y);
			atomicMin(&g_EyeHitBoxMin.z, pu.z);
			atomicMax(&g_EyeHitBoxMax.x, pu.x);
			atomicMax(&g_EyeHitBoxMax.y, pu.y);
			atomicMax(&g_EyeHitBoxMax.z, pu.z);
			r2.Init();
		}
	}
	g_RNGData(localState);
}

__global__ void k_EyePass(int2 off, int w, int h, k_sPpmEntry* a_Entries, k_sPpmPixel* a_Pixels, unsigned int* a_Grid, k_HashGrid a_Hash, unsigned int a_PassIndex)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	CudaRNG rng = g_RNGData();
	x += off.x; y += off.y;
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		Ray r;
		g_CameraData.GenRay(x2, y2, &r.direction, &r.origin,  rng.randomFloat(), rng.randomFloat());
		TraceResult r2;
		r2.Init();
		int depth = 0;
		float3 throughput = make_float3(1);
		while(k_TraceRay<true>(r.direction, r.origin, &r2) && depth++ < 10)
		{
			Onb sys = r2.m_pTri->lerpOnb(r2.m_fUV);
			sys.Redirect(r.direction);
			float3 wi;
			float pdf;
			e_KernelMaterial& mat = g_SceneData.m_sMatData[r2.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
			BxDFType sType;
			const e_KernelBXDF* bxdf;
			float3 f = mat.m_sBSDF.Sample_f(-r.direction, &wi, rng.randomFloat(), rng.randomFloat(), rng.randomFloat(), &pdf, sys, BSDF_ALL, &sType, &bxdf); 
			if((sType & BSDF_DIFFUSE) == BSDF_DIFFUSE)
			{
				float3 p = r(r2.m_fDist);
				unsigned int i = a_Hash.Hash(a_Hash.Transform(p));
				unsigned int j = y * w + x;
				unsigned int k = atomicExch(a_Grid + i, j);
				a_Entries[j].setData(throughput, p, sys.m_normal, -r.direction, bxdf, k);
				break;
			}
			else
			{
				if(pdf == 0 || fsumf(f) == 0)
					break;
				throughput = f * AbsDot(wi, sys.m_normal) / pdf;
				r = CalcNextRay(r, r2, wi, sys.m_normal);
				r2.Init();
			}
		}
	}
	g_RNGData(rng);
}

__global__ void k_PhotonPass(unsigned int spp, k_sPpmEntry* a_Entries, k_HashGrid a_Hash, unsigned int* a_Grid, float a_r2, float a_g, float a_r) 
{
	CudaRNG rng = g_RNGData();
	for(int _photonNum = 0; _photonNum < spp; _photonNum++)
	{
		int li = (int)((float)g_SceneData.m_sLightData.UsedCount * rng.randomFloat());
		Ray r = g_SceneData.m_sLightData[li].SampleRay(rng);
		r.origin += r.direction * 0.01f;
		int depth = -1;
		TraceResult r2;
		r2.Init();
		float3 Le = g_SceneData.m_sLightData[li].m_cPower;
		while(++depth < 10 && k_TraceRay<true>(r.direction, r.origin, &r2))
		{
			e_KernelMaterial m = g_SceneData.m_sMatData[r2.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
			Onb sys = r2.m_pTri->lerpOnb(r2.m_fUV);
			sys.Redirect(r.direction);
			float3 x = r(r2.m_fDist);
			BxDFType specularType = BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR);
			bool hasNonSpecular = (m.m_sBSDF.NumComponents() > m.m_sBSDF.NumComponents(specularType));
			if(hasNonSpecular && a_Hash.IsValidHash(x))
			{
				uint3 lo = a_Hash.Transform(x - make_float3(a_r)), hi = a_Hash.Transform(x + make_float3(a_r));
				for(int a = lo.x; a <= hi.x; a++)
					for(int b = lo.y; b <= hi.y; b++)
						for(int c = lo.z; c <= hi.z; c++)
						{
							unsigned int i0 = a_Hash.Hash(make_uint3(a,b,c)), i = a_Grid[i0];
							while(i != -1)
							{
								k_sPpmEntry* e = a_Entries + i;
								float dist2 = dot(e->Pos - x, e->Pos - x), _dot = dot(sys.m_normal, e->Nor);
								if(dist2 < a_r2 && _dot > 0.5f)
								{
									float3 wo = sys.worldTolocal(e->Dir), wi = sys.localToworld(-r.direction);
									float3 flux = Le * e->Bsdf->f(wo, wi) * AbsDot(-r.direction, sys.m_normal) * abs(_dot) * e->Weight;
									e->Tau = (e->Tau + flux) * a_g;
								}
								i = e->next;
							}
						}
			}
			float3 wo = -r.direction, wi;
			float pdf;
			BxDFType sampledType;
			float3 f = m.m_sBSDF.Sample_f(wo, &wi, rng.randomFloat(), rng.randomFloat(), rng.randomFloat(), &pdf, sys, BSDF_ALL, &sampledType);
			if(pdf == 0 || fsumf(f) == 0)
				break;
			Le = Le * f * AbsDot(wi, sys.m_normal) / pdf;
			float continueProb = MIN(1.0f, fmaxf(Le));
            if (rng.randomFloat() > continueProb)
                break;
			Le /= continueProb;
			r = CalcNextRay(r, r2, wi, sys.m_normal);
			r2.Init();
		}
	}
	g_RNGData(rng);
}

__global__ void k_GatherPass(k_sPpmEntry* a_Entries, k_sPpmPixel* a_Pixels, float numEmitted, RGBCOL* a_Target, unsigned int w, unsigned int h, float a_r2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h && numEmitted != 0)
	{
		unsigned int i = y * w + x;
		float3 r = a_Entries[i].Tau * (1.0f / (PI * a_r2 * numEmitted));
		RGBCOL c = Float3ToCOLORREF(clamp01(r));
		unsigned int i2 = (h - y - 1) * w + x;
		a_Target[i2] = c;
	}
}

void k_sPpmTracer::DoRender(RGBCOL* a_Buf)
{
	const unsigned int currPass = m_uCurrentEyePassIndex * m_uNumRunsPerEyePass + m_uCurrentRunIndex;
	const unsigned int p = 16;
	const unsigned long long p0 = 6 * 32, spp = 1, n = 180, PhotonsPerPass = p0 * n * spp;
	if(m_uCurrentRunIndex >= m_uNumRunsPerEyePass)
	{
		m_uCurrentRunIndex = 0;
		cudaMemset(m_pDeviceHashGrid, -1, m_uGridLength * sizeof(unsigned int));
		k_EyePass<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, m_pDeviceEntries, m_pDevicePixels, m_pDeviceHashGrid, m_sHash, m_uCurrentEyePassIndex);
		m_uCurrentEyePassIndex++;
	}
	float q = (float(currPass) * ALPHA + ALPHA) / (float(currPass) * ALPHA + 1);
	k_PhotonPass<<< n, p0 >>>(spp, m_pDeviceEntries, m_sHash, m_pDeviceHashGrid, m_fCurrentRadius, q, sqrtf(m_fCurrentRadius));
	m_uPhotonsEmitted += PhotonsPerPass;
	m_fCurrentRadius *= q;
	m_uCurrentRunIndex++;
	k_GatherPass<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>> (m_pDeviceEntries, m_pDevicePixels, (float)m_uPhotonsEmitted, a_Buf, w, h, m_fCurrentRadius);
}

void k_sPpmTracer::StartNewTrace()
{
	m_uPhotonsEmitted = 0;
	m_uCurrentEyePassIndex = 0;
	m_uCurrentRunIndex = m_uNumRunsPerEyePass;
	uint3 ma = make_uint3(FloatToUInt(-FLT_MAX)), mi = make_uint3(FloatToUInt(FLT_MAX));
	cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	k_GuessPass<<<dim3( 2, 2, 1), dim3(16, 16, 1)>>>(32, 32);
	cudaThreadSynchronize();
	AABB m_sEyeBox;
	cudaMemcpyFromSymbol(&m_sEyeBox.minV, g_EyeHitBoxMin, 12);
	cudaMemcpyFromSymbol(&m_sEyeBox.maxV, g_EyeHitBoxMax, 12);
	m_sEyeBox.minV = make_float3(UIntToFloat(m_sEyeBox.minV.x), UIntToFloat(m_sEyeBox.minV.y), UIntToFloat(m_sEyeBox.minV.z));
	m_sEyeBox.maxV = make_float3(UIntToFloat(m_sEyeBox.maxV.x), UIntToFloat(m_sEyeBox.maxV.y), UIntToFloat(m_sEyeBox.maxV.z));
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / w * m_fInitialRadiusScale, r2 = r* r;
	m_fCurrentRadius = r2;
	m_sHash = k_HashGrid(m_sEyeBox, sqrtf(m_fCurrentRadius), m_uGridLength);
	cudaMemset(m_pDevicePixels, 0, sizeof(k_sPpmPixel) * w * h);
	cudaMemset(m_pDeviceEntries, 0, w * h * sizeof(k_sPpmEntry));
}

void k_sPpmTracer::Debug(int2 pixel)
{

}