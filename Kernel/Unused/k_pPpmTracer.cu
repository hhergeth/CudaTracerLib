#include "k_pPpmTracer.h"
#include "k_TraceHelper.h"
#include "k_sPpmTracer.cu"

#define EPS 0.01f

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
__global__ void k_GuessPass(int w, int h, CudaRNG* a_Rngs, unsigned int a_RNGCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	int ind = (y * w + x) % a_RNGCount;
	CudaRNG localState = a_Rngs[ind];
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		float3 dir, ori;
		g_CameraData.GenRay(x2, y2, &dir, &ori,  localState.randomFloat(), localState.randomFloat());
		TraceResult r;
		r.Init();
		int d = -1;
		while(k_TraceRay<true>(dir, ori, &r) && ++d < 10)
		{
			float3 p = ori + dir * r.m_fDist;
			float2 uv = r.m_pTri->lerpUV(r.m_fUV);
			Onb sys = r.m_pTri->lerpOnb(r.m_fUV);
			float3 nl = sys.getDirNormal(dir);
			float3 inc;
			float pdf;
			e_KernelMaterial& mat = g_SceneData.m_sMatData[r.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
			float3 col = mat.m_sBSDF.Sample_f(-1.0f * dir, &inc, localState.randomFloat(), localState.randomFloat(), &pdf, sys);
			dir = inc;
			ori = p + dir * EPS;
			uint3 pu = make_uint3(FloatToUInt(p.x), FloatToUInt(p.y), FloatToUInt(p.z));
			atomicMin(&g_EyeHitBoxMin.x, pu.x);
			atomicMin(&g_EyeHitBoxMin.y, pu.y);
			atomicMin(&g_EyeHitBoxMin.z, pu.z);
			atomicMax(&g_EyeHitBoxMax.x, pu.x);
			atomicMax(&g_EyeHitBoxMax.y, pu.y);
			atomicMax(&g_EyeHitBoxMax.z, pu.z);
			r.Init();
		}
	}
}

CUDA_FUNC_IN float hash(const float3 idx, const float HashScale, const float HashNum)
{
	// use the same procedure as GPURnd
	float4 n = make_float4(idx, idx.x + idx.y - idx.z) * 4194304.0 / HashScale;

	const float4 q = make_float4(   1225.0,    1585.0,    2457.0,    2098.0);
	const float4 r = make_float4(   1112.0,     367.0,      92.0,     265.0);
	const float4 a = make_float4(   3423.0,    2646.0,    1707.0,    1999.0);
	const float4 m = make_float4(4194287.0, 4194277.0, 4194191.0, 4194167.0);

	float4 beta = floor(n / q);
	float4 p = a * (n - beta * q) - beta * r;
	beta = (signf(-p) + make_float4(1.0)) * make_float4(0.5) * m;
	n = (p + beta);

	return floor( frac(dot(n / m, make_float4(1.0, -1.0, 1.0, -1.0))) * HashNum );
}

__global__ void k_TracePhotons(CudaRNG* a_Rngs, unsigned int a_RNGCount, k_pPpmPhotonEntry* a_Buffer, float3 a_BoxMin, float HashNum, float HashScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = (y * blockDim.x * gridDim.x + x) % a_RNGCount;
	CudaRNG localState = a_Rngs[ind];

	int li = (int)((float)g_SceneData.m_sLightData.UsedCount * localState.randomFloat());
	Ray r = g_SceneData.m_sLightData[li].SampleRay(localState);
	r.origin += r.direction * EPS;
	
	int depth = -1;
	TraceResult r2;
	r2.Init();
	float3 cl = g_SceneData.m_sLightData[li].m_cPower;
	while(++depth < 10 && k_TraceRay<false>(r.direction, r.origin, &r2))
	{
		float3 pos = r(r2.m_fDist);
		e_KernelMaterial m = g_SceneData.m_sMatData[r2.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset)];

		float2 uv = r2.m_pTri->lerpUV(r2.m_fUV);
		Onb sys = r2.m_pTri->lerpOnb(r2.m_fUV);
		float3 inc;
		float pdf;
		float3 nor = sys.getDirNormal(r.direction);
		float3 col = m.m_sBSDF.Sample_f(-r.direction, &inc, localState.randomFloat(), localState.randomFloat(), &pdf, sys);

		float3 idx = (pos - a_BoxMin) * HashScale;
		unsigned int ind = hash(idx, HashScale, HashNum);
		a_Buffer[ind] = k_pPpmPhotonEntry(pos, cl, nor, r.direction);

		cl = cl * col + m.Emission;
		if(pdf == 0)
			break;
		if(depth > 3)
		{
			float a = fmaxf(cl), prob = a;
			if(localState.randomFloat() > prob)
				break;
			cl /= prob;
		}
		r.direction = inc;
		r.origin = pos + EPS * r.direction;
		r2.Init();
	}
}

__global__ void k_TraceEyeRays(int w, int h, CudaRNG* a_Rngs, unsigned int a_RNGCount, float r2, float3 r3, k_pPpmPhotonEntry* a_Buffer, float3 a_BoxMin, float HashNum, float HashScale, float3* a_Target, float q, float3 a_BoxSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = (y * w + x) % a_RNGCount;
	CudaRNG localState = a_Rngs[ind];
	if(x < w && y < h)
	{
		float3 currCol = a_Target[y * w + x], cl = make_float3(1);
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		float3 dir, ori;
		g_CameraData.GenRay(x2, y2, &dir, &ori,  localState.randomFloat(), localState.randomFloat());
		TraceResult r;
		r.Init();
		int d = -1;
		while(k_TraceRay<true>(dir, ori, &r) && ++d < 10)
		{
			float3 p = ori + dir * r.m_fDist;
			float2 uv = r.m_pTri->lerpUV(r.m_fUV);
			Onb sys = r.m_pTri->lerpOnb(r.m_fUV);
			float3 nl = sys.getDirNormal(dir);
			float3 inc;
			float pdf;
			e_KernelMaterial& mat = g_SceneData.m_sMatData[r.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
			BxDFType sampledBxDF;
			float3 col = mat.m_sBSDF.Sample_f(-dir, &inc, localState.randomFloat(), localState.randomFloat(), &pdf, sys, BSDF_ALL, &sampledBxDF);

			if((sampledBxDF & (BSDF_DIFFUSE | BSDF_GLOSSY)))
			{
				float3 RangeMin = clamp(p - r3 - a_BoxMin, make_float3(0), a_BoxSize) * HashScale;
				float3 RangeMax = clamp(p + r3 - a_BoxMin, make_float3(0), a_BoxSize) * HashScale;
				for (int iz = int(RangeMin.z); iz <= int(RangeMax.z); iz ++)
					for (int iy = int(RangeMin.y); iy <= int(RangeMax.y); iy++)
						for (int ix = int(RangeMin.x); ix <= int(RangeMax.x); ix++)
						{
							float3 HashIndex = make_float3(ix, iy, iz);
							unsigned int idx = hash(HashIndex, HashScale, HashNum);
							k_pPpmPhotonEntry e = a_Buffer[idx];
							float3 RangeMin = HashIndex / HashScale + a_BoxMin;
							float3 RangeMax = (HashIndex + make_float3(1.0)) / HashScale + a_BoxMin;
							//if ((RangeMin.x < PhotonPosition.x) && (PhotonPosition.x < RangeMax.x) && (RangeMin.y < PhotonPosition.y) && (PhotonPosition.y < RangeMax.y) &&	(RangeMin.z < PhotonPosition.z) && (PhotonPosition.z < RangeMax.z))
							{
								float3 dir = e.Pos - p;
								if (dot(dir, dir) < r2 && (-dot(dir, e.Dir) > 0.001))
								{
									float3 incFlux = e.Power * LIGHT_FACTOR;
									currCol = (currCol + incFlux / PI) * q * cl;//highly questionable!
								}
							}
						}
			}
			dir = inc;
			ori = p + dir * EPS;
			cl = cl * col;
		}
		a_Target[y * w + x] = currCol;
	}
}

__global__ void k_UpdateTarget(int w, int h, float3* a_TargetTmp, RGBCOL* a_Target, float r2, float a_PhotonCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int i2 = (h - y - 1) * w + x;
		float3 c = a_TargetTmp[y * w + x] * (1.0f / (PI * r2 * a_PhotonCount));
		a_Target[i2] = Float3ToCOLORREF(c);
	}
}

void k_pPpmTracer::DoRender(RGBCOL* a_Buf)
{
	cudaMemset(m_pDevicePhotonBuffer, 0, m_uMaxPhotonCount * sizeof(k_pPpmPhotonEntry));
	float hashScale = 1.0f / (sqrtf(m_fStartRadius) * 1.5f);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	m_sTimer.StartTimer();
	float N = m_uPassIndex++, q = (N * m_fAlpha + m_fAlpha) / (N * m_fAlpha + 1);
	k_TracePhotons<<<m_uMaxPhotonCount / 10 / (6 * 32), 6 * 32>>>((CudaRNG*)m_pRngData, m_uNumRNGs, m_pDevicePhotonBuffer, m_sEyeBox.minV, float(m_uMaxPhotonCount), hashScale);
	cudaThreadSynchronize();
	const unsigned int p = 16;
	k_TraceEyeRays<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, (CudaRNG*)m_pRngData, m_uNumRNGs, m_fCurrRadius, make_float3(sqrtf(m_fCurrRadius)), m_pDevicePhotonBuffer, m_sEyeBox.minV, float(m_uMaxPhotonCount), hashScale, m_pDeviceAccBuffer, q, m_sEyeBox.Size());
	cudaThreadSynchronize();
	double tRendering = m_sTimer.EndTimer();
	m_dTimeRendering += tRendering;
	m_dTimeSinceLastUpdate += tRendering;
	m_uNumPhotonsEmitted += m_uMaxPhotonCount / 10;
	m_fCurrRadius *= q;
	if(m_dTimeSinceLastUpdate > 0)
	{
		m_dTimeSinceLastUpdate = 0;
		k_UpdateTarget<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, m_pDeviceAccBuffer, a_Buf, m_fCurrRadius, float(m_uNumPhotonsEmitted));
		cudaThreadSynchronize();
	}
}

void k_pPpmTracer::StartNewTrace()
{
	m_uPassIndex = 0;
	uint3 ma = make_uint3(FloatToUInt(-FLT_MAX)), mi = make_uint3(FloatToUInt(FLT_MAX));
	cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12);
	cudaMemset(m_pDeviceAccBuffer, 0, sizeof(float3) * w * h);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	const unsigned int p = 16;
	k_GuessPass<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, (CudaRNG*)m_pRngData, m_uNumRNGs);
	cudaThreadSynchronize();
	cudaMemcpyFromSymbol(&m_sEyeBox.minV, g_EyeHitBoxMin, 12);
	cudaMemcpyFromSymbol(&m_sEyeBox.maxV, g_EyeHitBoxMax, 12);
	m_sEyeBox.minV = make_float3(UIntToFloat(m_sEyeBox.minV.x), UIntToFloat(m_sEyeBox.minV.y), UIntToFloat(m_sEyeBox.minV.z));
	m_sEyeBox.maxV = make_float3(UIntToFloat(m_sEyeBox.maxV.x), UIntToFloat(m_sEyeBox.maxV.y), UIntToFloat(m_sEyeBox.maxV.z));
	float r = fsumf(m_sEyeBox.maxV - m_sEyeBox.minV) / w * m_fInitialRadiusScale, r2 = r* r;
	m_fStartRadius = m_fCurrRadius = r2;
	m_uNumPhotonsEmitted = 0;
	m_dTimeSinceLastUpdate = m_dTimeRendering = 0;
}

void k_pPpmTracer::Debug(int2 pixel)
{

}