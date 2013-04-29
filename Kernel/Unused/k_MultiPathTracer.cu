#include "k_MultiPathTracer.h"
#include "k_TraceHelper.h"
#include <time.h>

#include "k_sPpmTracer.cu"

CUDA_DEVICE unsigned int g_NextWriteCounter;

__global__ void startKernel(unsigned int w, unsigned int h, BufferEntry* a_Buffer, CudaRNG* a_RNGs, unsigned int a_NumRNGs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	int ind = (y * w + x) % a_NumRNGs;
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		float3 dir, ori;
		g_CameraData.GenRay(x2, y2, &dir, &ori);//, a_RNGs[ind].randomFloat(), a_RNGs[ind].randomFloat()
		TraceResult r;
		a_Buffer[y * w + x] = BufferEntry(ori, dir, x, y);
	}
}

__global__ void traceKernel(BufferEntry* a_Buffer, unsigned int N)
{
	int rayidx;
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        const bool          terminated     = 1;//nodeAddr == EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));	

        if(terminated)
        {			
            if (idxTerminated == 0)
				rayBase = atomicAdd(&g_NextRayCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
                break;
		}

		float3 d = a_Buffer[rayidx].Dir;
		float3 p = a_Buffer[rayidx].Pos;
		TraceResult r2;
		r2.Init();
		k_TraceRay<true>(d, p, &r2);
		a_Buffer[rayidx].Res = r2;
	}
	while(true);
}

__global__ void evalKernel(float4* a_DataTmp, RGBCOL* a_Data, unsigned int N, unsigned int w, unsigned int h, BufferEntry* a_Buffer, unsigned int depth, float pass, CudaRNG* a_RNGs, unsigned int a_NumRNGs)
{
	int ind0 = blockIdx.x * blockDim.x + threadIdx.x, ind1 = ind0 % a_NumRNGs;
	if(ind0 < N)
	{
		BufferEntry e = a_Buffer[ind0];
		if(e.Res.hasHit() && depth < 7)
		{
			e_KernelMesh mesh2 = g_SceneData.m_sMeshData[e.Res.m_pNode->m_uMeshIndex];
			e_KernelMaterial m = g_SceneData.m_sMatData[mesh2.m_uMaterialOffset + e.Res.m_pTri->getMatIndex(mesh2.m_uMaterialOffset)];
			e.cl += e.cf * m.Emission;
			float3 x = e.Pos + e.Dir * e.Res.m_fDist;
			float2 uv = e.Res.m_pTri->lerpUV(e.Res.m_fUV);
			Onb sys = e.Res.m_pTri->lerpOnb(e.Res.m_fUV);
			float3 nl = sys.getDirNormal(e.Dir);
			float3 inc;
			float pdf;
			bool diff;
			float3 f = m.m_sBSDF.Sample_f(-e.Dir, &inc, a_RNGs[ind1].randomFloat(), a_RNGs[ind1].randomFloat(), &pdf, sys);
			float p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; 
			if (depth > 5)
				if (a_RNGs[ind1].randomFloat() < p)
					f = f / p;
				else goto label001;
			if(!pdf)
				goto label001;
			e.cf = e.cf * f;
			e.Dir = inc;
			e.Pos = x + inc * 0.01f;
			unsigned int ind = atomicInc(&g_NextWriteCounter, -1);
			a_Buffer[ind] = e;
		}
		else
		{
label001:
			float4* data = a_DataTmp + e.y * w + e.x;
			*data += make_float4(e.cl, 0);
			a_Data[(h - e.y - 1) * w + e.x] = Float3ToCOLORREF(clamp01(!*data / pass));
		}
	}
}

void k_MultiPathTracer::DoRender(RGBCOL* a_Buf)
{
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	cudaMemset(m_pBuffer, 0, w * h * sizeof(BufferEntry));
	m_uLastCount = w * h;
	const unsigned int p = 16;
	startKernel<<< dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, m_pBuffer, (CudaRNG*)m_pRngData, m_uNumRNGs);
	int d = 0;
	while(m_uLastCount)
	{
		int b = 0;
		cudaMemcpyToSymbol(g_NextRayCounter, &b, 4);
		traceKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(m_pBuffer, m_uLastCount);
		cudaMemcpyToSymbol(g_NextWriteCounter, &b, 4);
		evalKernel<<< m_uLastCount / 1024 + 1, 1024>>>(m_pTmpData, a_Buf, m_uLastCount, w, h, m_pBuffer, d++, m_uPassesDone, (CudaRNG*)m_pRngData, m_uNumRNGs);
		cudaMemcpyFromSymbol(&m_uLastCount, g_NextWriteCounter, 4);
	}
}

void k_MultiPathTracer::StartNewTrace()
{
	cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
}

void k_MultiPathTracer::Resize(unsigned int _w, unsigned int _h)
{
	k_RandTracerBase::Resize(_w, _h);
	if(m_pTmpData)
		cudaFree(m_pTmpData);
	cudaMalloc(&m_pTmpData, sizeof(float4) * w * h);
	cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
	if(m_pBuffer)
		cudaFree(m_pBuffer);
	cudaMalloc(&m_pBuffer, sizeof(BufferEntry) * w * h);
	cudaMemset(m_pBuffer, 0, w * h * sizeof(BufferEntry));
}

void k_MultiPathTracer::Debug(int2 pixel)
{/*
	m_pScene->UpdateInvalidated();
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	debugPixel<<<1,1>>>(w,h,pixel);*/
}