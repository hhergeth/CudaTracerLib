#include "k_PhotonTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter3;

CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec, e_Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(pRec.p, pRec.n, pRec.uv);
	Spectrum value = weight * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
	if(!value.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
	{
		const e_KernelLight* emitter = (const e_KernelLight*)pRec.object;
		value *= emitter->evalDirection(DirectionSamplingRecord(dRec.d), pRec);
		g_Image.Splat(int(dRec.uv.x), int(dRec.uv.y), value);
	}
}

CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, BSDFSamplingRecord& bRec, const TraceResult r2, e_Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(bRec.map.P, bRec.map.sys.n, bRec.map.uv);
	Spectrum value = weight * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
	if(!value.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
	{
		bRec.wo = bRec.map.sys.toLocal(dRec.d);
		value *= r2.getMat().bsdf.f(bRec);
		g_Image.Splat(int(dRec.uv.x), int(dRec.uv.y),  value);
	}
}

CUDA_FUNC_IN void doWork(e_Image& g_Image, CudaRNG& rng)
{
	PositionSamplingRecord pRec;
	Spectrum power = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2()), throughput = Spectrum(1.0f);

	handleEmission(power, pRec, g_Image, rng);

	DirectionSamplingRecord dRec;
	power *= ((const e_KernelLight*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());
	Ray r(pRec.p, dRec.d);
	TraceResult r2;
	r2.Init(true);
	int depth = -1;
	BSDFSamplingRecord bRec;
	while(++depth < 12 && k_TraceRay(r.direction, r.origin, &r2))
	{
		r2.getBsdfSample(r, rng, &bRec);
		bRec.mode = EImportance;
		
		handleSurfaceInteraction(power * throughput, bRec, r2, g_Image, rng);
		
		Spectrum bsdfWeight = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		r = Ray(bRec.map.P, bRec.getOutgoing());
		r2.Init();
		if(bsdfWeight.isZero())
			break;
		throughput *= bsdfWeight;
		if(depth > 5)
		{
			float q = MIN(throughput.max(), 0.95f);
			if(rng.randomFloat() >= q)
				break;
			throughput /= q;
		}
	}
}

__global__ void pathKernel(unsigned int N, e_Image g_Image)
{
	CudaRNG rng = g_RNGData();
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
				rayBase = atomicAdd(&g_NextRayCounter3, numTerminated);

            rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
                break;
		}

		doWork(g_Image, rng);
	}
	while(true);
	g_RNGData(rng);
}

static e_Image I2;
void k_PhotonTracer::DoRender(e_Image* I)
{
	I2 = *I;
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter3, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	pathKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(256 * 256, *I);
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay();
}

void k_PhotonTracer::Debug(int2 pixel)
{
	CudaRNG rng = g_RNGData();
	doWork(I2, rng);
}