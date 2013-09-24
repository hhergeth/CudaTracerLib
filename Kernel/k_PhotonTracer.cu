#include "k_PhotonTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter3;

CUDA_FUNC_IN void doWork(e_Image& g_Image, CudaRNG& rng)
{
	Ray r;
	const e_KernelLight* emitter;
	//Spectrum Le = g_SceneData.sampleEmitterRay(r, emitter, rng.randomFloat2(), rng.randomFloat2());DirectSamplingRecord dRec;
	PositionSamplingRecord pRec;
	unsigned int i = threadId;
	Spectrum Le = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2());
	emitter = (const e_KernelLight*)pRec.object;
	if(Le.isZero())
		return;
	DirectSamplingRecord dRec(pRec.p, pRec.n, pRec.uv);
	Spectrum imp = g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
	if(!g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
	{
		Spectrum weight = emitter->evalDirection(DirectionSamplingRecord(dRec.d), pRec);
		g_Image.AddSample(int(dRec.uv.x), int(dRec.uv.y), imp * Le * weight);
	}
	DirectionSamplingRecord dRec2;
	Le *= emitter->sampleDirection(dRec2, pRec, rng.randomFloat2());
	r = Ray(pRec.p, dRec2.d);
	TraceResult r2;
	r2.Init(true);
	int depth = -1;
	BSDFSamplingRecord bRec;
	while(++depth < 12 && k_TraceRay(r.direction, r.origin, &r2))
	{
		r2.getBsdfSample(r, rng, &bRec);
		bRec.mode = EImportance;
		dRec = DirectSamplingRecord(bRec.map.P, bRec.map.sys.n, bRec.map.uv);
		imp = g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
		if(!g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
		{
			bRec.wo = bRec.map.sys.toLocal(dRec.d);
			Spectrum f = r2.getMat().bsdf.f(bRec);
			g_Image.AddSample(int(dRec.uv.x), int(dRec.uv.y),  Le * f * imp );
		}
		
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		r = Ray(bRec.map.P, bRec.getOutgoing());
		r2.Init();
		if(!bRec.sampledType)
			break;
		Spectrum ac = Le * f;
		if(depth > 5)
		{
			float prob = MIN(1.0f, ac.max() / Le.max());
			if(rng.randomFloat() > prob)
				break;
			Le = ac / prob;
		}
		else Le = ac;
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

void k_PhotonTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter3, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	pathKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(256 * 256, *I);
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->UpdateDisplay();
}