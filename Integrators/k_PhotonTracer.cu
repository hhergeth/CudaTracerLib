#include "k_PhotonTracer.h"
#include "../Kernel/k_TraceHelper.h"
#include "../Kernel/k_TraceAlgorithms.h"
#include "../Engine/e_Light.h"
#include "../Engine/e_Sensor.h"

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE unsigned int g_NextRayCounter3;

CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec, e_Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(pRec.p, pRec.n);
	Spectrum value = weight * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
	if (!value.isZero() && ::V(dRec.p, dRec.ref))
	{
		const e_KernelLight* emitter = (const e_KernelLight*)pRec.object;
		value *= emitter->evalDirection(DirectionSamplingRecord(dRec.d), pRec);
		g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
}

CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, BSDFSamplingRecord& bRec, const TraceResult& r2, e_Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	Spectrum value = weight * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
	if(!value.isZero() && ::V(dRec.p, dRec.ref))
	{
		bRec.wo = bRec.dg.toLocal(dRec.d);
		value *= r2.getMat().bsdf.f(bRec);
		g_Image.Splat(dRec.uv.x, dRec.uv.y,  value);
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
	r2.Init();
	int depth = -1;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while(++depth < 12 && k_TraceRay(r.direction, r.origin, &r2))
	{
		r2.getBsdfSample(r, bRec, ETransportMode::EImportance, &rng);
		
		handleSurfaceInteraction(power * throughput, bRec, r2, g_Image, rng);

		Spectrum bsdfWeight = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());

		r = Ray(bRec.dg.P, bRec.getOutgoing());
		r2.Init();
		if(bsdfWeight.isZero())
			break;
		throughput *= bsdfWeight;
		if(depth > 5)
		{
			float q = min(throughput.max(), 0.95f);
			if(rng.randomFloat() >= q)
				break;
			throughput /= q;
		}
	}
}

__global__ void pathKernel(unsigned int N, e_Image g_Image)
{
	CudaRNG rng = g_RNGData();
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	volatile int& rayBase = nextRayArray[threadIdx.y];
	do
	{
		if (threadIdx.x == 0)
			rayBase = atomicAdd(&g_NextRayCounter3, blockDim.x);

		int rayidx = rayBase + threadIdx.x;
		if (rayidx >= N)
			break;

		doWork(g_Image, rng);
	} while (true);
	g_RNGData(rng);
}

void k_PhotonTracer::DoRender(e_Image* I)
{
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NextRayCounter3, &zero, sizeof(unsigned int)));
	k_INITIALIZE(m_pScene, g_sRngs);
	pathKernel << < 180, dim3(32, MaxBlockHeight, 1) >> >(w * h, *I);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

void k_PhotonTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	doWork(*I, rng);
	g_RNGData(rng);
}