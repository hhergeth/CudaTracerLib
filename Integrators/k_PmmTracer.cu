#include "k_PmmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include "k_PmmHelper.h"

CUDA_DEVICE e_SpatialLinkedMap<SpatialEntry> g_sMap;
CUDA_DEVICE e_SpatialSet<DirectionModel> g_dMap;

__global__ void tracePhotons()
{
	CudaRNG rng = g_RNGData();
	TraceResult r2;
	Ray r;
	g_SceneData.sampleEmitterRay(r, rng.randomFloat2(), rng.randomFloat2());
	int depth = 0;
	while((r2 = k_TraceRay(r)).hasHit() && depth++ < 7)
	{
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		if(f.isZero())
			break;
		float3 p = r(r2.m_fDist);
		g_sMap.store(p, bRec.wi); 
		if (depth > 5)
			if (rng.randomFloat() >= f.max())
				break;
		r = Ray(p, bRec.getOutgoing());
		r2.Init();
	}
	g_RNGData(rng);
}

template<int MAX_SAMPLES> __global__ void updateCache(float ny)
{
	unsigned int tIdx = getGlobalIdx_3D_3D(), x = blockIdx.x * gridDim.x + threadIdx.x, 
											  y = blockIdx.y * gridDim.y + threadIdx.y, 
											  z = blockIdx.z * gridDim.z + threadIdx.z;
	if(tIdx < g_dMap.NumEntries())
	{
		uint3 i = make_uint3(x,y,z);
		float3 mi = g_dMap.hashMap.InverseTransform(i), ma = g_dMap.hashMap.InverseTransform(i + make_uint3(1));
		g_dMap(tIdx).Update<MAX_SAMPLES>(g_sMap, mi, ma, ny);
	}
}

__global__ void visualize(e_Image I, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = k_TraceRay(r);
		float num = 0;
		if(r2.hasHit())
		{
			float3 p = r(r2.m_fDist);
			num = g_dMap(p).numSamples;
		}
		I.AddSample(x, y, Spectrum(num / 50.0f));
	}
}

void k_PmmTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	I->Clear();
	k_INITIALIZE(m_pScene, g_sRngs);
	unsigned int p = 16, w, h;
	I->getExtent(w, h);
	visualize<<<dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(*I, w, h);
	cudaError_t r = cudaThreadSynchronize();
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(1.0f);
}

float ny(int i)
{
	const float alpha = 0.75f;
	return std::pow(float(i), -alpha);
}

void k_PmmTracer::StartNewTrace(e_Image* I)
{
	sMap.ResetBuffer();
	AABB box = this->GetEyeHitPointBox(m_pScene, m_pCamera, true);
	//AABB box = m_pScene->getBox(m_pScene->getNodes());
	sMap.SetSceneDimensions(box, length(box.Size()) / 100.0f);
	cudaMemcpyToSymbol(g_sMap, &sMap, sizeof(sMap));
	ThrowCudaErrors();
	k_INITIALIZE(m_pScene, g_sRngs);
	while(sMap.deviceDataIdx < sMap.numData)
	{
		tracePhotons<<< 20, 256>>>();
		cudaMemcpyFromSymbol(&sMap, g_sMap, sizeof(sMap));
	}
	dMap.ResetBuffer();
	dMap.SetSceneDimensions(box, length(box.Size()) / 100.0f);
	CudaRNG rng = g_RNGData();
	DirectionModel* models = new DirectionModel[dMap.NumEntries()];
	for(unsigned int i = 0; i < dMap.NumEntries(); i++)
		models[i].Initialze(rng);
	cudaMemcpy(dMap.deviceData, models, dMap.NumEntries() * sizeof(DirectionModel), cudaMemcpyHostToDevice);
	delete [] models;
	g_RNGData(rng);
	cudaMemcpyToSymbol(g_dMap, &dMap, sizeof(dMap));
	int l = 6, L = dMap.gridSize / l + 1;
	updateCache<8><<<dim3(L,L,L), dim3(l,l,l)>>>(ny(1));
}

void k_PmmTracer::Debug(int2 p)
{
	/*k_INITIALIZE(m_pScene, g_sRngs);
	float3* deviceDirs;
	unsigned int* deviceNum;
	CUDA_MALLOC(&deviceDirs, sizeof(float3) * 10000);
	CUDA_MALLOC(&deviceNum, sizeof(unsigned int));
	copyDirections<<<1,1>>>(p.x, p.y, deviceDirs, deviceNum);
	unsigned int N;
	cudaMemcpy(&N, deviceNum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	float3* directions = (float3*)alloca(sizeof(float3) * N);
	cudaMemcpy(directions, deviceDirs, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	CUDA_FREE(deviceDirs);
	CUDA_FREE(deviceNum);
	plotPoints(directions, N);*/

	k_INITIALIZE(m_pScene, g_sRngs);
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);
	TraceResult r2 = k_TraceRay(r);
	float3 pa = r(r2.m_fDist);
	unsigned int idx = dMap.hashMap.Hash(pa);
	DirectionModel model;
	cudaMemcpy(&model, dMap.deviceData + idx, sizeof(model), cudaMemcpyDeviceToHost);
	plotModel(model);
}