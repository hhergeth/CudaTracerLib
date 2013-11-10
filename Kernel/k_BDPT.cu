#include "k_BDPT.h"
#include "k_TraceHelper.h"
#include <time.h>
#include "k_TraceAlgorithms.h"

#define MAX_SUBPATH_LENGTH 5
struct PathVertex
{
	float3 p;
	float3 wi;
	float3 wo;
	TraceResult r2;
	Spectrum cumulative;
};
struct Path
{
	PathVertex EyePath[MAX_SUBPATH_LENGTH];
	PathVertex LightPath[MAX_SUBPATH_LENGTH];
	unsigned int s;
	unsigned int t;
	CUDA_FUNC_IN Path()
	{
		s = 0;
		t = 0;
	}
};

CUDA_FUNC_IN float G(const float3& N_x, const float3& N_y, const float3& x, const float3& y)
{
	float3 theta = normalize(y - x);
	return AbsDot(N_x, theta) * AbsDot(N_y, -theta) / DistanceSquared(x, y);
}

CUDA_FUNC_IN void randomWalk(PathVertex* vertices, unsigned int* N, Ray r, CudaRNG& rng, bool eye)
{
	Spectrum cumulative(1.0f); 
	while(*N < MAX_SUBPATH_LENGTH)
	{
		TraceResult r2 = k_TraceRay(r);
		if(!r2.hasHit())
			break;
		PathVertex& v = vertices[*N];
		v.r2 = r2;
		v.p = r(r2.m_fDist);
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		if(eye)
		{
			v.wo = -r.direction;
			r.direction = v.wi = bRec.getOutgoing();
			cumulative *= f;
		}
		else
		{
			v.wi = -r.direction;
			r.direction = v.wo = bRec.getOutgoing();
			cumulative *= f;
		}
		v.cumulative = cumulative;
		if(cumulative.isZero())
			break;
		*N++;
		r.origin = v.p;
	}
}

CUDA_FUNC_IN Spectrum evalPath(const Path& P, int nEye, int nLight, CudaRNG& rng)
{
	//eye vertex
	const PathVertex& e_i = P.EyePath[nEye - 1];
	//light vertex
	const PathVertex& l_j = P.LightPath[nLight - 1];

	Spectrum L(1.0f);
	if(nEye > 1)
		L *= P.EyePath[nEye - 2].cumulative;
	if(nLight > 1)
		L *= P.LightPath[nLight - 2].cumulative;

	float3 dir = l_j.p - e_i.p;
	float l = length(dir);
	dir /= l;
	Ray r(e_i.p, dir);
	if(!g_SceneData.Occluded(r, 0, l))
	{
		BSDFSamplingRecord bRec;
		e_i.r2.getBsdfSample(Ray(e_i.p, e_i.wi), rng, &bRec, dir);
		L *= e_i.r2.getMat().bsdf.f(bRec);
		float3 N_x = bRec.map.sys.n;
		l_j.r2.getBsdfSample(Ray(l_j.p, -dir), rng, &bRec, l_j.wo);
		L *= l_j.r2.getMat().bsdf.f(bRec);
		float3 N_y = bRec.map.sys.n;
		float g = G(N_x, N_y, e_i.p, l_j.p);
		L *= g;
	}
	return L;
}

CUDA_FUNC_IN void BDPT(int x, int y, e_Image& g_Image, CudaRNG& rng)
{
	Path P;
	Ray r;
	Spectrum imp = g_CameraData.sampleRay(r, make_float2(x, y), rng.randomFloat2());
	randomWalk(P.EyePath, &P.s, r, rng, true);
	const e_KernelLight* light;
	imp = g_SceneData.sampleEmitterRay(r, light, rng.randomFloat2(), rng.randomFloat2());
	randomWalk(P.LightPath, &P.t, r, rng, false);

	while(P.s < MAX_SUBPATH_LENGTH)
	{
		TraceResult r2 = k_TraceRay(r);
		if(!r2.hasHit())
			break;
		P.EyePath[P.s].L = imp;
		P.EyePath[P.s].incRay = r;
		P.EyePath[P.s].r2 = r2;
		P.s++;

		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		bRec.mode = EImportance;
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		imp *= f;
		r = Ray(r(r2.m_fDist), bRec.getOutgoing());
	}
	

	while(P.t < MAX_SUBPATH_LENGTH)
	{
		TraceResult r2 = k_TraceRay(r);
		if(!r2.hasHit())
			break;
		P.LightPath[P.t].L = imp;
		P.LightPath[P.t].incRay = r;
		P.LightPath[P.t].r2 = r2;
		P.t++;

		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		imp *= f;
		r = Ray(r(r2.m_fDist), bRec.getOutgoing());
	}

	Spectrum L(0.0f);
	for(int i = 0; i < P.s; i++)
	{
		//case ii
		UniformSampleAllLights(0, 0, 1);
		//case iv
		for(int j = 0; j < P.t; j++)
		{
						
		}
	}
	g_Image.AddSample(x, y, L);
}

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter12;

__global__ void pathKernel(unsigned int width, unsigned int height, int N, e_Image g_Image)
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
				rayBase = atomicAdd(&g_NextRayCounter12, numTerminated);

            rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
                break;
		}

		unsigned int x = rayidx % width, y = rayidx / width;
		BDPT(x, y, g_Image, rng);
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixel12(unsigned int width, unsigned int height, int2 p)
{
	CudaRNG rng = g_RNGData();
	Ray r = g_CameraData.GenRay(p.x, p.y);	
	PathTrace(r.direction, r.origin, rng);
}

void k_BDPT::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter12, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	//int N = w * h, T = 180 * 4 * 32 * 2;
	//for(int i = 0; i < N; i += T)
		pathKernel<<< 180, dim3(32, 4, 1)>>>(w, h, w*h, *I);
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(m_uPassesDone);
}

void k_BDPT::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	debugPixel12<<<1,1>>>(w,h,pixel);
}