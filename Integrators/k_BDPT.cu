#include "k_BDPT.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_FUNC_IN bool V(const float3& a, const float3& b)
{
	float3 d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l);
}

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
	CUDA_FUNC_IN PathVertex& vertex(int a, int b, int i)
	{
		if(i < a)
			return EyePath[i];
		else return LightPath[t - i - a - 1];
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
			return;
		PathVertex& v = vertices[*N];
		(*N)++;
		v.r2 = r2;
		v.p = r(r2.m_fDist);
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		if(eye)
		{
			v.wo = -r.direction;
			r.direction = v.wi = normalize(bRec.getOutgoing());
			cumulative *= f;
		}
		else
		{
			v.wi = -r.direction;
			r.direction = v.wo = normalize(bRec.getOutgoing());
			cumulative *= f;
		}
		v.cumulative = cumulative;
		if(cumulative.isZero())
			return;
		r.origin = v.p;
	}
}

CUDA_FUNC_IN Spectrum evalPath(const Path& P, int nEye, int nLight, CudaRNG& rng)
{
	const PathVertex& ev = P.EyePath[nEye - 1];
	const PathVertex& lv = P.LightPath[nLight - 1];
	Spectrum L(1.0f);
	if(nEye > 1)
		L *= P.EyePath[nEye - 2].cumulative;
	if(nLight > 1)
		L *= P.LightPath[nLight - 2].cumulative;

	float3 dir = normalize(lv.p - ev.p);
	BSDFSamplingRecord bRec;
	ev.r2.getBsdfSample(Ray(ev.p, -1.0f * ev.wi), rng, &bRec, dir);
	L *= ev.r2.getMat().bsdf.f(bRec);
	float pdf_i = ev.r2.getMat().bsdf.pdf(bRec);
	float3 N_x = bRec.map.sys.n;
	lv.r2.getBsdfSample(Ray(lv.p, dir), rng, &bRec, lv.wo);
	L *= lv.r2.getMat().bsdf.f(bRec);
	float3 N_y = bRec.map.sys.n;
	float g = G(N_x, N_y, ev.p, lv.p);
	L *= g;
	return L;
}

CUDA_FUNC_IN float pathWeight(int i, int j)
{
	return 1;
}

CUDA_FUNC_IN float misWeight(Path& P, int i, int j)
{

}

CUDA_FUNC_IN void BDPT(int x, int y, int w, int h, e_Image& g_Image, CudaRNG& rng)
{
	Path P;
	Ray r;
	Spectrum imp = g_SceneData.sampleSensorRay(r, make_float2(x, y), rng.randomFloat2());
	randomWalk(P.EyePath, &P.s, r, rng, true);
	const e_KernelLight* light;
	Spectrum Le  = g_SceneData.sampleEmitterRay(r, light, rng.randomFloat2(), rng.randomFloat2());
	randomWalk(P.LightPath, &P.t, r, rng, false);
	
	Spectrum L(0.0f);
	BSDFSamplingRecord bRec;
	for(unsigned int i = 1; i < P.s + 1; i++)
	{
		const PathVertex& ev = P.EyePath[i - 1];
		if(!ev.r2.hasHit())
			break;//urgs wtf?

		//case ii
		ev.r2.getBsdfSample(Ray(ev.p, -1.0f * ev.wo), rng, &bRec);
		DirectSamplingRecord dRec(ev.p, bRec.map.sys.n);
		Spectrum localLe = light->sampleDirect(dRec, rng.randomFloat2());
		bRec.wo = bRec.map.sys.toLocal(dRec.d);
		if(V(ev.p, dRec.p))
		{
			if(i > 1)
				localLe *= P.EyePath[i - 2].cumulative;
			const float bsdfPdf = ev.r2.getMat().bsdf.pdf(bRec);
			const float misWeight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
			L += localLe * ev.r2.getMat().bsdf.f(bRec) * pathWeight(i, 0);
		}

		//case iv
		for(unsigned int j = 1; j < P.t + 1; j++)
		{
			const PathVertex& lv = P.LightPath[j - 1];
			if(V(ev.p, lv.p))
				L += Le * evalPath(P, i, j, rng) * pathWeight(i, j);
		}
	}
	
	for(unsigned int j = 1; j < P.t + 1; j++)
	{
		const PathVertex& lv = P.LightPath[j - 1];
		if(!lv.r2.hasHit())
			break;//urgs wtf?
		
		lv.r2.getBsdfSample(Ray(lv.p, -1.0f * lv.wi), rng, &bRec);
		DirectSamplingRecord dRec(lv.p, bRec.map.sys.n);
		Spectrum localLe = Le * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
		if(V(dRec.p, lv.p))
		{
			if(j > 1)
				localLe *= P.LightPath[j - 2].cumulative;
			bRec.wo = bRec.map.sys.toLocal(dRec.d);
			localLe *= lv.r2.getMat().bsdf.f(bRec);
			if(dRec.uv.x >= 0 && dRec.uv.x < w && dRec.uv.y >= 0 && dRec.uv.y < h)
				g_Image.Splat((int)dRec.uv.x, (int)dRec.uv.y, localLe * pathWeight(0, j) / float(P.t));
		}
	}

	g_Image.AddSample(x, y, L);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		//Li(g_Image, rng, x, y);
		BDPT(x,y,w,h,g_Image, rng);
	g_RNGData(rng);
}

static e_Image* gI;
void k_BDPT::DoRender(e_Image* I)
{
	gI = I;
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene, g_sRngs);
	int p = 16;
	if(w < 200 && h < 200)
		pathKernel<<< dim3((w + p - 1) / p, (h + p - 1) / p,1), dim3(p, p, 1)>>>(w, h, 0, 0, *I);
	else
	{
		unsigned int q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				pathKernel<<< dim3(q, q,1), dim3(p, p, 1)>>>(w, h, pq * i, pq * j, *I);
	}
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(float(w*h) / float(m_uPassesDone * w * h));
}

void k_BDPT::Debug(int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	BDPT(pixel.x,pixel.y,w,h,*gI, rng);
}