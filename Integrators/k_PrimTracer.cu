#include "k_PrimTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include "..\Engine\e_Core.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter2;

CUDA_FUNC_IN Spectrum trace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rng)
{
	TraceResult r2 = k_TraceRay(r);
	if(r2.hasHit())
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(r, rng, &bRec);
		dg.computePartials(r, rX, rY);
		//return Spectrum(dg.dvdx, dg.dvdy, 0);
		//return Spectrum(math::clamp01(dot(bRec.dg.sys.n, -normalize(r.direction))));
		Spectrum through = Transmittance(r, 0, r2.m_fDist, -1);
		Spectrum L = r2.Le(r(r2.m_fDist), bRec.dg.sys, -r.direction);
		//return L + r2.getMat().bsdf.getDiffuseReflectance(bRec);
		Spectrum f = L + r2.getMat().bsdf.sample(bRec, rng.randomFloat2()) * through;
		int depth = 0;
		while(r2.getMat().bsdf.hasComponent(EDelta) && depth < 5)
		{
			depth++;
			r = Ray(r(r2.m_fDist), bRec.getOutgoing());
			r2 = k_TraceRay(r);
			if(r2.hasHit())
			{
				r2.getBsdfSample(r, rng, &bRec);
				f *= r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			}
			else break;
		}
		return f;
	}
	else return g_SceneData.EvalEnvironment(r);
}

CUDA_FUNC_IN Spectrum traceR(Ray& r, CudaRNG& rng)
{
	//return 0.5f + (rng.randomFloat() * 0.5f - 0.25f);

	const bool DIRECT = 1;
	TraceResult r2;
	r2.Init();
	Spectrum c = Spectrum(1.0f), L = Spectrum(0.0f);
	unsigned int depth = 0;
	bool specBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while(k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
	{
		c *= Transmittance(r, 0, r2.m_fDist, -1);
		r2.getBsdfSample(r, rng, &bRec);
		/*
		DirectSamplingRecord dRecLight(r(r2.m_fDist), bRec.ng, bRec.map.uv);
		Spectrum le = g_SceneData.sampleEmitterDirect(dRecLight, rng.randomFloat2());
		DirectSamplingRecord dRecSensor(r(r2.m_fDist), bRec.ng, bRec.map.uv);
		Spectrum im = g_SceneData.sampleSensorDirect(dRecSensor, rng.randomFloat2());
		if(!g_SceneData.Occluded(Ray(r(r2.m_fDist), dRecLight.d), 0, dRecLight.dist))
		{
			return im * 1000000.0f;

			float3 wi = normalize(dRecLight.p - r(r2.m_fDist));
			float3 wo = normalize(dRecSensor.p - r(r2.m_fDist));
			bRec.wi = bRec.map.sys.toLocal(wo);
			bRec.wo = bRec.map.sys.toLocal(wi);
			Spectrum f = r2.getMat().bsdf.f(bRec);
			float pdf = r2.getMat().bsdf.pdf(bRec);
			//return absdot(wi, bRec.map.sys.n) * absdot(wo, bRec.map.sys.n) * f / pdf * le;
			float pdf2 = 1.0f / (absdot(wo, bRec.map.sys.n) * absdot(wo, bRec.map.sys.n) * absdot(wo, bRec.map.sys.n)) * 1.0f / (dRecLight.dist * dRecLight.dist);
			return le * f / (pdf / pdf2);
		}
		else return 0.0f;*/
		//return Spectrum(dot(bRec.ng, -r.direction));
		if(depth == 1 || specBounce || !DIRECT)
			L += r2.Le(r(r2.m_fDist), bRec.dg.sys, -r.direction);
		if(DIRECT)
			L += c * UniformSampleAllLights(bRec, r2.getMat(), 1);
		float pdf;
		Spectrum f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());

		float p = f.max();
		if (rng.randomFloat() < p)
			f = f / p;
		else break;

		c = c * f;
		if((bRec.sampledType & EDiffuse) == EDiffuse)
		{
			L += c;
			break;
		}
		specBounce = (bRec.sampledType & EDelta) != 0;
		r.origin = r(r2.m_fDist);
		r.direction = bRec.getOutgoing();
		r2.Init();
	}
	if(!r2.hasHit())
		L += c * g_SceneData.EvalEnvironment(r);
	return L;
}

CUDA_FUNC_IN Spectrum traceS(Ray& r, CudaRNG& rng)
{
	TraceResult r2 = k_TraceRay(r);
	if (!r2.hasHit())
		return Spectrum(0.0f);
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	r2.getBsdfSample(r, rng, &bRec);
	//Spectrum f = r2.getMat().bsdf.sample(bRec, make_float2(0.0f));
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
	bRec.wo = bRec.dg.toLocal(dRec.d);
	Spectrum f = r2.getMat().bsdf.f(bRec);
	if (r2.getMat().bsdf.hasComponent(ETypeCombinations::EDiffuse))
		return f * ::V(bRec.dg.P, dRec.p) / r2.getMat().bsdf.pdf(bRec) * Frame::cosTheta(bRec.wo);
	else if (r2.getMat().bsdf.hasComponent(ETypeCombinations::EDelta))
		return ::V(bRec.dg.P, dRec.p);
	else return r2.getMat().bsdf.sample(bRec, Vec2f(0.0f)) * Frame::cosTheta(bRec.wo) * ::V(bRec.dg.P, dRec.p);
}

__global__ void primaryKernel(long long width, long long height, e_Image g_Image)
{
	CudaRNG rng = g_RNGData();
	int rayidx;
	int N = width * height;
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
				rayBase = atomicAdd(&g_NextRayCounter2, numTerminated);

            rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
                break;
		}
		unsigned int x = rayidx % width, y = rayidx / width;
		Vec2f pixelSample = Vec2f(x, y);
		
		Spectrum c = Spectrum(0.0f);
		float N2 = 1;
		for(float f = 0; f < N2; f++)
		{
			Ray r, rX, rY;
//			Spectrum imp = g_SceneData.sampleSensorRay(r, pixelSample, rng.randomFloat2());
			Spectrum imp = g_SceneData.m_Camera.sampleRayDifferential(r, rX, rY, pixelSample, rng.randomFloat2());
			c += imp * trace(r, rX, rY, rng);
			//c += imp * traceS(r, rng);
		}
		g_Image.AddSample(x, y, c / N2);
		
		//Ray r = g_CameraData.GenRay(x, y, width, height, rng.randomFloat(), rng.randomFloat());
		//TraceResult r2 = k_TraceRay(r);
		//float3 c = make_float3(r2.m_fDist/length(g_SceneData.m_sBox.Size())*2.0f);
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixe2l(unsigned int width, unsigned int height, Vec2i p)
{
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);
	CudaRNG rng = g_RNGData();
	Spectrum q =  trace(r, r, r, rng);
}

void k_PrimTracer::DoRender(e_Image* I)
{
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter2, &zero, sizeof(unsigned int));
	primaryKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, *I);
}

void k_PrimTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	//FW::printf("%f,%f",pixel.x/float(w),pixel.y/float(h));
	k_INITIALIZE(m_pScene, g_sRngs);
	//debugPixe2l<<<1,1>>>(w,h,pixel);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
	trace(r, rX, rY, rng);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}