#include "k_PrimTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE e_Image g_DepthImage2;
CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter2;

CUDA_FUNC_IN Spectrum trace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rng, float& depth)
{
	TraceResult r2 = k_TraceRay(r);
	if(r2.hasHit())
	{
		depth = CalcZBufferDepth(g_SceneData.m_Camera.As()->m_fNearFarDepths.x, g_SceneData.m_Camera.As()->m_fNearFarDepths.y, r2.m_fDist);
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
		dg.computePartials(r, rX, rY);
		//return Spectrum(dg.dvdx, dg.dvdy, 0);
		//Vec3f n = (bRec.dg.n + Vec3f(1)) / 2;
		//return Spectrum(n.x, n.y, n.z);
		return Spectrum(math::clamp01(dot(bRec.dg.n, -normalize(r.direction))));
		Spectrum through = Transmittance(r, 0, r2.m_fDist);
		Spectrum L = 0.0f;// r2.Le(r(r2.m_fDist), bRec.dg.sys, -r.direction);
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
				r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
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
		c *= Transmittance(r, 0, r2.m_fDist);
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
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
			L += c * UniformSampleAllLights(bRec, r2.getMat(), 1, rng);
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
	r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
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

texture<float2, 2, cudaReadModeElementType> t_ConeMap;
__device__ Spectrum traceTerrain(Ray& r, CudaRNG& rng)
{
	const int cone_steps = 1024;
	const int binary_steps = 32;

	AABB box(Vec3f(0), Vec3f(1));
	float t_min, t_max;
	if (!box.Intersect(r, &t_min, &t_max))
		return Spectrum(0, 0, 1);

	r.origin = r(t_min + 1.0f / 128.0f);
	float ray_ratio = length(Vec2f(r.direction.x, r.direction.z));
	Vec3f pos = r.origin;
	float lastD;
	bool leftBox = false, currentOverTerrain = true; int N = 0;
	while (N++ < cone_steps)
	{
		const float e = 0;
		leftBox = (pos.x < e || pos.x > 1 - e || pos.z < e || pos.z > 1 - e);
		float2 tex = tex2D(t_ConeMap, pos.x, pos.z);
		currentOverTerrain = tex.x <= pos.y;
		if (leftBox || !currentOverTerrain)
			break;
		float c = tex.y;
		float h = (pos.y - tex.x);
		float d = lastD = c*h / (ray_ratio - c * r.direction.y);
		pos += r.direction * d;
	}
	if (leftBox )//|| N > cone_steps - 2
		return Spectrum(0, 1, 0);
	//return length(pos - r.origin);

	for (int i = 0; i < binary_steps; i++)
	{
		float2 tex = tex2D(t_ConeMap, pos.x, pos.z);
		lastD *= 0.5f;
		if (tex.x >= pos.y)
			pos -= lastD * r.direction;
		else pos += lastD * r.direction;
	}

	return length(pos - r.origin);
}

__global__ void primaryKernel(long long width, long long height, e_Image g_Image, bool depthImage)
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
		float d = 1;
		for(float f = 0; f < N2; f++)
		{
			Ray r, rX, rY;
//			Spectrum imp = g_SceneData.sampleSensorRay(r, pixelSample, rng.randomFloat2());
			Spectrum imp = g_SceneData.m_Camera.sampleRayDifferential(r, rX, rY, pixelSample, rng.randomFloat2());
			//c += imp * traceTerrain(r, rng);
			c += imp * trace(r, rX, rY, rng, d);
		}

		g_Image.AddSample(x, y, c / N2);
		if (depthImage)
			g_DepthImage2.SetSample(x, y, *(RGBCOL*)&d);
		
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
	float d;
	Spectrum q =  trace(r, r, r, rng, d);
}

//static e_KernelMIPMap mimMap;
//static size_t pitch;
void k_PrimTracer::DoRender(e_Image* I)
{
	if (depthImage)
	{
		cudaMemcpyToSymbol(g_DepthImage2, depthImage, sizeof(e_Image));
		depthImage->StartRendering();
	}

	//t_ConeMap.normalized = true;
	//t_ConeMap.addressMode[0] = t_ConeMap.addressMode[1] = cudaAddressModeClamp;
	//t_ConeMap.sRGB = false;
	//cudaError_t	r = cudaBindTexture2D(0, &t_ConeMap, mimMap.m_pDeviceData, &t_ConeMap.channelDesc, mimMap.m_uWidth, mimMap.m_uHeight, pitch);
	//ThrowCudaErrors(r);

	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter2, &zero, sizeof(unsigned int));
	primaryKernel << < 180, dim3(32, MaxBlockHeight, 1) >> >(w, h, *I, depthImage ? 1 : 0);
	if (depthImage)
		depthImage->EndRendering();
}

void k_PrimTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//debugPixe2l<<<1,1>>>(w,h,pixel);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
	float d;
	trace(r, rX, rY, rng, d);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}

k_PrimTracer::k_PrimTracer()
	: m_bDirect(false), depthImage(0)
{
	//const char* QQ = "../Data/tmp.dat";
	//OutputStream out(QQ);
	//e_MIPMap::CreateRelaxedConeMap("../Data/1.bmp", out);
	//out.Close();

	//InputStream in(QQ);
	//heightMap = e_MIPMap(in);
	//in.Close();
	//
	//mimMap = heightMap.getKernelData();
	//pitch = mimMap.m_uWidth * 8;
	//CUDA_FREE(mimMap.m_pDeviceData);
	//cudaError_t r = cudaMallocPitch(&mimMap.m_pDeviceData, &pitch, mimMap.m_uWidth, mimMap.m_uHeight);
	//ThrowCudaErrors(r);
	//r = cudaMemcpy2D(mimMap.m_pDeviceData, pitch, mimMap.m_pHostData, mimMap.m_uWidth * 4, mimMap.m_uWidth, mimMap.m_uHeight, cudaMemcpyHostToDevice);
	//ThrowCudaErrors(r);
}