#include "k_PrimTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include "../Base/FileStream.h"
#include "../CudaMemoryManager.h"
#include "../Engine/e_FileTexture.h"

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
		return Spectrum(absdot(dg.n, -r.direction));
		//return Spectrum(dg.dvdx, dg.dvdy, 0);
		//Vec3f n = (bRec.dg.sys.n + Vec3f(1)) / 2;
		//return Spectrum(n.x, n.y, n.z);
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
			through *= Transmittance(r, 0, r2.m_fDist);
			if(r2.hasHit())
			{
				r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
				//f *= r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				through *= r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			}
			else break;
		}
		return through * (r2.hasHit() ? UniformSampleOneLight(bRec, r2.getMat(), rng) : Spectrum(1.0f));
	}
	else return g_SceneData.EvalEnvironment(r, rX, rY);
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

CUDA_FUNC_IN Spectrum traceGame(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rng, float& zDepth)
{
	TraceResult r2 = k_TraceRay(r);
	if (r2.hasHit())
	{
		zDepth = r2.m_fDist;
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
		dg.computePartials(r, rX, rY);

		Spectrum through = Transmittance(r, 0, r2.m_fDist);
		Spectrum Le_primary_hit = r2.Le(dg.P, bRec.dg.sys, -r.direction);

		int depth = 0;
		while (r2.getMat().bsdf.hasComponent(EDelta) && depth++ < 5)
		{
			zDepth = FLT_MAX;
			Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = Ray(dg.P, bRec.getOutgoing());
			r2 = k_TraceRay(r);
			if (!r2.hasHit())
				return through * f * g_SceneData.EvalEnvironment(r);
			else
			{
				r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
				through *= f * Transmittance(r, 0, r2.m_fDist);
			}
		}

		Spectrum accu(0.0f);
		int N = 2;
		for (int i = 0; i < N; i++)
		{
			Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			Ray r_indicrect = Ray(dg.P, bRec.getOutgoing());
			TraceResult r2_indicrect = k_TraceRay(r_indicrect);
			Spectrum through_indicrect = Transmittance(r_indicrect, 0, r2_indicrect.m_fDist);
			Spectrum L_indirect(0.0f);
			if (r2_indicrect.hasHit())
			{
				DifferentialGeometry dg_indicrect;
				BSDFSamplingRecord bRec_indicrect(dg_indicrect);
				r2_indicrect.getBsdfSample(r_indicrect, bRec_indicrect, ETransportMode::ERadiance, &rng);
				L_indirect = UniformSampleOneLight(bRec_indicrect, r2_indicrect.getMat(), rng);
			}
			else
			{
				L_indirect = g_SceneData.EvalEnvironment(r_indicrect);
			}
			accu += f * through_indicrect * L_indirect;
		}

		return through * (Le_primary_hit + accu / float(N));
	}
	else return g_SceneData.EvalEnvironment(r, rX, rY);
}

texture<float2, 2, cudaReadModeElementType> t_ConeMap;
float2* t_ConeMapHost;
int t_ConeMapWidth, t_ConeMapHeight;
CUDA_FUNC_IN float2 sample_tex(const Vec3f& pos)
{
#ifdef ISCUDA
	float2 tex = tex2D(t_ConeMap, pos.x, pos.z);
#else
	float2 tex = t_ConeMapHost[t_ConeMapWidth * int(pos.z * t_ConeMapHeight) + int(pos.x * t_ConeMapWidth)];
#endif
	return tex;
}
CUDA_FUNC_IN Vec3f sample_normal(const Vec3f& pos)
{
	const float o = 1.0f / 2048;
	float l = sample_tex(pos - Vec3f(o, 0, 0)).x, r = sample_tex(pos + Vec3f(o, 0, 0)).x;
	float d = sample_tex(pos - Vec3f(0, o, 0)).x, u = sample_tex(pos + Vec3f(0, o, 0)).x;

	Vec3f va = normalize(Vec3f(2*o, 0, r - l));
	Vec3f vb = normalize(Vec3f(0, 2*o, u - d));
	return normalize(cross(va, vb));

	//if (r - l != 0)
	//	return Vec3f(1,-1/(r-l),(u-d)/(r-l)).normalized();
	//if (u - d != 0)
	//	return Vec3f((r-l)/(u-d),-1/(u-d),1).normalized();
	//return Vec3f(0,1,0);
}
CUDA_FUNC_IN Spectrum traceTerrain(Ray& r, CudaRNG& rng)
{
	const int cone_steps = 1024;
	const int binary_steps = 32;

	AABB box(Vec3f(0), Vec3f(1));
	float t_min, t_max;
	if (!box.Intersect(r, &t_min, &t_max))
		return Spectrum(0, 0, 1);

	r.origin = r(t_min + 5e-2f);
	float ray_ratio = length(Vec2f(r.direction.x, r.direction.z));
	Vec3f pos = r.origin;
	float lastD;
	bool leftBox = false, currentOverTerrain = true; int N = 0;
	while (N++ < cone_steps)
	{
		const float e = 0, e2 = 1e-2f;
		leftBox = (pos.x < e || pos.x > 1 - e || pos.z < e || pos.z > 1 - e);
		float2 tex = sample_tex(pos);
		leftBox |= tex.y < 1e3f && (((pos.x < e2 && pos.x >= 0) || (pos.x > 1 - e2 && pos.x <= 1) || (pos.z < e2 && pos.z >= 0) || (pos.z > 1 - e2 && pos.z <= 1)));
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
		float2 tex = sample_tex(pos);
		lastD *= 0.5f;
		if (tex.x >= pos.y)
			pos -= lastD * r.direction;
		else pos += lastD * r.direction;
	}

	return -dot(sample_normal(pos), r.direction);//length(pos - r.origin);
}

__global__ void primaryKernel(int width, int height, e_Image g_Image, bool depthImage)
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
		int x = rayidx % width, y = rayidx / width;

		Ray r, rX, rY;
		Spectrum imp = g_SceneData.m_Camera.sampleRayDifferential(r, rX, rY, Vec2f(x, y), rng.randomFloat2());
		float d;
		Spectrum L = imp * trace(r, rX, rY, rng, d);
			
		g_Image.AddSample(x, y, L);
		if (depthImage)
			g_DepthImage2.SetSample(x, y, *(RGBCOL*)&d);
	}
	while(true);
	g_RNGData(rng);
}

#define BLOCK_SIZE 16
__global__ void primaryKernelBlocked(int width, int height, e_Image g_Image, bool depthImage, Spectrum* lastImage1, Spectrum* lastImage2, e_Sensor lastSensor, int nIteration)
{
	CudaRNG rng = g_RNGData();
	int x = 2*(blockIdx.x * blockDim.x + threadIdx.x), y = 2*(blockIdx.y * blockDim.y + threadIdx.y);
	if (x < width && y < height)
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);

		Ray primaryRay;
		TraceResult primaryRes;
		Spectrum Le(0.0f);
		Spectrum primaryThrough[4];
		for (int i = 0; i < 4; i++)
		{
			int x2 = x + i % 2, y2 = y + i / 2;
			primaryThrough[i] = g_SceneData.m_Camera.sampleRay(primaryRay, Vec2f(x2, y2), rng.randomFloat2());
			primaryRes = k_TraceRay(primaryRay);
			if (primaryRes.hasHit())
			{
				primaryRes.getBsdfSample(primaryRay, bRec, ETransportMode::ERadiance, &rng);
				Le = primaryRes.Le(dg.P, bRec.dg.sys, -primaryRay.direction);
				primaryThrough[i] *= primaryRes.getMat().bsdf.sample(bRec, rng.randomFloat2());
				float d = CalcZBufferDepth(g_SceneData.m_Camera.As()->m_fNearFarDepths.x, g_SceneData.m_Camera.As()->m_fNearFarDepths.y, primaryRes.m_fDist);
				if (depthImage && x2 < width && y2 < height)
					g_DepthImage2.SetSample(x2, y2, *(RGBCOL*)&d);
			}
			else primaryThrough[i] = g_SceneData.EvalEnvironment(primaryRay);
		}

		Spectrum through_indicrect(1.0f);
		Spectrum L_indirect(0.0f);
		if (primaryRes.hasHit())
		{
			int N_INDIRECT = 4;
			for (int i = 0; i < N_INDIRECT; i++)
			{
				Ray r_indicrect = Ray(dg.P, bRec.getOutgoing());
				TraceResult r2_indicrect = k_TraceRay(r_indicrect);
				through_indicrect = Transmittance(r_indicrect, 0, r2_indicrect.m_fDist);
				if (r2_indicrect.hasHit())
				{
					DifferentialGeometry dg_indicrect;
					BSDFSamplingRecord bRec_indicrect(dg_indicrect);
					r2_indicrect.getBsdfSample(r_indicrect, bRec_indicrect, ETransportMode::ERadiance, &rng);
					L_indirect += UniformSampleOneLight(bRec_indicrect, r2_indicrect.getMat(), rng);
				}
				else
				{
					L_indirect += g_SceneData.EvalEnvironment(r_indicrect);
				}
			}
			L_indirect /= N_INDIRECT;

			/*if (nIteration > 2)
			{
				DirectSamplingRecord dRec(primaryRay(primaryRes.m_fDist), Vec3f(0.0f));
				lastSensor.sampleDirect(dRec, Vec2f(0, 0));
				if (dRec.pdf)
				{
					int oy = int(dRec.uv.y) / 2, ox = int(dRec.uv.x) / 2, w2 = width / 2;
					Spectrum lu = lastImage1[oy * w2 + ox], ru = lastImage1[oy * w2 + min(w2 - 1, ox + 1)],
						ld = lastImage1[min(height/2 - 1, oy + 1) * w2 + ox], rd = lastImage1[min(height/2 - 1, oy + 1) * w2 + min(w2 - 1, ox + 1)];
					Spectrum lL = math::bilerp(dRec.uv - dRec.uv.floor(), lu, ru, ld, rd);
					//Spectrum lL = lastImage1[oy * w2 + ox];
					//Spectrum lL = (lu + ru + ld + rd) / 4.0f;
					const float p = 0.5f;
					L_indirect = p * lL + (1 - p) * L_indirect;
				}
			}
			lastImage2[y / 2 * width / 2 + x / 2] = L_indirect;*/
		}
		else lastImage2[y / 2 * width / 2 + x / 2] = primaryThrough[0];

		/*CUDA_SHARED Spectrum accuData;
		accuData = Spectrum(0.0f);
		__syncthreads();
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			atomicAdd(&accuData[i], L_indirect[i]);
		__syncthreads();
		L_indirect = accuData / (blockDim.x * blockDim.y);*/

		/*CUDA_SHARED RGBCOL indirectData[BLOCK_SIZE*BLOCK_SIZE];
		indirectData[threadIdx.y * blockDim.x + threadIdx.x] = L_indirect.toRGBCOL();
		__syncthreads();
		int filterW = 2;
		e_GaussianFilter filt(filterW*2, filterW*2, 0.5f);
		int xstart = math::clamp((int)threadIdx.x - filterW / 2, filterW / 2+1, BLOCK_SIZE - filterW / 2-1),
			ystart = math::clamp((int)threadIdx.y - filterW / 2, filterW / 2+1, BLOCK_SIZE - filterW / 2-1);
		Spectrum filterVal(0.0f);
		float filterWeight = 0.0f;
		for (int i = -filterW/2; i <= filterW/2; i++)
			for (int j = -filterW / 2; j <= filterW / 2; j++)
			{
				float f = filt.Evaluate(i,j);
				Spectrum v;
				v.fromRGBCOL(indirectData[(ystart + j) * blockDim.x + xstart + i]);
				filterVal += v * f;
				filterWeight += f;
			}
		L_indirect = filterVal / filterWeight;*/

		for (int i = 0; i < 4; i++)
		{
			Spectrum L = primaryRes.hasHit() ? Transmittance(primaryRay, 0, primaryRes.m_fDist) * (Le + primaryThrough[i] * through_indicrect * L_indirect) : primaryThrough[i];
			int x2 = x + i % 2, y2 = y + i / 2;
			if (x2 < width && y2 < height)
				g_Image.AddSample(x2, y2, L);
		}
	}
	g_RNGData(rng);
}

__global__ void debugPixe2l(unsigned int width, unsigned int height, Vec2i p)
{
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);
	CudaRNG rng = g_RNGData();
	float d;
	Spectrum q =  trace(r, r, r, rng, d);
	q = traceTerrain(r, rng);
}

static e_KernelMIPMap mimMap;
static size_t pitch;
static int iterations = 0;
void k_PrimTracer::DoRender(e_Image* I)
{
	if (depthImage)
	{
		cudaMemcpyToSymbol(g_DepthImage2, depthImage, sizeof(e_Image));
		depthImage->StartRendering();
	}

	/*t_ConeMap.normalized = true;
	t_ConeMap.addressMode[0] = t_ConeMap.addressMode[1] = cudaAddressModeClamp;
	t_ConeMap.sRGB = false;
	t_ConeMap.filterMode = cudaFilterModeLinear;
	cudaError_t	r = cudaBindTexture2D(0, &t_ConeMap, mimMap.m_pDeviceData, &t_ConeMap.channelDesc, mimMap.m_uWidth, mimMap.m_uHeight, pitch);
	ThrowCudaErrors(r);
	t_ConeMapHost = (float2*)mimMap.m_pHostData;
	t_ConeMapWidth = mimMap.m_uWidth;
	t_ConeMapHeight = mimMap.m_uHeight;*/

	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter2, &zero, sizeof(unsigned int));
	primaryKernel << < 180, dim3(32, MaxBlockHeight, 1) >> >(w, h, *I, depthImage ? 1 : 0);
	swapk(&m_pDeviceLastImage1, &m_pDeviceLastImage2);
	//primaryKernelBlocked << <dim3(w / (2 * BLOCK_SIZE) + 1, h / (2 * BLOCK_SIZE) + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(w, h, *I, depthImage ? 1 : 0, m_pDeviceLastImage1, m_pDeviceLastImage2, lastSensor, iterations++);
	if (depthImage)
		depthImage->EndRendering();
	lastSensor = g_SceneData.m_Camera;
}

void k_PrimTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//debugPixe2l<<<1,1>>>(w,h,pixel);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
	float d;
	traceGame(r, rX, rY, rng, d);
	//traceTerrain(r, rng);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}

k_PrimTracer::k_PrimTracer()
	: m_bDirect(false), depthImage(0), m_pDeviceLastImage1(0), m_pDeviceLastImage2(0)
{
	const char* QQ = "../Data/tmp.dat";
	//OutputStream out(QQ);
	//e_MIPMap::CreateRelaxedConeMap("../Data/1.bmp", out);
	//out.Close();

	/*InputStream in(QQ);
	auto heightMap = e_MIPMap(in.getFilePath(), in);
	in.Close();
	//
	mimMap = heightMap.getKernelData();
	pitch = 0;
	CUDA_FREE(mimMap.m_pDeviceData);
	cudaError_t r = cudaMallocPitch(&mimMap.m_pDeviceData, &pitch, mimMap.m_uWidth * 8, mimMap.m_uHeight);
	ThrowCudaErrors(r);
	r = cudaMemcpy2D(mimMap.m_pDeviceData, pitch, mimMap.m_pHostData, mimMap.m_uWidth * 8, mimMap.m_uWidth, mimMap.m_uHeight, cudaMemcpyHostToDevice);
	ThrowCudaErrors(r);*/
}

void k_PrimTracer::Resize(unsigned int _w, unsigned int _h)
{
	if (m_pDeviceLastImage1)
		CUDA_FREE(m_pDeviceLastImage1);
	if (m_pDeviceLastImage2)
		CUDA_FREE(m_pDeviceLastImage2);
	CUDA_MALLOC(&m_pDeviceLastImage1, _w * _h * sizeof(Spectrum));
	CUDA_MALLOC(&m_pDeviceLastImage2, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastImage1, 0, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastImage2, 0, _w * _h * sizeof(Spectrum));
	Platform::SetMemory(&lastSensor, sizeof(lastSensor));
	iterations = 0;
	k_Tracer<false, false>::Resize(_w, _h);
}