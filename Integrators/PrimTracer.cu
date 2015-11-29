#include "PrimTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Base/FileStream.h>
#include <CudaMemoryManager.h>
#include <Engine/MIPMap.h>

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE DeviceDepthImage g_DepthImage2;
CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter2;

CUDA_FUNC_IN void computePixel(int x, int y, CudaRNG& rng, Image g_Image, bool depthImage, PrimTracer::DrawMode mode, int maxPathLength)
{
	Ray r, rX, rY;
	Spectrum imp = g_SceneData.m_Camera.sampleRayDifferential(r, rX, rY, Vec2f(x, y), rng.randomFloat2());
	TraceResult prim_res = Traceray(r);
	Spectrum L(0.0f), through(1.0f);
	if (prim_res.hasHit())
	{
		if (mode == PrimTracer::DrawMode::linear_depth)
		{
			Vec2f nf = g_SceneData.m_Camera.As()->m_fNearFarDepths;
			L = Spectrum((prim_res.m_fDist - nf.x) / (nf.y - nf.x));
		}
		else if (mode == PrimTracer::DrawMode::D3D_depth)
			L = Spectrum(DeviceDepthImage::NormalizeDepthD3D(prim_res.m_fDist));
		else
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			prim_res.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			dg.computePartials(r, rX, rY);

			if (mode == PrimTracer::DrawMode::v_absdot_n_geo)
				L = Spectrum(absdot(-r.direction, dg.n));
			else if (mode == mode == PrimTracer::DrawMode::v_dot_n_geo)
				L = Spectrum(dot(-r.direction, dg.n));
			else if (mode == PrimTracer::DrawMode::v_dot_n_shade)
				L = Spectrum(dot(-r.direction, dg.sys.n));
			else if (mode == PrimTracer::DrawMode::n_geo_colored || mode == PrimTracer::DrawMode::n_shade_colored)
			{
				Vec3f n = ((mode == PrimTracer::DrawMode::n_geo_colored ? bRec.dg.n : bRec.dg.sys.n) + Vec3f(1)) / 2;
				L = Spectrum(n.x, n.y, n.z);
			}
			else if (mode == PrimTracer::DrawMode::uv)
				L = Spectrum(dg.uv[0].x, dg.uv[0].y, 0);
			else if (mode == PrimTracer::DrawMode::bary_coords)
				L = Spectrum(dg.bary.x, dg.bary.y, 0);
			else
			{
				Spectrum Le = prim_res.Le(dg.P, bRec.dg.sys, -r.direction);
				Spectrum f = prim_res.getMat().bsdf.sample(bRec, rng.randomFloat2());
				through = Transmittance(r, 0, prim_res.m_fDist);
				bool isDelta = prim_res.getMat().bsdf.hasComponent(EDelta);
				if (mode == PrimTracer::DrawMode::first_Le || (!isDelta && mode == PrimTracer::DrawMode::first_non_delta_Le))
					L = Le;
				else if (mode == PrimTracer::DrawMode::first_f || (!isDelta && mode == PrimTracer::DrawMode::first_non_delta_f))
					L = f;
				else if (mode == PrimTracer::DrawMode::first_f_direct || (!isDelta && mode == PrimTracer::DrawMode::first_non_delta_f_direct))
					L = Le + through * (UniformSampleOneLight(bRec, prim_res.getMat(), rng) + f * 0.5f);
				else
				{
					through *= f;
					int depth = 0;
					do
					{
						r = Ray(dg.P, bRec.getOutgoing());
						prim_res = Traceray(r);
						through *= Transmittance(r, 0, prim_res.m_fDist);
						if (prim_res.hasHit())
						{
							prim_res.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
							f = prim_res.getMat().bsdf.sample(bRec, rng.randomFloat2());
							if (!prim_res.getMat().bsdf.hasComponent(ESmooth))
								through *= f;
						}
					} while (depth++ < maxPathLength && prim_res.hasHit() && !prim_res.getMat().bsdf.hasComponent(ESmooth));

					if (prim_res.hasHit() && prim_res.getMat().bsdf.hasComponent(ESmooth))
					{
						Le = prim_res.Le(dg.P, bRec.dg.sys, -r.direction);
						if (mode == PrimTracer::DrawMode::first_non_delta_Le)
							L = Le;
						else if (mode == PrimTracer::DrawMode::first_non_delta_f)
							L = f;
						else if (mode == PrimTracer::DrawMode::first_non_delta_f_direct)
							L = Le + through * (UniformSampleOneLight(bRec, prim_res.getMat(), rng) + f * 0.5f);
					}
					else;
				}
			}
		}
	}
	else;
	g_Image.AddSample(x, y, L);
	if (depthImage)
		g_DepthImage2.Store(x, y, prim_res.m_fDist);
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

	Vec3f va = normalize(Vec3f(2 * o, 0, r - l));
	Vec3f vb = normalize(Vec3f(0, 2 * o, u - d));
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
	if (leftBox)//|| N > cone_steps - 2
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

__global__ void primaryKernel(int width, int height, Image g_Image, bool depthImage, PrimTracer::DrawMode mode, int maxPathLength)
{
	CudaRNG rng = g_RNGData();
	int rayidx;
	int N = width * height;
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		const bool          terminated = 1;//nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(&g_NextRayCounter2, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
				break;
		}
		int x = rayidx % width, y = rayidx / width;
		computePixel(x, y, rng, g_Image, depthImage, mode, maxPathLength);
	} while (true);
	g_RNGData(rng);
}

//static KernelMIPMap mimMap;
//static size_t pitch;
//static int iterations = 0;
void PrimTracer::DoRender(Image* I)
{
	if (hasDepthBuffer())
		CopyToSymbol(g_DepthImage2, getDeviceDepthBuffer());

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
	primaryKernel << < 180, dim3(32, MaxBlockHeight, 1) >> >(w, h, *I, hasDepthBuffer(), m_sParameters.getValue(KEY_DrawingMode()), m_sParameters.getValue(KEY_MaxPathLength()));
}

void PrimTracer::Debug(Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
	//traceTerrain(r, rng);
}

void PrimTracer::CreateSliders(SliderCreateCallback a_Callback) const
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}

PrimTracer::PrimTracer()
{
	m_sParameters << KEY_DrawingMode() << CreateInterval<DrawMode>(DrawMode::first_non_delta_f_direct, DrawMode::linear_depth, DrawMode::first_non_delta_f_direct)
				  << KEY_MaxPathLength() << CreateInterval<int>(7, 1, INT_MAX);
	//const char* QQ = "../Data/tmp.dat";
	//OutputStream out(QQ);
	//MIPMap::CreateRelaxedConeMap("../Data/1.bmp", out);
	//out.Close();

	/*InputStream in(QQ);
	auto heightMap = MIPMap(in.getFilePath(), in);
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



}