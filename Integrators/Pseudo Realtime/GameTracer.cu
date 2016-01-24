#include "GameTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <CudaMemoryManager.h>
#include <Engine/Light.h>

namespace CudaTracerLib {

CUDA_DEVICE DeviceDepthImage g_DepthImageGT;

#define INDIRECT_SCALE 1

#define BLOCK_SIZE 16
__global__ void primaryKernelBlocked(int width, int height, Image g_Image, bool depthImage, 
									 Spectrum* lastDirectImage, Spectrum* nextDirectImage, Spectrum* lastIndirectImage, Spectrum* nextIndirectImage,
									 Sensor lastSensor, int nIteration)
{
	CudaRNG rng = g_RNGData();
	int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x), y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
	if (x < width && y < height)
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		Ray primaryRay;
		TraceResult primaryRes;

		Spectrum primary_Le(0.0f);//average of all 4 primary emittances
		Spectrum primary_f[4];//this includes W_e and volume transmittance
		Vec2f uv_sets[4];
		int primary_rays_hit = 0;
		for (int i = 0; i < 4; i++)
		{
			int x2 = x + i % 2, y2 = y + i / 2;
			Ray r_i;
			primary_f[i] = g_SceneData.m_Camera.sampleRay(r_i, Vec2f(x2, y2), rng.randomFloat2());
			auto r2_i = traceRay(r_i);
			primary_f[i] *= Transmittance(r_i, 0, r2_i.m_fDist);
			if (r2_i.hasHit())
			{
				primaryRay = r_i;
				primaryRes = r2_i;
				primary_rays_hit++;
				primaryRes.getBsdfSample(primaryRay, bRec, ETransportMode::ERadiance, &rng);
				uv_sets[i] = dg.uv[0];
				primary_Le += primaryRes.Le(dg.P, bRec.dg.sys, -primaryRay.dir());
			}
			else primary_f[i] = g_SceneData.EvalEnvironment(primaryRay);
			if (depthImage && x2 < width && y2 < height)
				g_DepthImageGT.Store(x2, y2, primaryRes.m_fDist);
		}
		if (!primary_rays_hit)
		{
			nextDirectImage[y / 2 * width / 2 + x / 2] = nextIndirectImage[y / 2 * width / 2 + x / 2] = Spectrum(0.0f);
			g_RNGData(rng);
			return;
		}
		primary_Le /= primary_rays_hit;

		//do N_d next event estimation samples using the same input geometry
		const int N_d = 8;
		Spectrum Est_Ld(0.0f);
		int direct_sampling_succes = 0;
		Vec3f direct_sampling_point(0.0f);
		for (int i = 0; i < N_d; i++)
		{
			float pdf;
			Vec2f sample = rng.randomFloat2();
			const Light* light = g_SceneData.sampleEmitter(pdf, sample);
			if (!light) continue;
			DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
			Spectrum value = light->sampleDirect(dRec, sample) / pdf;
			if (!value.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
			{ 
				direct_sampling_succes++;
				direct_sampling_point += dRec.p;
				Est_Ld += value * Transmittance(Ray(dRec.ref, dRec.d), 0, dRec.dist);
			}
		}
		Est_Ld /= direct_sampling_succes;
		direct_sampling_point /= direct_sampling_succes;

		//do N_i samples of indirect lighting = sample random point in the scene, compute primary f for each pixel and compute do next event estimation
		const int N_i = 4;
		Spectrum Est_Li[4];
		for (int i = 0; i < 4; i++)
			Est_Li[i] = Spectrum(0.0f);
		for (int i = 0; i < N_i; i++)
		{
			primaryRes.getMat().bsdf.sample(bRec, rng.randomFloat2());
			Ray indirect_ray = Ray(dg.P, bRec.getOutgoing());
			auto indirect_res = traceRay(indirect_ray);
			if (indirect_res.hasHit())
			{
				DifferentialGeometry indirect_dg;
				BSDFSamplingRecord indirect_bRec(indirect_dg);
				indirect_res.getBsdfSample(indirect_ray, indirect_bRec, ETransportMode::ERadiance, &rng);
				float pdf;
				Vec2f sample = rng.randomFloat2();
				const Light* light = g_SceneData.sampleEmitter(pdf, sample);
				if (!light) continue;
				DirectSamplingRecord dRec(indirect_bRec.dg.P, indirect_bRec.dg.sys.n);
				Spectrum value = light->sampleDirect(dRec, sample) / pdf;
				if (!value.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
				{
					indirect_bRec.wo = indirect_bRec.dg.sys.toLocal(dRec.d);
					Spectrum indirect_f = indirect_res.getMat().bsdf.f(indirect_bRec);
					for (int j = 0; j < 4; j++)
					{
						dg.uv[0] = uv_sets[j];
						//not neccessary to set wo, still beeing set from sampling(different uv though...)
						Est_Li[j] += primaryRes.getMat().bsdf.f(bRec) * indirect_f * value / float(N_i);
					}
				}
			}
		}

		//for noise reduction average the result with the last frames results
		Spectrum avg_Est_Li(0.0f);
		for (int i = 0; i < 4; i++)
			avg_Est_Li += Est_Li[i] / 4.0f;

		int Inew = (y / 2) * (width / 2) + (x / 2);
		nextDirectImage[Inew] = Est_Ld;
		nextIndirectImage[Inew] = avg_Est_Li;

		if (nIteration > 2)
		{
			DirectSamplingRecord dRec(dg.P, NormalizedT<Vec3f>(0.0f));
			lastSensor.sampleDirect(dRec, Vec2f(0, 0));
			if (dRec.pdf)
			{
				const float p = 0.5f;
				int oy = int(dRec.uv.y)/2, ox = int(dRec.uv.x)/2;
#define LOAD(img, ax, ay) img[min(oy+ay,height/2-1) * (width / 2) +  min(ox+ax,width/2-1)]
				if (direct_sampling_succes)
				{
					Spectrum lu = LOAD(lastDirectImage, 0, 0), ru = LOAD(lastDirectImage, 1, 0),
							 ld = LOAD(lastDirectImage, 0, 1), rd = LOAD(lastDirectImage, 1, 1);
					Spectrum lastDirect = math::bilerp2(lu, ru, ld, rd, dRec.uv - dRec.uv.floor());
					if (lastDirect.max() != 0)
						Est_Ld = math::lerp(lastDirect, Est_Ld, p);
				}
				Spectrum lu = LOAD(lastIndirectImage, 0, 0), ru = LOAD(lastIndirectImage, 1, 0),
						 ld = LOAD(lastIndirectImage, 0, 1), rd = LOAD(lastIndirectImage, 1, 1);
				Spectrum lastIndirect = math::bilerp2(lu, ru, ld, rd, dRec.uv - dRec.uv.floor());
				if (lastIndirect.max() != 0)
					for (int i = 0; i < 4; i++)
						Est_Li[i] = math::lerp(lastIndirect, Est_Li[i], p);
#undef LOAD
			}
		}

		for (int i = 0; i < 4; i++)
		{
			Spectrum direct(0.0f);
			if (direct_sampling_succes)
			{
				dg.uv[0] = uv_sets[i];
				bRec.wo = normalize(bRec.dg.sys.toLocal(direct_sampling_point - dg.P));
				direct = Est_Ld * primaryRes.getMat().bsdf.f(bRec);
			}
			Spectrum L = primary_f[i] * (primary_Le + direct) + Est_Li[i] * INDIRECT_SCALE;
			int x2 = x + i % 2, y2 = y + i / 2;
			if (x2 < width && y2 < height)
				g_Image.AddSample(x2, y2, L);
		}
	}
	g_RNGData(rng);
}

void GameTracer::DoRender(Image* I)
{
	if (hasDepthBuffer())
		CopyToSymbol(g_DepthImageGT, getDeviceDepthBuffer());
	primaryKernelBlocked << <dim3(w / (2 * BLOCK_SIZE) + 1, h / (2 * BLOCK_SIZE) + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(w, h, *I, hasDepthBuffer(), 
		m_pDeviceLastDirectImage1, m_pDeviceLastDirectImage2, m_pDeviceLastIndirectImage1, m_pDeviceLastIndirectImage2, lastSensor, iterations++);
	lastSensor = g_SceneData.m_Camera;
	swapk(m_pDeviceLastDirectImage1, m_pDeviceLastDirectImage2);
	swapk(m_pDeviceLastIndirectImage1, m_pDeviceLastIndirectImage2);
}

void GameTracer::Debug(Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f((float)pixel.x, (float)pixel.y), rng.randomFloat2());
}

void GameTracer::Resize(unsigned int w, unsigned int h)
{
	if (m_pDeviceLastDirectImage1)
	{
		CUDA_FREE(m_pDeviceLastDirectImage1);
		CUDA_FREE(m_pDeviceLastDirectImage2);
		CUDA_FREE(m_pDeviceLastIndirectImage1);
		CUDA_FREE(m_pDeviceLastIndirectImage2);
	}
	unsigned int _w = w / 2, _h = h / 2;
	CUDA_MALLOC(&m_pDeviceLastDirectImage1, _w * _h * sizeof(Spectrum));
	CUDA_MALLOC(&m_pDeviceLastDirectImage2, _w * _h * sizeof(Spectrum));
	CUDA_MALLOC(&m_pDeviceLastIndirectImage1, _w * _h * sizeof(Spectrum));
	CUDA_MALLOC(&m_pDeviceLastIndirectImage2, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastDirectImage1, 0, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastDirectImage2, 0, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastIndirectImage1, 0, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastIndirectImage2, 0, _w * _h * sizeof(Spectrum));
	Platform::SetMemory(&lastSensor, sizeof(lastSensor));
	iterations = 0;
	Tracer<false, false>::Resize(w, h);
}

}