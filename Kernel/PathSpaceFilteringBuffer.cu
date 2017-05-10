#include "PathSpaceFilteringBuffer.h"
#include "Tracer.h"
#include "TraceAlgorithms.h"
#include <Engine/DynamicScene.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

void PathSpaceFilteringBuffer::PrepareForRendering(Image& I, DynamicScene* scene)
{
	auto eye_hit_points = scene->getSceneBox();// TracerBase::GetEyeHitPointBox(scene, false);
	m_hitPointBuffer.SetGridDimensions(eye_hit_points);

	m_settings.alpha_direct = m_settings.alpha_indirect = 1.0f;
	m_lastSensor = g_SceneData.m_Camera;
}

typedef PathSpaceFilteringBuffer::reuseInfo rInfo_t;
template<bool USE_DEPTH_IMAGE> CUDA_FUNC_IN void computePixel(Image& I, DeviceDepthImage& depthImg, PathSpaceFilteringBuffer& path_buffer, unsigned int x, unsigned int y, rInfo_t* accumBuffer_old, rInfo_t* accumBuffer_new, Sensor lastSensor)
{
	auto rng = g_SamplerData(y * I.getWidth() + x);
	NormalizedT<Ray> ray, rayX, rayY;
	auto W = g_SceneData.sampleSensorRay(ray, rayX, rayY, Vec2f((float)x, (float)y) + rng.randomFloat2(), rng.randomFloat2());
	auto res = traceRay(ray);
	int depth = 0;
	Spectrum L_emitted = 0.0f;
	while (res.hasHit() && depth++ < 4)
	{
		BSDFSamplingRecord bRec;
		res.getBsdfSample(ray, bRec, ERadiance);

		L_emitted += W * res.Le(bRec.dg.P, bRec.dg.sys, -ray.dir());

		if (depth == 1)
		{
			if (USE_DEPTH_IMAGE)
				depthImg.Store(x, y, res.m_fDist);
		}

		if (res.getMat().bsdf.hasComponent(EDelta) || res.getMat().bsdf.hasComponent(EGlossy))
		{
			W *= res.getMat().bsdf.sample(bRec, rng.randomFloat2());
			ray = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
			res = traceRay(ray);
		}
		else
		{
			Spectrum L_query = 0.0f;
			int n_found = 0;
			path_buffer.m_hitPointBuffer.ForAllCells(bRec.dg.P, bRec.dg.P, [&](const Vec3u& cell_idx)
			{
				auto& cell = path_buffer.m_hitPointBuffer(cell_idx);
				if (cell.N)
				{
					L_query += cell.L / cell.N;
					n_found++;
				}
			});
			if (n_found != 0)
			{
				bRec.wo = NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f);
				auto f_r = res.getMat().bsdf.f(bRec);
				L_query = L_query / (float)n_found * f_r;
			}

			Spectrum L_indirect = L_query;
			Spectrum L_direct = UniformSampleAllLights(bRec, res.getMat(), 1, rng);
			if (path_buffer.m_settings.use_prevFrames)
			{
				PathSpaceFilteringBuffer::reuseInfo& accum_new = accumBuffer_new[y * I.getWidth() + x];

				DirectSamplingRecord dRec(bRec.dg.P, NormalizedT<Vec3f>(0.0f));
				lastSensor.sampleDirect(dRec, Vec2f(0, 0));
				if (dRec.pdf)
				{
					int old_x = math::clamp((int)dRec.uv.x, 0, (int)I.getWidth()), old_y = math::clamp((int)dRec.uv.y, 0, (int)I.getHeight());
					auto accum_old = accumBuffer_old[old_y * I.getWidth() + old_x];
					float rel_d_diff = math::abs(accum_old.d.ToFloat() - res.m_fDist) / accum_old.d;
					if (rel_d_diff < 0.2f && dot(Uchar2ToNormalizedFloat3(accum_old.n), bRec.dg.n) > 0.5f)
					{
						Spectrum old_indirect, old_direct;
						old_indirect.fromRGBE(accum_old.indirect_col);
						old_direct.fromRGBE(accum_old.direct_col);
						if (n_found > 0)
							L_indirect = old_indirect * (1.0f - path_buffer.m_settings.alpha_indirect) + L_indirect * path_buffer.m_settings.alpha_indirect;
						L_direct = old_direct * (1.0f - path_buffer.m_settings.alpha_direct) + L_direct * path_buffer.m_settings.alpha_direct;
					}
				}

				accum_new.indirect_col = L_indirect.toRGBE();
				accum_new.direct_col = L_direct.toRGBE();
				accum_new.d = half(res.m_fDist);
				accum_new.n = NormalizedFloat3ToUchar2(bRec.dg.n);
			}

			I.AddSample(x, y, W * (L_emitted + L_direct + L_indirect));
			break;
		}
	}
}

template<bool USE_DEPTH_IMAGE> CUDA_GLOBAL void computePixelsKernel(Image I, DeviceDepthImage depthImg, PathSpaceFilteringBuffer path_buffer, rInfo_t* accumBuffer_old, rInfo_t* accumBuffer_new, Sensor prevSensor)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < I.getWidth() && y < I.getHeight())
		computePixel<USE_DEPTH_IMAGE>(I, depthImg, path_buffer, x, y, accumBuffer_old, accumBuffer_new, prevSensor);
}

void PathSpaceFilteringBuffer::ComputePixelValues(Image& I, DynamicScene* scene, DeviceDepthImage* depthImage)
{
	UpdateKernel(scene);
	int p0 = 16;
	if(depthImage)
		computePixelsKernel<true> << <dim3(I.getWidth() / p0 + 1, I.getHeight() / p0 + 1, 1), dim3(p0, p0, 1) >> >(I, *depthImage, *this, m_accumBuffer1, m_accumBuffer2, m_lastSensor);
	else computePixelsKernel<false> << <dim3(I.getWidth() / p0 + 1, I.getHeight() / p0 + 1, 1), dim3(p0, p0, 1) >> >(I, DeviceDepthImage(), *this, m_accumBuffer1, m_accumBuffer2, m_lastSensor);
	ThrowCudaErrors(cudaDeviceSynchronize());
	m_lastSensor = g_SceneData.m_Camera;

	swapk(m_accumBuffer1, m_accumBuffer2);
}

}