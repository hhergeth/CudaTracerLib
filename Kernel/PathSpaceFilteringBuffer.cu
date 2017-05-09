#include "PathSpaceFilteringBuffer.h"
#include "Tracer.h"
#include "TraceAlgorithms.h"
#include <Engine/DynamicScene.h>

namespace CudaTracerLib {

void PathSpaceFilteringBuffer::PrepareForRendering(Image& I, DynamicScene* scene)
{
	auto eye_hit_points = scene->getSceneBox();// TracerBase::GetEyeHitPointBox(scene, false);
	m_hitPointBuffer.SetGridDimensions(eye_hit_points);

	m_settings.alpha = 1.0f;
	m_lastSensor = g_SceneData.m_Camera;
}

template<bool USE_DEPTH_IMAGE> CUDA_FUNC_IN void computePixel(Image& I, DeviceDepthImage& depthImg, PathSpaceFilteringBuffer& path_buffer, unsigned int x, unsigned int y, Vec3f* accumBuffer_old, Vec3f* accumBuffer_new, Sensor lastSensor)
{
	auto rng = g_SamplerData(y * I.getWidth() + x);
	NormalizedT<Ray> ray, rayX, rayY;
	auto W = g_SceneData.sampleSensorRay(ray, rayX, rayY, Vec2f((float)x, (float)y) + rng.randomFloat2(), rng.randomFloat2());
	auto res = traceRay(ray);
	int depth = 0;
	float query_rad = path_buffer.m_settings.globalRadScale * path_buffer.m_pixelRad;
	while (res.hasHit() && depth++ < 4)
	{
		BSDFSamplingRecord bRec;
		res.getBsdfSample(ray, bRec, ERadiance);

		if (depth == 1)
		{
			if (path_buffer.m_settings.use_footprint)
			{
				bRec.dg.computePartials(ray, rayX, rayY);
				query_rad = path_buffer.computeRad(bRec.dg);
			}

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
			path_buffer.m_hitPointBuffer.ForAll<64>(bRec.dg.P, [&]( unsigned int idx, const PathSpaceFilteringBuffer::path_entry& ent)
			{
				if (distanceSquared(ent.p, bRec.dg.P) < query_rad * query_rad && dot(Uchar2ToNormalizedFloat3(ent.nor), bRec.dg.sys.n) > 0.75f)
				{
					bRec.wo = Uchar2ToNormalizedFloat3(ent.wi);
					Spectrum L_i;
					L_i.fromRGBE(ent.Li);
					L_query += L_i * Frame::cosTheta(bRec.wo);
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
			if (path_buffer.m_settings.use_prevFrames)
			{
				Vec3f& accum_new_rgb = accumBuffer_new[y * I.getWidth() + x];
				Vec3f query_rgb;
				L_query.toLinearRGB(query_rgb.x, query_rgb.y, query_rgb.z);

				DirectSamplingRecord dRec(bRec.dg.P, NormalizedT<Vec3f>(0.0f));
				lastSensor.sampleDirect(dRec, Vec2f(0, 0));
				if (dRec.pdf)
				{
					Vec3f oldAccum = accumBuffer_old[(int)dRec.uv.y * I.getWidth() + (int)dRec.uv.x];
					if (n_found > 0)
						accum_new_rgb = oldAccum * (1.0f - path_buffer.m_settings.alpha) + query_rgb * path_buffer.m_settings.alpha;
					else accum_new_rgb = query_rgb;
					L_indirect.fromLinearRGB(accum_new_rgb.x, accum_new_rgb.y, accum_new_rgb.z);
				}
				else
				{
					accum_new_rgb = query_rgb;
				}
			}

			Spectrum L_direct = UniformSampleOneLight(bRec, res.getMat(), rng);

			I.AddSample(x, y, W * (L_direct + L_indirect));
			break;
		}
	}
}

template<bool USE_DEPTH_IMAGE> CUDA_GLOBAL void computePixelsKernel(Image I, DeviceDepthImage depthImg, PathSpaceFilteringBuffer path_buffer, Vec3f* accumBuffer_old, Vec3f* accumBuffer_new, Sensor prevSensor)
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
	m_settings.alpha = m_paraSettings.getValue(KEY_PrevFrameAlpha());

	swapk(m_accumBuffer1, m_accumBuffer2);
}

}