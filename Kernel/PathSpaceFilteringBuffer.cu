#include "PathSpaceFilteringBuffer.h"
#include "Tracer.h"
#include "TraceAlgorithms.h"

namespace CudaTracerLib {

void PathSpaceFilteringBuffer::PrepareForRendering(Image& I, DynamicScene* scene)
{
	auto eye_hit_points = TracerBase::GetEyeHitPointBox(scene, false);
	m_hitPointBuffer.SetGridDimensions(eye_hit_points);
	m_firstPass = true;
}

struct settingsData
{
	float globalRadScale;
	float pixelFootprintScale;
	float alpha;

	bool use_global;
	bool use_footprint;
	bool use_prevFrames;
};

template<bool USE_DEPTH_IMAGE> CUDA_FUNC_IN void computePixel(Image& I, DeviceDepthImage& depthImg, SpatialGridList_Linked<PathSpaceFilteringBuffer::path_entry>& path_buffer, unsigned int x, unsigned int y, Vec3f* accumBuffer_old, Vec3f* accumBuffer_new, Sensor lastSensor, float globalRad, settingsData& settings)
{
	float query_rad = settings.use_global ? settings.globalRadScale * globalRad : 0.0f;
	auto rng = g_SamplerData(y * I.getWidth() + x);
	NormalizedT<Ray> ray, rayX, rayY;
	auto W = g_SceneData.sampleSensorRay(ray, rayX, rayY, Vec2f((float)x, (float)y) + rng.randomFloat2(), rng.randomFloat2());
	auto res = traceRay(ray);
	int depth = 0;
	while (res.hasHit() && depth++ < 4)
	{
		BSDFSamplingRecord bRec;
		res.getBsdfSample(ray, bRec, ERadiance);

		if (depth == 1)
		{
			if (settings.use_footprint)
			{
				bRec.dg.computePartials(ray, rayX, rayY);
				Vec3f dp_dx, dp_dy;
				bRec.dg.compute_dp_ds(dp_dx, dp_dy);
				float avg_dist_next_pixel = fmaxf(dp_dx.length(), dp_dy.length());
				float footprint_rad = settings.pixelFootprintScale * avg_dist_next_pixel;
				query_rad = settings.use_global ? fmaxf(query_rad, footprint_rad) : footprint_rad;
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
			auto query_box = bRec.dg.ComputeOnSurfaceDiskBounds(query_rad);
			Spectrum L_query = 0.0f;
			int n_found = 0;
			path_buffer.ForAll<16>(query_box.minV, query_box.maxV, [&](const Vec3u& cell_idx, unsigned int idx, const PathSpaceFilteringBuffer::path_entry& ent)
			{
				if (distanceSquared(ent.p, bRec.dg.P) < query_rad * query_rad && dot(Uchar2ToNormalizedFloat3(ent.nor), bRec.dg.sys.n) > 0.75f)
				{
					bRec.wo = Uchar2ToNormalizedFloat3(ent.wi);
					auto f_r = res.getMat().bsdf.f(bRec);
					Spectrum L_i;
					L_i.fromRGBE(ent.Li);
					L_query += f_r * L_i;
					n_found++;
				}
			});
			if(n_found != 0)
				L_query /= (float)n_found;

			Spectrum L_indirect = L_query;
			if (settings.use_prevFrames)
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
						accum_new_rgb = oldAccum * (1.0f - settings.alpha) + query_rgb * settings.alpha;
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

template<bool USE_DEPTH_IMAGE> CUDA_GLOBAL void computePixelsKernel(Image I, DeviceDepthImage depthImg, SpatialGridList_Linked<PathSpaceFilteringBuffer::path_entry> path_buffer, Vec3f* accumBuffer_old, Vec3f* accumBuffer_new, Sensor prevSensor, float globalRad, settingsData settings)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < I.getWidth() && y < I.getHeight())
		computePixel<USE_DEPTH_IMAGE>(I, depthImg, path_buffer, x, y, accumBuffer_old, accumBuffer_new, prevSensor, globalRad, settings);
}

void PathSpaceFilteringBuffer::ComputePixelValues(Image& I, DynamicScene* scene, DeviceDepthImage* depthImage)
{
	settingsData settings;
	settings.globalRadScale = m_paraSettings.getValue(KEY_GlobalRadiusScale());
	settings.pixelFootprintScale = m_paraSettings.getValue(KEY_PixelFootprintScale());
	settings.alpha = m_paraSettings.getValue(KEY_PrevFrameAlpha());
	settings.use_global = m_paraSettings.getValue(KEY_UseRadius_GlobalScale());
	settings.use_footprint = m_paraSettings.getValue(KEY_UseRadius_PixelFootprintSize());
	settings.use_prevFrames = m_paraSettings.getValue(KEY_UsePreviousFrames());

	if (m_firstPass)
	{
		settings.alpha = 1.0f;
		m_lastSensor = g_SceneData.m_Camera;
	}

	UpdateKernel(scene);
	int p0 = 16;
	if(depthImage)
		computePixelsKernel<true> << <dim3(I.getWidth() / p0 + 1, I.getHeight() / p0 + 1, 1), dim3(p0, p0, 1) >> >(I, *depthImage, m_hitPointBuffer, m_accumBuffer1, m_accumBuffer2, m_lastSensor, m_pixelRad, settings);
	else computePixelsKernel<false> << <dim3(I.getWidth() / p0 + 1, I.getHeight() / p0 + 1, 1), dim3(p0, p0, 1) >> >(I, DeviceDepthImage(), m_hitPointBuffer, m_accumBuffer1, m_accumBuffer2, m_lastSensor, m_pixelRad, settings);
	ThrowCudaErrors(cudaDeviceSynchronize());
	m_lastSensor = g_SceneData.m_Camera;
	m_firstPass = false;
	swapk(m_accumBuffer1, m_accumBuffer2);
}

}