#include "PathSpaceFilteringBuffer.h"
#include "Tracer.h"
#include "TraceAlgorithms.h"

namespace CudaTracerLib {

void PathSpaceFilteringBuffer::PrepareForRendering(Image& I, DynamicScene* scene)
{
	auto eye_hit_points = TracerBase::GetEyeHitPointBox(scene, false);
	m_hitPointBuffer.SetGridDimensions(eye_hit_points);

	m_pixelRad0 = eye_hit_points.Size().length() / max(I.getWidth(), I.getHeight()) * 0.25f;
	m_numIteration = 0;
}

CUDA_FUNC_IN void computePixel(Image& I, SpatialGridList_Linked<PathSpaceFilteringBuffer::path_entry>& path_buffer, float rad, unsigned int x, unsigned int y)
{
	auto rng = g_SamplerData(y * I.getWidth() + x);
	NormalizedT<Ray> ray, rayX, rayY;
	auto W = g_SceneData.sampleSensorRay(ray, rayX, rayY, Vec2f(x, y) + rng.randomFloat2(), rng.randomFloat2());
	auto res = traceRay(ray);
	int depth = 0;
	while (res.hasHit() && depth++ < 4)
	{
		BSDFSamplingRecord bRec;
		res.getBsdfSample(ray, bRec, ERadiance);

		if (depth == 1)
		{
			bRec.dg.computePartials(ray, rayX, rayY);
			Vec3f dp_dx, dp_dy;
			bRec.dg.compute_dp_ds(dp_dx, dp_dy);
			float avg_dist_next_pixel = fmaxf(dp_dx.length(), dp_dy.length());
			rad = fmaxf(rad, avg_dist_next_pixel);
		}

		if (res.getMat().bsdf.hasComponent(EDelta) || res.getMat().bsdf.hasComponent(EGlossy))
		{
			W *= res.getMat().bsdf.sample(bRec, rng.randomFloat2());
			ray = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
			res = traceRay(ray);
		}
		else
		{
			auto query_box = bRec.dg.ComputeOnSurfaceDiskBounds(rad);
			Spectrum L_o = 0.0f;
			int n_found = 0;
			path_buffer.ForAll(query_box.minV, query_box.maxV, [&](const Vec3u& cell_idx, unsigned int idx, const PathSpaceFilteringBuffer::path_entry& ent)
			{
				if (distanceSquared(ent.p, bRec.dg.P) < rad * rad && dot(Uchar2ToNormalizedFloat3(ent.nor), bRec.dg.sys.n) > 0.75f)
				{
					bRec.wo = Uchar2ToNormalizedFloat3(ent.wi);
					auto f_r = res.getMat().bsdf.f(bRec);
					Spectrum L_i;
					L_i.fromRGBE(ent.Li);
					L_o += f_r * L_i;
					n_found++;
				}
			});
			L_o /= (float)n_found;

			L_o += UniformSampleAllLights(bRec, res.getMat(), 1, rng);
			L_o = n_found / 10.0f;
			I.AddSample(x, y, W * L_o);
			break;
		}
	}
}

CUDA_GLOBAL void computePixelsKernel(Image I, SpatialGridList_Linked<PathSpaceFilteringBuffer::path_entry> path_buffer, float rad)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < I.getWidth() && y < I.getHeight())
		computePixel(I, path_buffer, rad, x, y);
}

void PathSpaceFilteringBuffer::ComputePixelValues(Image& I, DynamicScene* scene)
{
	auto rad_i = m_pixelRad0 * math::pow((float)m_numIteration, ((2.0f / 3.0f) - 1) / 2.0f);
	UpdateKernel(scene);
	int p0 = 16;
	computePixelsKernel << <dim3(I.getWidth() / p0 + 1, I.getHeight() / p0 + 1, 1), dim3(p0, p0, 1) >> >(I, m_hitPointBuffer, rad_i);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

}