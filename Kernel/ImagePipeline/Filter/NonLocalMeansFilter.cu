#include "NonLocalMeansFilter.h"
#include <Kernel/TraceHelper.h>
#include <Engine/DifferentialGeometry.h>
#include <Engine/Material.h>

namespace CudaTracerLib
{

#define CACHE_SIZE 32
#define BLOCK_SIZE 16
#define BLOCK_OFF ((CACHE_SIZE - BLOCK_SIZE) / 2)
#define NUM_PIXELS_COPY_PER_THREAD (CACHE_SIZE / BLOCK_SIZE)
CUDA_SHARED RGBE g_cachedImgData[CACHE_SIZE * CACHE_SIZE];
//CUDA_SHARED NonLocalMeansFilter::FeatureData g_cachedFeatureData[CACHE_SIZE * CACHE_SIZE];

CUDA_DEVICE void copyToShared(RGBE* deviceDataCached, NonLocalMeansFilter::FeatureData* deviceFeatureData, int w, int h)
{
	int block_start_x = blockDim.x * blockIdx.x - BLOCK_OFF, block_start_y = blockDim.y * blockIdx.y - BLOCK_OFF;
	int off_x = threadIdx.x, off_y = threadIdx.y;
	int n_per_t = NUM_PIXELS_COPY_PER_THREAD;
	for(int i = 0; i < n_per_t; i++)
		for (int j = 0; j < n_per_t; j++)
		{
			int l_x = off_x * n_per_t + i, l_y = off_y * n_per_t + j;
			int x = block_start_x + l_x, y = block_start_y + l_y;
			if (x >= 0 && x < w && y >= 0 && y < h)
			{
				g_cachedImgData[l_y * CACHE_SIZE + l_x] = deviceDataCached[y * w + x];
				//g_cachedFeatureData[l_y * CACHE_SIZE + l_x] = deviceFeatureData[y * w + x];
			}
		}
}

CUDA_DEVICE Spectrum loadFromShared(int x, int y)
{
	int block_start_x = blockDim.x * blockIdx.x - BLOCK_OFF, block_start_y = blockDim.y * blockIdx.y - BLOCK_OFF;
	int local_x = x - block_start_x, local_y = y - block_start_y;
	if (local_x < 0 || local_x >= CACHE_SIZE ||
		local_y < 0 || local_y >= CACHE_SIZE)
	{
		printf("Invalid access %d, %d\n", local_x, local_y);
		return 0.0f;
	}
	Spectrum col;
	col.fromRGBE(g_cachedImgData[local_y * CACHE_SIZE + local_x]);
	return col;
}

/*CUDA_DEVICE NonLocalMeansFilter::FeatureData loadFromSharedFeature(int x, int y)
{
	int block_start_x = blockDim.x * blockIdx.x - 8, block_start_y = blockDim.y * blockIdx.y - 8;
	int local_x = x - block_start_x, local_y = y - block_start_y;
	if (local_x < 0 || local_x >= CACHE_SIZE ||
		local_y < 0 || local_y >= CACHE_SIZE)
	{
		printf("Invalid access %d, %d\n", local_x, local_y);
		return NonLocalMeansFilter::FeatureData();
	}
	return g_cachedFeatureData[local_y * CACHE_SIZE + local_x];
}*/

CUDA_DEVICE float patchDistance(int p_x, int p_y, int q_x, int q_y, int F, int w, int h)
{
	float d_range = 0, weight = 0;
	for(int x = -F; x <= F; x++)
		for (int y = -F; y <= F; y++)
		{
			if (p_x + x < 0 || p_x + x >= w || p_y + y < 0 || p_y + y >= h ||
				q_x + x < 0 || q_x + x >= w || q_y + y < 0 || q_y + y >= h)
				continue;

			auto c_a = loadFromShared(p_x + x, p_y + y).saturate();
			auto c_b = loadFromShared(q_x + x, q_y + y).saturate();
			d_range += math::sqr(c_a - c_b).sum();
			weight += 3;
		}

	return weight != 0 ? d_range / weight : 0;
}

CUDA_DEVICE float weight(int p_x, int p_y, int q_x, int q_y, int w, int h, int F, float sigma2, float k)
{
	float d_range = patchDistance(p_x, p_y, q_x, q_y, F, w, h);
	return math::exp(-max(0.0f, d_range - 2 * sigma2) / (k * k * 2 * sigma2));
}

CUDA_GLOBAL void applyNonLinear(Image img, RGBE* deviceDataCached, NonLocalMeansFilter::FeatureData* deviceFeatueData, int R, int F, float k, float sigma2Scale, PixelVarianceBuffer varBuf)
{
	copyToShared(deviceDataCached, deviceFeatueData, img.getWidth(), img.getHeight());
	__syncthreads();

	int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y, w = img.getWidth(), h = img.getHeight();
	if (x < w && y < h)
	{
		float C_p = 0;//normalization weight
		Spectrum c_p_hat(0.0f);
		float sigma2 = varBuf(x,y).computeVariance() * sigma2Scale;
		for(int xo = -R; xo <= R; xo++)
			for (int yo = -R; yo <= R; yo++)
			{
				int q_x = x + xo, q_y = y + yo;
				if (q_x < 0 || q_x >= w || q_y < 0 || q_y >= h)
					continue;
				auto c_q = loadFromShared(q_x, q_y);
				float we = weight(x, y, q_x, q_y, w, h, F, sigma2, k);
				C_p += we;
				c_p_hat += we * c_q;
			}

		img.getFilteredData(x, y) = Spectrum(c_p_hat / C_p).toRGBE();
	}
}

CUDA_GLOBAL void copyToCached(Image img, RGBE* deviceDataCached, float splatScale)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y, w = img.getWidth(), h = img.getHeight();
	if (x < img.getWidth() && y < img.getHeight())
	{
		auto c_p = img.getPixelData(x, y).toSpectrum(splatScale);
		deviceDataCached[y * img.getWidth() + x] = c_p.toRGBE();
	}
}

CUDA_GLOBAL void initializeFeatureBuffer(NonLocalMeansFilter::FeatureData* deviceFeatureBuffer, int w, int h)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		auto fDat = NonLocalMeansFilter::FeatureData::Init();

		auto ray = g_SceneData.GenerateSensorRay(x, y);
		auto res = traceRay(ray);
		if (res.hasHit())
		{
			BSDFSamplingRecord bRec;
			res.getBsdfSample(ray, bRec, ETransportMode::ERadiance);

			fDat.m_depth = res.m_fDist;
			fDat.m_normal = NormalizedFloat3ToUchar2(bRec.dg.n);
			fDat.m_albedo = res.getMat().bsdf.f(bRec).toRGBCOL();
		}
		else fDat.m_depth = half(FLT_MAX);

		deviceFeatureBuffer[y * w + x] = fDat;
	}
}

void NonLocalMeansFilter::Apply(Image& img, int numPasses, float splatScale, const PixelVarianceBuffer& varBuffer)
{
	const int R = 5, F = 3;
	const float k = m_settings.getValue(KEY_k()), sigma2Scale = m_settings.getValue(KEY_sigma2Scale());

	if (BLOCK_OFF < R + F)
		throw std::runtime_error("Cache size too small for filtering window size!");

	int xResolution = img.getWidth(), yResolution = img.getHeight();

	if (numPasses == 0)
	{
		initializeFeatureBuffer << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(m_featureBuffer, xResolution, yResolution);
	}

	//copy the data to the cached version
	copyToCached << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(img, m_cachedImg, splatScale);
	ThrowCudaErrors(cudaThreadSynchronize());
	applyNonLinear << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(img, m_cachedImg, m_featureBuffer, R, F, k, sigma2Scale, varBuffer);
	ThrowCudaErrors(cudaThreadSynchronize());
}

}