#include "NonLocalMeansFilter.h"
#include <Kernel/TraceHelper.h>
#include <Engine/DifferentialGeometry.h>
#include <Engine/Material.h>

namespace CudaTracerLib
{

#define CACHE_SIZE 64
#define BLOCK_SIZE 16
#define BLOCK_OFF ((CACHE_SIZE - BLOCK_SIZE) / 2)
#define NUM_PIXELS_COPY_PER_THREAD (CACHE_SIZE / BLOCK_SIZE)
CUDA_SHARED RGBE g_cachedImgData[CACHE_SIZE * CACHE_SIZE];
CUDA_SHARED half g_cachedVarData[CACHE_SIZE * CACHE_SIZE];
//CUDA_SHARED NonLocalMeansFilter::FeatureData g_cachedFeatureData[CACHE_SIZE * CACHE_SIZE];

template<bool USE_FEATURE_BUF, bool USE_VAR_BUF> CUDA_DEVICE void copyToShared(RGBE* deviceDataCached, NonLocalMeansFilter::FeatureData* deviceFeatureData, PixelVarianceBuffer* varBuf, int w, int h, int x_off, int y_off)
{
	int block_start_x = blockDim.x * blockIdx.x - BLOCK_OFF + x_off, block_start_y = blockDim.y * blockIdx.y - BLOCK_OFF + y_off;
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
				if (USE_VAR_BUF)
					g_cachedVarData[l_y * CACHE_SIZE + l_x] = varBuf->operator()(x, y).computeVariance();
				//if(USE_FEATURE_BUF)
					//g_cachedFeatureData[l_y * CACHE_SIZE + l_x] = deviceFeatureData[y * w + x];
			}
		}
}

CUDA_DEVICE Spectrum loadFromShared(int x, int y, int x_off, int y_off, float* var = 0)
{
	int block_start_x = blockDim.x * blockIdx.x - BLOCK_OFF + x_off, block_start_y = blockDim.y * blockIdx.y - BLOCK_OFF + y_off;
	int local_x = x - block_start_x, local_y = y - block_start_y;
	if (local_x < 0 || local_x >= CACHE_SIZE ||
		local_y < 0 || local_y >= CACHE_SIZE)
	{
		printf("Invalid access %d, %d\n", local_x, local_y);
		return 0.0f;
	}
	Spectrum col;
	col.fromRGBE(g_cachedImgData[local_y * CACHE_SIZE + local_x]);
	if (var)
		*var = g_cachedVarData[local_y * CACHE_SIZE + local_x].ToFloat();
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

CUDA_DEVICE float patchDistance(int p_x, int p_y, int q_x, int q_y, int F, int w, int h, int x_off, int y_off, float k, float sigma2Scale)
{
	const float eps = 1e-10f;
	const float alpha = 1.0f;
	float d_range = 0, weight = 0;;
	for(int x = -F; x <= F; x++)
		for (int y = -F; y <= F; y++)
		{
			if (p_x + x < 0 || p_x + x >= w || p_y + y < 0 || p_y + y >= h ||
				q_x + x < 0 || q_x + x >= w || q_y + y < 0 || q_y + y >= h)
				continue;

			float var_p, var_q;
			auto c_p = loadFromShared(p_x + x, p_y + y, x_off, y_off, &var_p);//.saturate();
			auto c_q = loadFromShared(q_x + x, q_y + y, x_off, y_off, &var_q);//.saturate();
			var_p *= sigma2Scale; var_q *= sigma2Scale;
			float u_diff = math::sqr(c_p - c_q).avg();
			float d = (u_diff - alpha * (var_p + min(var_p, var_q))) / (eps + k * k * (var_p + var_q));
			d_range += d;
			weight++;
		}

	return weight != 0 ? d_range / weight : 0;
}

CUDA_DEVICE float weight(int p_x, int p_y, int q_x, int q_y, int w, int h, int x_off, int y_off, int F, float k, float sigma2Scale)
{
	const float d_range = patchDistance(p_x, p_y, q_x, q_y, F, w, h, x_off, y_off, k, sigma2Scale);

	const float weight = math::exp(-max(0.0f, d_range));
	return weight < 0.05f ? 0.0f : weight;
}

CUDA_GLOBAL void computeWeights(Image img, RGBE* deviceDataCached, NonLocalMeansFilter::FeatureData* deviceFeatueData, int R, int F, float k, float sigma2Scale, PixelVarianceBuffer varBuf, NonLocalMeansFilter::FilterWeightBuffer weightBuffer, int x_off, int y_off)
{
	copyToShared<true, true>(deviceDataCached, deviceFeatueData, &varBuf, img.getWidth(), img.getHeight(), x_off, y_off);
	__syncthreads();

	int x = threadIdx.x + blockDim.x * blockIdx.x + x_off, y = threadIdx.y + blockDim.y * blockIdx.y + y_off, w = img.getWidth(), h = img.getHeight();
	if (x < w && y < h)
	{
		auto buffer = weightBuffer(x, y);
		for(int xo = -R; xo <= R; xo++)
			for (int yo = -R; yo <= R; yo++)
			{
				int q_x = x + xo, q_y = y + yo;
				if (q_x < 0 || q_x >= w || q_y < 0 || q_y >= h)
					continue;
				float we = weight(x, y, q_x, q_y, w, h, x_off, y_off, F, k, sigma2Scale);
				buffer(xo, yo) = we;
			}
	}
}

CUDA_GLOBAL void applyWeights(Image img, RGBE* deviceDataCached, NonLocalMeansFilter::FilterWeightBuffer weightBuffer, int R, int F)
{
	copyToShared<false, false>(deviceDataCached, 0, 0, img.getWidth(), img.getHeight(), 0, 0);
	__syncthreads();

	int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y, w = img.getWidth(), h = img.getHeight();
	if (x < w && y < h)
	{
		auto buffer = weightBuffer(x, y);
		Spectrum c_p_hat(0.0f);
		float C_p = 0;//normalization weight
		for (int xo = -R; xo <= R; xo++)
			for (int yo = -R; yo <= R; yo++)
			{
				int q_x = x + xo, q_y = y + yo;
				if (q_x < 0 || q_x >= w || q_y < 0 || q_y >= h)
					continue;
				float we = buffer(xo, yo);
				if (math::IsNaN(we))
					continue;
				auto c_q = loadFromShared(q_x, q_y, 0, 0);
				C_p += we;
				c_p_hat += we * c_q;
			}
		img.getFilteredData(x, y) = Spectrum(C_p > 1e-4f ? c_p_hat / C_p : loadFromShared(x, y, 0, 0)).toRGBE();
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
	const int R = 6, F = 3;
	const float k = m_settings.getValue(KEY_k()), sigma2Scale = m_settings.getValue(KEY_sigma2Scale());
	const int n_update = m_settings.getValue(KEY_UpdateWeightPeriodicity());

	if (BLOCK_OFF < R + F)
		throw std::runtime_error("Cache size too small for filtering window size!");

	int xResolution = img.getWidth(), yResolution = img.getHeight();

	bool force_update = false;
	if (!m_weightBuffer.canUseBuffer(R, F))
	{
		m_weightBuffer.adaptBuffer(R, F, xResolution, yResolution);
		force_update = true;
	}

	if (last_iter_weight_update + 1 != numPasses)
	{
		initializeFeatureBuffer << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (m_featureBuffer, xResolution, yResolution);
	}

	//copy the data to the cached version
	copyToCached << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(img, m_cachedImg, splatScale);
	ThrowCudaErrors(cudaThreadSynchronize());

	if (last_iter_weight_update + 1 != numPasses || (numPasses % n_update) == 0 || force_update)
	{
		m_weightBuffer.ClearBuffer();
		cudaFuncSetCacheConfig(computeWeights, cudaFuncCachePreferShared);
		const int block_width = 200;
		const int n_blocks_x = (xResolution + block_width - 1) / block_width, n_blocks_y = (yResolution + block_width - 1) / block_width;
		const int n_cuda_blocks = (block_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
		for(int i = 0; i < n_blocks_x; i++)
			for(int j = 0; j < n_blocks_y; j++)
				computeWeights << <dim3(n_cuda_blocks, n_cuda_blocks), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(img, m_cachedImg, m_featureBuffer, R, F, k, sigma2Scale, varBuffer, m_weightBuffer, i * block_width, j * block_width);
		ThrowCudaErrors(cudaThreadSynchronize());
	}

	cudaFuncSetCacheConfig(applyWeights, cudaFuncCachePreferShared);
	applyWeights << <dim3(xResolution / BLOCK_SIZE + 1, yResolution / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(img, m_cachedImg, m_weightBuffer, R, F);
	ThrowCudaErrors(cudaThreadSynchronize());

	last_iter_weight_update = numPasses;
}

}