#include "k_BlockSampler.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <numeric>

CUDA_FUNC_IN float weight(k_SamplerpixelData& f)
{
	float var = f.E2 / f.n - (f.E / f.n) * (f.E / f.n);
	return var;
	//return math::sqrt(var) / (f.max - f.min);
}

void k_BlockSampleImage::Add(int x, int y, const Spectrum& c)
{
	img.AddSample(x, y, c);
	float l = math::clamp(c.getLuminance(), 0.0f, 1000.0f);
	unsigned int i = y * w + x;
	k_SamplerpixelData& pix = m_pLumData[i];
	pix.E += l;
	pix.E2 += l * l;
	pix.n++;
	/*if (l == 0)
		return;
	m_pLumData[i].max = max(m_pLumData[i].max, l);
	m_pLumData[i].min = min(m_pLumData[i].min, l);*/
}

CUDA_FUNC_IN float computeGradient(e_Image img, unsigned int x, unsigned int y)
{
	Spectrum divSum(0.0f);
	Spectrum p = img.getPixel(0, x, y).saturate();
	if (x)
		divSum += (p - img.getPixel(0, x - 1, y).saturate()).abs();
	if (x < img.getWidth() - 1)
		divSum += (p - img.getPixel(0, x + 1, y).saturate()).abs();
	if (y)
		divSum += (p - img.getPixel(0, x, y - 1).saturate()).abs();
	if (y < img.getHeight() - 1)
		divSum += (p - img.getPixel(0, x, y + 1).saturate()).abs();
	return divSum.average() / float(!!x + !!y + (x < img.getWidth() - 1) + (y < img.getHeight() - 1));
}

CUDA_GLOBAL void evalPass(e_Image img, k_SamplerpixelData* lumData, float* blockData, int nx, unsigned int* deviceNumSamples)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		k_SamplerpixelData& ent = lumData[y * img.getWidth() + x];
		float VAR = weight(ent);
		//float w_i = computeGradient(img, x, y);
		float w_i = fabsf(ent.max - VAR);

		unsigned int ix = x / blockSize, iy = y / blockSize;
		unsigned int x2 = (ix + 1) * blockSize, y2 = (iy + 1) * blockSize;
		unsigned int bw = min(img.getWidth(), x2) - ix * blockSize;
		unsigned int bh = min(img.getHeight(), y2) - iy * blockSize;
		int idx2 = iy * nx + ix;
		atomicAdd(blockData + idx2, w_i / float(bw * bh));
		if (x % blockSize == 0 && y % blockSize == 0)
			deviceNumSamples[idx2] = (unsigned int)img.accessPixel(x,y).weightSum;
	}
}

CUDA_GLOBAL void copyKernel(e_Image img, k_SamplerpixelData* lumData, unsigned int* deviceIndexData, int nx)
{
	Vec2i off((blockIdx.y % BLOCK_FACTOR) * 32, (blockIdx.y / BLOCK_FACTOR) * 32);
	unsigned int bIdx = deviceIndexData[blockIdx.x];
	unsigned int ix = bIdx % nx, iy = bIdx / nx;
	unsigned int x = ix * blockSize + threadIdx.x + off.x;
	unsigned int y = iy * blockSize + threadIdx.y + off.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		k_SamplerpixelData& ent = lumData[y * img.getWidth() + x];
		ent.max = weight(ent);
	}
}

struct countA
{
	float thresh;

	countA(float f)
		: thresh(f)
	{
	}

	CUDA_FUNC_IN bool operator()(float f) const
	{
		return f > thresh;
	}
};

struct order
{
	float* data;

	order(float* f)
		: data(f)
	{
	}

	CUDA_FUNC_IN bool operator()(unsigned int a, unsigned int b) const
	{
		return data[a] > data[b];
	}
};

void k_BlockSampler::AddPass()
{
	return;
	static bool initIndices = false;
	static std::vector<unsigned int> g_IndicesH, g_IndicesH2;
	if (!initIndices)
	{
		initIndices = true;
		g_IndicesH.resize(2048 * 2048);
		std::iota(g_IndicesH.begin(), g_IndicesH.end(), 0);
		g_IndicesH2.resize(2048 * 2048);
	}

	const int N = 25;
	m_uPassesDone++;
	if ((m_uPassesDone % N) == 0)
	{
		cudaMemset(m_pDeviceBlockData, 0, totalNumBlocks() * sizeof(float));
		int nx = (img->getWidth() + 32 - 1) / 32, ny = (img->getHeight() + 32 - 1) / 32;
		evalPass << <dim3(nx, ny), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceBlockData, nx, m_pDeviceSamplesData);
		cudaMemcpy(m_pHostBlockData, m_pDeviceBlockData, sizeof(float) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pHostSamplesData, m_pDeviceSamplesData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], totalNumBlocks() * sizeof(unsigned int), cudaMemcpyHostToDevice);
		thrust::sort(thrust::device_ptr<unsigned int>(m_pDeviceIndexData), thrust::device_ptr<unsigned int>(m_pDeviceIndexData + totalNumBlocks()), order(m_pDeviceBlockData));
		hasValidData = true;
		//cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		//m_uNumBlocksToLaunch = totalNumBlocks() / 2;
		float w_max = 0;
		for (int i = 0; i < totalNumBlocks(); i++)
			w_max = max(w_max, m_pHostBlockData[i]);
		m_uNumBlocksToLaunch = 0;
		cudaMemcpy(&g_IndicesH2[0], m_pDeviceIndexData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		for (int i = 0; i < totalNumBlocks(); i++)
		{
			float wd = m_pHostBlockData[g_IndicesH2[i]] / w_max;
			float rnd = float(rand()) / float(RAND_MAX);
			if (rnd < wd)
				m_pHostIndexData[m_uNumBlocksToLaunch++] = g_IndicesH2[i];
		}
		//copyKernel << <dim3(m_uNumBlocksToLaunch, BLOCK_FACTOR * BLOCK_FACTOR), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceIndexData, nx);
	}
}

void k_BlockSampler::Clear()
{
	m_uPassesDone = 0;
	hasValidData = false;
	cudaMemset(m_pLumData, 0, img->getWidth() * img->getHeight() * sizeof(k_SamplerpixelData));
	Platform::SetMemory(m_pHostSamplesData, totalNumBlocks() * sizeof(unsigned int));
}

struct Colorizer
{
	float avg;
	float stdDev;
	float min, max;

	float mappedAvg;
	float mappedDev;

	Colorizer(float avg, float var, float min, float max, float mappedAvg = 0.5f, float mappedDev = 0.35f)
		: avg(avg), stdDev(math::sqrt(var)), min(min), max(max), mappedAvg(mappedAvg), mappedDev(mappedDev)
	{

	}

	template<typename T> static Colorizer FromData(const T* data, unsigned int N)
	{
		float avg = 0, avg2 = 0, min = FLT_MAX, max = -FLT_MAX;
		for (int i = 0; i < N; i++)
		{
			float f = data[i];
			avg += f;
			avg2 += f * f;
			min = ::min(min, f);
			max = ::max(max, f);
		}
		avg /= N;
		avg2 /= N;
		return Colorizer(avg, avg2 - avg * avg, min, max);
	}

	template<typename T, typename Converter> static Colorizer FromData(const T* data, unsigned int N, const Converter& cnv)
	{
		float avg = 0, avg2 = 0, min = FLT_MAX, max = -FLT_MAX;
		for (int i = 0; i < N; i++)
		{
			float f = cnv(data[i]);
			avg += f;
			avg2 += f * f;
			min = ::min(min, f);
			max = ::max(max, f);
		}
		avg /= N;
		avg2 /= N;
		return Colorizer(avg, avg2 - avg * avg, min, max);
	}

	CUDA_FUNC_IN float operator()(float f)
	{
		return (f - min) / (max - min);

		if (abs(f - avg) < stdDev)
		{
			float l = (f - avg) / stdDev;
			return mappedAvg + mappedDev * l;
		}
		else
		{
			float s = math::sign(f - avg);
			float b = avg + stdDev * s;
			float m = mappedAvg + mappedDev * s;
			float l = abs(f - b);
			float range = 0.5f - mappedDev;
			return m + l * range * s;
		}
	}

	CUDA_FUNC_IN Spectrum colorize(float f)
	{
		return colorizeMapped(operator()(f));
	}

	CUDA_FUNC_IN Spectrum colorizeMapped(float t)
	{
		Spectrum qs;
		qs.fromHSL(1.0f / 3.0f - t / 3.0f, 1, 0.5f);
		return qs;
	}
};

CUDA_GLOBAL void drawPass(e_Image img, k_SamplerpixelData* lumData, float* blockData, Colorizer col, bool blocks, int nx)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		float VAR = 0;
		if (blocks)
			VAR = blockData[(y / blockSize) * nx + (x / blockSize)];
		//else VAR = weight(lumData[y * img.getWidth() + x]);
		else VAR = computeGradient(img, x, y);
		img.SetSample(x, y, Spectrum(col(VAR)).toRGBCOL());
	}
}

void k_BlockSampler::DrawVariance(bool blocks) const
{
	Colorizer col = Colorizer::FromData(m_pHostBlockData, totalNumBlocks());
	int nx = (img->getWidth() + blockSize - 1) / blockSize;
	drawPass << < dim3(img->getWidth() / 32 + 1, img->getHeight() / 32 + 1), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceBlockData, col, blocks, nx);
	img->disableUpdate();
}