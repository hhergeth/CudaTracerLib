#include "k_BlockSampler.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <numeric>
/*
CUDA_GLOBAL void addPassG(unsigned int w, unsigned int h, e_Image img, k_BlockSampler::pixelData* cudaPixels, float* blockData, float numSamplesDone, float splat)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	CUDA_SHARED float sumI;
	if(threadIdx.x == 0 && threadIdx.y == 0)
		sumI = 0.0f;
	__syncthreads();
	if(x < w && y < h)
	{
		float avg = img.getPixel(splat, x, y).average();
		cudaPixels[y * w + x].I += avg;
		cudaPixels[y * w + x].I2 += avg * avg;
		float var = variance(cudaPixels[y * w + x].I, cudaPixels[y * w + x].I2, numSamplesDone);
		atomicAdd(&sumI, var);
		__syncthreads();
		if(threadIdx.x == 0 && threadIdx.y == 0)
			blockData[blockIdx.y * gridDim.x + blockIdx.x] = sumI / float(blockDim.x * blockDim.y);
	}
}*/

/*CUDA_GLOBAL void copyPass(unsigned int w, unsigned int h, e_Image img, k_BlockSampler::pixelData* cudaPixels, float splat)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < w && y < h)
		cudaPixels[y * w + x].V = img.getPixel(splat, x, y).average();
}

CUDA_GLOBAL void evalPass(unsigned int w, unsigned int h, e_Image img, k_BlockSampler::pixelData* cudaPixels, float* blockData, float numSamplesDone, float splat)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	CUDA_SHARED float sumI;
	if(threadIdx.x == 0 && threadIdx.y == 0)
		sumI = 0.0f;
	if(x < w && y < h)
	{
		float v = img.getPixel(splat, x, y).average() - cudaPixels[y * w + x].V;
		atomicAdd(&sumI, abs(v));
		__syncthreads();
		if(threadIdx.x == 0 && threadIdx.y == 0)
			blockData[blockIdx.y * gridDim.x + blockIdx.x] = sumI / float(blockDim.x * blockDim.y);
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
};*/

//static std::vector<unsigned int> g_IndicesH;

//void k_BlockSampler::AddPass(const e_Image& img)
//{
	//unsigned int w, h;
	//img.getExtent(w, h);
	/*nSamplesDone++;
	addPassG<<<dim3(m_uBlockDimX,m_uBlockDimY,1), dim3(m_uBlockSize,m_uBlockSize,1) >>>(w,h,img,cudaPixels,m_pDeviceBlockData,nSamplesDone,1.0f);

	if(nSamplesDone == 29)
	{
		cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], m_uNumBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
		m_uNumBlocksToLaunch = m_uNumBlocks;
	}
	if(nSamplesDone != 30)
		return;
	cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], m_uNumBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//float sum = thrust::reduce(thrust::device_ptr<float>(m_pDeviceBlockData), thrust::device_ptr<float>(m_pDeviceBlockData + m_uNumBlocks));
	//float avg = sum / float(m_uNumBlocks);

	thrust::sort(thrust::device_ptr<unsigned int>(m_pDeviceIndexData), thrust::device_ptr<unsigned int>(m_pDeviceIndexData + m_uNumBlocks), order(m_pDeviceBlockData));
	cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * m_uNumBlocks, cudaMemcpyDeviceToHost);
	//countA c(avg * 1.0f);
	//m_uNumBlocksToLaunch = thrust::count_if(thrust::device_ptr<float>(m_pDeviceBlockData), thrust::device_ptr<float>(m_pDeviceBlockData + m_uNumBlocks), c);
	m_uNumBlocksToLaunch = m_uNumBlocks / 2;

	//if(m_uNumBlocksToLaunch == 0)
	//	m_uNumBlocksToLaunch = m_uNumBlocks;
	hasValidData = true;
	nSamplesDone = 0;
	cudaMemset(cudaPixels, 0, w * h * sizeof(pixelData));
	*/
	/*
	float* DAT = new float[m_uNumBlocks];
	cudaMemcpy(DAT, m_pDeviceBlockData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);
	unsigned int* DAT2 = new unsigned int[m_uNumBlocks];
	cudaMemcpy(DAT2, m_pDeviceIndexData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);*/
	//std::cout << "# to launch : " << m_uNumBlocksToLaunch << "\n";

	/*const int N = 30;
	nSamplesDone++;
	if(nSamplesDone == 1)
		copyPass<<<dim3(m_uBlockDimX,m_uBlockDimY,1), dim3(m_uBlockSize,m_uBlockSize,1) >>>(w, h, img, cudaPixels, 1);
	if((nSamplesDone % N) == 0)
	{
		evalPass<<<dim3(m_uBlockDimX,m_uBlockDimY,1), dim3(m_uBlockSize,m_uBlockSize,1) >>>(w,h,img,cudaPixels,m_pDeviceBlockData,nSamplesDone,1.0f);
		cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], m_uNumBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
		thrust::sort(thrust::device_ptr<unsigned int>(m_pDeviceIndexData), thrust::device_ptr<unsigned int>(m_pDeviceIndexData + m_uNumBlocks), order(m_pDeviceBlockData));
		m_uNumBlocksToLaunch = m_uNumBlocks / 2;
		hasValidData = true;
		copyPass<<<dim3(m_uBlockDimX,m_uBlockDimY,1), dim3(m_uBlockSize,m_uBlockSize,1) >>>(w, h, img, cudaPixels, 1);
		cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * m_uNumBlocks, cudaMemcpyDeviceToHost);*/
		/*
		float* DAT = new float[m_uNumBlocks];
		cudaMemcpy(DAT, m_pDeviceBlockData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);
		unsigned int* DAT2 = new unsigned int[m_uNumBlocks];
		cudaMemcpy(DAT2, m_pDeviceIndexData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);
		std::cout << "# to launch : " << m_uNumBlocksToLaunch << "\n";*/
	//}
//}
/*
void k_BlockSampler::Initialize(unsigned int w, unsigned int h)
{
	m_uTargetHeight = h;
	hasValidData = false;
	m_uBlockDimX = int(ceilf(float(w) / m_uBlockSize));
	m_uBlockDimY = int(ceilf(float(h) / m_uBlockSize));
	int N = m_uBlockDimX * m_uBlockDimY;
	if(N != m_uNumBlocks)
	{
		if(m_pDeviceBlockData)
		{
			cudaFree(m_pDeviceBlockData);
			cudaFree(m_pDeviceIndexData);
			cudaFree(cudaPixels);
			delete [] m_pHostIndexData;
		}
		CUDA_MALLOC(&m_pDeviceBlockData, N * sizeof(float));
		CUDA_MALLOC(&m_pDeviceIndexData, N * sizeof(unsigned int));
		CUDA_MALLOC(&cudaPixels, w * h * sizeof(pixelData));
		m_pHostIndexData = new unsigned int[N];
	}
	m_uNumBlocksToLaunch = m_uNumBlocks = N;
	nSamplesDone = 0;
	cudaMemset(cudaPixels, 0, w * h * sizeof(pixelData));

	if(g_IndicesH.size() == 0)
	{
		g_IndicesH.resize(2048 * 2048);
		std::iota(g_IndicesH.begin(), g_IndicesH.end(), 0);
	}
}*/

CUDA_FUNC_IN float weight(k_SamplerPixelData& f)
{
	float var = f.E2 / f.n - (f.E / f.n) * (f.E / f.n);
	return var;
	//return sqrtf(var) / (f.max - f.min);
}

void k_BlockSampleImage::Add(int x, int y, const Spectrum& c)
{
	img.AddSample(x, y, c);
	float l = clamp(c.getLuminance(), 0.0f, 1000.0f);
	unsigned int i = y * w + x;
	m_pLumData[i].E += l;
	m_pLumData[i].E2 += l * l;
	m_pLumData[i].n++;
	if (l == 0)
		return;
	m_pLumData[i].max = MAX(m_pLumData[i].max, l);
	m_pLumData[i].min = MIN(m_pLumData[i].min, l);
}

CUDA_FUNC_IN Spectrum clamp01(const Spectrum& c)
{
	Spectrum f;
	for (int i = 0; i < SPECTRUM_SAMPLES; i++)
		f[i] = clamp01(c[i]);
	return f;
}

CUDA_FUNC_IN float computeGradient(e_Image img, unsigned int x, unsigned int y)
{
	Spectrum divSum(0.0f);
	Spectrum p = clamp01(img.getPixel(0, x, y));
	if (x)
		divSum += (p - clamp01(img.getPixel(0, x - 1, y))).abs();
	if (x < img.getWidth() - 1)
		divSum += (p - clamp01(img.getPixel(0, x + 1, y))).abs();
	if (y)
		divSum += (p - clamp01(img.getPixel(0, x, y - 1))).abs();
	if (y < img.getHeight() - 1)
		divSum += (p - clamp01(img.getPixel(0, x, y + 1))).abs();
	return divSum.average() / float(!!x + !!y + (x < img.getWidth() - 1) + (y < img.getHeight() - 1));
}

CUDA_GLOBAL void evalPass(e_Image img, k_SamplerPixelData* lumData, float* blockData, int nx, unsigned int* deviceNumSamples)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		//float VAR = weight(lumData[y * img.getWidth() + x]);
		float VAR = computeGradient(img, x, y);

		unsigned int ix = x / blockSize, iy = y / blockSize;
		unsigned int x2 = (ix + 1) * blockSize, y2 = (iy + 1) * blockSize;
		unsigned int bw = MIN(img.getWidth(), x2) - ix * blockSize;
		unsigned int bh = MIN(img.getHeight(), y2) - iy * blockSize;
		int idx2 = iy * nx + ix;
		atomicAdd(blockData + idx2, VAR / float(bw * bh));
		if (x % blockSize == 0 && y % blockSize == 0)
			deviceNumSamples[idx2] = (unsigned int)img.accessPixel(x,y).weightSum;
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
	static bool initIndices = false;
	static std::vector<unsigned int> g_IndicesH;
	if (!initIndices)
	{
		initIndices = true;
		if (g_IndicesH.size() == 0)
		{
			g_IndicesH.resize(2048 * 2048);
			std::iota(g_IndicesH.begin(), g_IndicesH.end(), 0);
		}
	}

	const int N = 25;
	m_uPassesDone++;
	if ((m_uPassesDone % N) == 0)
	{
		cudaMemset(m_pDeviceBlockData, 0, totalNumBlocks() * sizeof(float));
		int nx = (img->getWidth() + blockSize - 1) / blockSize, ny = (img->getHeight() + blockSize - 1) / blockSize;
		evalPass << <dim3(nx * 2, ny * 2), dim3(blockSize / 2, blockSize / 2) >> >(*img, m_pLumData, m_pDeviceBlockData, nx, m_pDeviceSamplesData);
		cudaMemcpy(m_pHostBlockData, m_pDeviceBlockData, sizeof(float) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pHostSamplesData, m_pDeviceSamplesData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], totalNumBlocks() * sizeof(unsigned int), cudaMemcpyHostToDevice);
		thrust::sort(thrust::device_ptr<unsigned int>(m_pDeviceIndexData), thrust::device_ptr<unsigned int>(m_pDeviceIndexData + totalNumBlocks()), order(m_pDeviceBlockData));
		m_uNumBlocksToLaunch = totalNumBlocks() / 2;
		hasValidData = true;
		cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
	}
}

void k_BlockSampler::Clear()
{
	m_uPassesDone = 0;
	hasValidData = false;
	cudaMemset(m_pLumData, 0, img->getWidth() * img->getHeight() * sizeof(k_SamplerPixelData));
}

struct Colorizer
{
	float avg;
	float stdDev;
	float min, max;

	float mappedAvg;
	float mappedDev;

	Colorizer(float avg, float var, float min, float max, float mappedAvg = 0.5f, float mappedDev = 0.35f)
		: avg(avg), stdDev(sqrtf(var)), min(min), max(max), mappedAvg(mappedAvg), mappedDev(mappedDev)
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
			min = MIN(min, f);
			max = MAX(max, f);
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
			min = MIN(min, f);
			max = MAX(max, f);
		}
		avg /= N;
		avg2 /= N;
		return Colorizer(avg, avg2 - avg * avg, min, max);
	}

	CUDA_FUNC_IN float operator()(float f)
	{
		return (f - min) / (max - min);

		if (fabsf(f - avg) < stdDev)
		{
			float l = (f - avg) / stdDev;
			return mappedAvg + mappedDev * l;
		}
		else
		{
			float s = signf(f - avg);
			float b = avg + stdDev * s;
			float m = mappedAvg + mappedDev * s;
			float l = fabsf(f - b);
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

CUDA_GLOBAL void drawPass(e_Image img, k_SamplerPixelData* lumData, float* blockData, Colorizer col, bool blocks, int nx)
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