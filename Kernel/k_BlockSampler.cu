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

CUDA_GLOBAL void copyPass(unsigned int w, unsigned int h, e_Image img, k_BlockSampler::pixelData* cudaPixels, float splat)
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
};

static std::vector<unsigned int> g_IndicesH;

void k_BlockSampler::AddPass(const e_Image& img)
{
	unsigned int w, h;
	img.getExtent(w, h);
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

	const int N = 30;
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
		cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * m_uNumBlocks, cudaMemcpyDeviceToHost);
		/*
		float* DAT = new float[m_uNumBlocks];
		cudaMemcpy(DAT, m_pDeviceBlockData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);
		unsigned int* DAT2 = new unsigned int[m_uNumBlocks];
		cudaMemcpy(DAT2, m_pDeviceIndexData, 4 * m_uNumBlocks, cudaMemcpyDeviceToHost);
		std::cout << "# to launch : " << m_uNumBlocksToLaunch << "\n";*/
	}
}

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
}