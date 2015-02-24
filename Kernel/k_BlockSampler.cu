#include "k_BlockSampler.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <numeric>

#define MAX_R 16
#define MAX_F 8
#define nlBlockSize 16
#define CACHE_LINE_WIDTH ( nlBlockSize)
CUDA_SHARED Spectrum s_PixelData[CACHE_LINE_WIDTH * CACHE_LINE_WIDTH];
CUDA_SHARED Vec2i s_BlockStart;
CUDA_ONLY_FUNC Spectrum loadPixel(int x, int y, e_Image& img)
{
	int lx = x - s_BlockStart.x, ly = y - s_BlockStart.y;
	if (lx < 0 || ly < 0)
		printf("Invalid pixel cache access at %i, %i", lx, ly);
	//if (lx < 0 || ly < 0 || lx >= CACHE_LINE_WIDTH || ly >= CACHE_LINE_WIDTH)
	//	return img.getPixel(x, y);
	return s_PixelData[CACHE_LINE_WIDTH * ly + lx];
}
CUDA_ONLY_FUNC float d2(int x1, int y1, int x2, int y2, e_Image& img, k_SamplerpixelData* lumData)
{
	k_SamplerpixelData& e1 = lumData[y1 * img.getWidth() + x1], &e2 = lumData[y2 * img.getWidth() + x2];
	float var_p = e1.E2 / e1.n - e1.E / e1.n * e1.E / e1.n, var_q = e2.E2 / e2.n - e2.E / e2.n * e2.E / e2.n;
	float var_pq = min(var_p, var_q);

	Spectrum u_i = loadPixel(x1, y1, img);
	Spectrum u_j = loadPixel(x2, y2, img);

	const float alpha = 1;
	const float eps = 10e-5f;
	const float k = 0.45f;

	Spectrum w_p_q_s = ((u_i - u_j) * (u_i - u_j) - alpha * (var_p + var_pq)) / (eps + k * k * (var_p + var_q));
	float w_p_q = w_p_q_s.average();
	return w_p_q < 0.05f ? 0 : w_p_q;
}
CUDA_ONLY_FUNC float D2(int x1, int y1, int x2, int y2, int f, e_Image& img, k_SamplerpixelData* lumData)
{
	float sum = 0.0f;
	int w = img.getWidth(), h = img.getHeight();
	for (int i = -f; i <= f; i++)
		for (int j = -f; j <= f; j++)
		{
			int px = math::clamp(x1 + i, 0, w - 1), py = math::clamp(y1 + j, 0, h - 1);
			int qx = math::clamp(x2 + i, 0, w - 1), qy = math::clamp(y2 + j, 0, h - 1);
			sum += d2(px, py, qx, qy, img, lumData);
		}
	return sum / math::sqr(2.0f * f + 1);
}
CUDA_ONLY_FUNC float w(int x1, int y1, int x2, int y2, int f, e_Image& img, k_SamplerpixelData* lumData)
{
	float d2 = D2(x1, y1, x2, y2, f, img, lumData);
	return math::exp(-max(0.0f, d2));
}
CUDA_GLOBAL void nlmeans(int f, int r, e_Image I, k_SamplerpixelData* lumData)
{
	s_BlockStart = Vec2i(blockIdx.x * nlBlockSize - f - r, blockIdx.y * nlBlockSize - f - r);

	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= I.getWidth() || y >= I.getHeight())
		return;
	Vec2i l = Vec2i(x, y) - s_BlockStart;
	s_PixelData[l.y * CACHE_LINE_WIDTH + l.x] = I.getPixel(x, y);
	__syncthreads();

	float C_p = 0;
	Spectrum u_p = 0.0f;
	for (int i = max(0, x - r); i <= min((int)I.getWidth() - 1, x + r); i++)
		for (int j = max(0, y - r); j <= min((int)I.getHeight() - 1, y + r); j++)
		{
			float we = w(x, y, i, j, f, I, lumData);
			C_p += we;
			u_p += we * I.getPixel(i, j);
		}
	u_p *= 1.0f / C_p;
	Spectrum srgb;
	u_p.toSRGB(srgb[0], srgb[1], srgb[2]);

	I.SetSample(x, y, srgb.toRGBCOL());
}

CUDA_FUNC_IN float weight(k_SamplerpixelData& f, bool update)
{
	float VAR = f.E2 / f.n - (f.E / f.n) * (f.E / f.n);
	float w_i = math::abs(f.lastVar - VAR);
	if (f.flag & 1)
	{
		if (update)
		{
			f.lastVar = VAR;
			f.flag = 0;
		}
	}
	else
	{
		if (update)
			f.flag = f.flag ? f.flag << 1 : 2;
		w_i = VAR * f.flag;
	}
	return w_i;
}

void k_BlockSampleImage::Add(int x, int y, const Spectrum& c)
{
	img.AddSample(x, y, c);
#ifdef ISCUDA
	float l = math::clamp(c.average(), 0.0f, 1000.0f);
	unsigned int i = y * w + x;
	k_SamplerpixelData& pix = m_pLumData[i];
	pix.E += l;
	pix.E2 += l * l;
	pix.flag = 1;
	pix.n++;
#endif
}

CUDA_GLOBAL void evalPass(e_Image img, k_SamplerpixelData* lumData, float* blockData, int nx, unsigned int* deviceNumSamples, unsigned int* contribPixels, float* deviceWeight)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;

	unsigned int ix = x / blockSize, iy = y / blockSize;
	unsigned int x2 = (ix + 1) * blockSize, y2 = (iy + 1) * blockSize;
	unsigned int bw = min(img.getWidth(), x2) - ix * blockSize;
	unsigned int bh = min(img.getHeight(), y2) - iy * blockSize;
	int idx2 = iy * nx + ix;

	if (x < img.getWidth() && y < img.getHeight())
	{
		k_SamplerpixelData& ent = lumData[y * img.getWidth() + x];
		if (x % blockSize == 0 && y % blockSize == 0)
			deviceNumSamples[idx2] = (unsigned int)ent.n;
		float w_i = weight(ent, true);
		
		if (w_i > 1e-5f)
			atomicInc(contribPixels + idx2, 0xffffffff);
		atomicAdd(blockData + idx2, deviceWeight[idx2] * w_i);
	}
}

struct order
{
	float* data;
	unsigned int* contribPixels;

	order(float* f, unsigned int* c)
		: data(f), contribPixels(c)
	{
	}

	CUDA_FUNC_IN bool operator()(unsigned int a, unsigned int b) const
	{
		return data[a] / float(contribPixels[a]) > data[b] / float(contribPixels[b]);
	}
};

void k_BlockSampler::AddPass()
{
	static bool initIndices = false;
	static std::vector<unsigned int> g_IndicesH, g_IndicesH2;
	if (!initIndices)
	{
		initIndices = true;
		g_IndicesH.resize(2048 * 2048);
		std::iota(g_IndicesH.begin(), g_IndicesH.end(), 0);
		g_IndicesH2.resize(2048 * 2048);
	}

	const int N = 5;
	m_uPassesDone++;
	if (0&&(m_uPassesDone % N) == 0)
	{
		cudaMemcpy(m_pDeviceWeight, m_pHostWeight, sizeof(float) * totalNumBlocks(), cudaMemcpyHostToDevice);
		cudaMemset(m_pDeviceContribPixels, 0, sizeof(unsigned int) * totalNumBlocks());
		cudaMemset(m_pDeviceBlockData, 0, totalNumBlocks() * sizeof(float));
		int nx = (img->getWidth() + 32 - 1) / 32, ny = (img->getHeight() + 32 - 1) / 32;
		evalPass << <dim3(nx, ny), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceBlockData, numBlocksRow(), m_pDeviceSamplesData, m_pDeviceContribPixels, m_pDeviceWeight);
		cudaMemcpy(m_pHostBlockData, m_pDeviceBlockData, sizeof(float) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pHostSamplesData, m_pDeviceSamplesData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pDeviceIndexData, &g_IndicesH[0], totalNumBlocks() * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(m_pHostContribPixels, m_pDeviceContribPixels, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		thrust::sort(thrust::device_ptr<unsigned int>(m_pDeviceIndexData), thrust::device_ptr<unsigned int>(m_pDeviceIndexData + totalNumBlocks()), order(m_pDeviceBlockData, m_pDeviceContribPixels));
		hasValidData = true;
		cudaMemcpy(m_pHostIndexData, m_pDeviceIndexData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		//m_uNumBlocksToLaunch = totalNumBlocks() / 2;
		/*float w_max = 0;
		for (unsigned int i = 0; i < totalNumBlocks(); i++)
			w_max = max(w_max, m_pHostBlockData[i]);
		m_uNumBlocksToLaunch = 0;
		cudaMemcpy(&g_IndicesH2[0], m_pDeviceIndexData, sizeof(unsigned int) * totalNumBlocks(), cudaMemcpyDeviceToHost);
		for (unsigned int i = 0; i < totalNumBlocks(); i++)
		{
			float wd = m_pHostBlockData[g_IndicesH2[i]] / w_max;
			float rnd = float(rand()) / float(RAND_MAX);
			if (rnd < wd)
				m_pHostIndexData[m_uNumBlocksToLaunch++] = g_IndicesH2[i];
		}*/
		//copyKernel << <dim3(m_uNumBlocksToLaunch, BLOCK_FACTOR * BLOCK_FACTOR), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceIndexData, nx);
		for (unsigned int i = 0; i < totalNumBlocks(); i++)
			m_pHostBlockData[i] /= float(m_pHostContribPixels[i]);
		m_uNumBlocksToLaunch = 0;
		while (m_uNumBlocksToLaunch < totalNumBlocks() && m_pHostBlockData[m_pHostIndexData[m_uNumBlocksToLaunch]] > 1e-5f)
			m_uNumBlocksToLaunch++;
		if (m_uNumBlocksToLaunch > totalNumBlocks() / 4)
			m_uNumBlocksToLaunch /= 2;
	}

	//img->disableUpdate();
	int f = 3, r = 10;
	//nlmeans << <dim3(img->getWidth() / nlBlockSize + 1, img->getHeight() / nlBlockSize + 1), dim3(nlBlockSize + f + r, nlBlockSize + f + r) >> > (f, r, *img, m_pLumData);
}

void k_BlockSampler::Clear()
{
	m_uPassesDone = 0;
	hasValidData = false;
	cudaMemset(m_pDeviceBlockData, 0, sizeof(float) * totalNumBlocks());
	cudaMemset(m_pLumData, 0, img->getWidth() * img->getHeight() * sizeof(k_SamplerpixelData));
	cudaMemset(m_pDeviceContribPixels, 0, sizeof(unsigned int) * totalNumBlocks());
	Platform::SetMemory(m_pHostSamplesData, totalNumBlocks() * sizeof(unsigned int));
	Platform::SetMemory(m_pHostBlockData, totalNumBlocks() * sizeof(float));
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
		for (unsigned int i = 0; i < N; i++)
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
		for (unsigned int i = 0; i < N; i++)
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
		/*if (math::abs(f - avg) < stdDev)
		{
			float l = (f - avg) / stdDev;
			return mappedAvg + mappedDev * l;
		}
		else
		{
			float s = math::sign(f - avg);
			float b = avg + stdDev * s;
			float m = mappedAvg + mappedDev * s;
			float l = math::abs(f - b);
			float range = 0.5f - mappedDev;
			return m + l * range * s;
		}*/
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

CUDA_GLOBAL void drawPass(e_Image img, k_SamplerpixelData* lumData, float* blockData, unsigned int* contribPixels, Colorizer col, bool blocks, int nx)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int ix = x / blockSize, iy = y / blockSize;
	unsigned int idx2 = iy * nx + ix;
	if (x < img.getWidth() && y < img.getHeight())
	{
		float VAR = 0;
		if (blocks)
			VAR = blockData[idx2] / float(contribPixels[idx2]);
		else VAR = weight(lumData[y * img.getWidth() + x], false);
		img.SetSample(x, y, Spectrum(col(VAR)).toRGBCOL());
	}
}

void k_BlockSampler::DrawVariance(bool blocks) const
{
	Colorizer col = Colorizer::FromData(m_pHostBlockData, totalNumBlocks());
	drawPass << < dim3(img->getWidth() / 32 + 1, img->getHeight() / 32 + 1), dim3(32, 32) >> >(*img, m_pLumData, m_pDeviceBlockData, m_pDeviceContribPixels, col, blocks, numBlocksRow());
	img->disableUpdate();
}