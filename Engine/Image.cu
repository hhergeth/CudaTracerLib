#include "Image.h"
#include <cuda_surface_types.h>
#include <Base/Platform.h>
#include <Math/Vector.h>

namespace CudaTracerLib {

template<typename F> CUDA_FUNC_IN void splat(Image* img, float sx, float sy, int x, int y, int xResolution, int yResolution, const Spectrum& L, F add)
{
	add(img->getPixelData(x, y), L);

	/*float g = sx - x, f = sy - y;
	add(img->getPixelData(x, y), L * (1 - f) * (1 - g));
	if (x < xResolution - 1)
		add(img->getPixelData(x + 1, y), L * (1 - f) * (g));
	if (y < yResolution - 1)
		add(img->getPixelData(x, y + 1), L * (f) * (1 - g));
	if (x < xResolution - 1 && yResolution - 1)
		add(img->getPixelData(x + 1, y + 1), L * (f) * (g));*/
}

void Image::AddSample(float sx, float sy, const Spectrum &_L)
{
	Spectrum L = _L;
	L.clampNegative();
	int x = math::Floor2Int(sx), y = math::Floor2Int(sy);
	if (x < 0 || x >= xResolution || y < 0 || y >= yResolution || L.isNaN() || !L.isValid())
		return;
	//the times 4 is necessary because the samples will later be normalized * 4
	splat(this, sx, sy, x, y, xResolution, yResolution, L, [](PixelData& ref, const Spectrum& L)
	{
		float rgb[3];
		L.toLinearRGB(rgb[0], rgb[1], rgb[2]);
#ifdef ISCUDA
		for (int i = 0; i < 3; i++)
			atomicAdd(ref.rgb + i, rgb[i]);
		atomicAdd(&ref.weightSum, 1.0f);
#else
		for (int i = 0; i < 3; i++)
			ref.rgb[i] += rgb[i];
		ref.weightSum += 1.0f;
#endif
	});
}

void Image::Splat(float sx, float sy, const Spectrum &_L)
{
	if (_L.isNaN() || !_L.isValid())
		return;
	Spectrum L = _L;
	L.clampNegative();
	int x = math::Floor2Int(sx), y = math::Floor2Int(sy);
	if (x < 0 || x >= xResolution || y < 0 || y >= yResolution)
		return;

	splat(this, sx, sy, x, y, xResolution, yResolution, L, [](PixelData& ref, const Spectrum& L)
	{
		float rgb[3];
		L.toLinearRGB(rgb[0], rgb[1], rgb[2]);
#ifdef ISCUDA
		for (int i = 0; i < 3; i++)
			atomicAdd(ref.rgbSplat + i, rgb[i]);
#else
		for (int i = 0; i < 3; i++)
			ref.rgbSplat[i] += rgb[i];
#endif
	});

}

CUDA_DEVICE void Image::SetSample(int x, int y, RGBCOL c)
{
	m_viewTarget[y * xResolution + x] = c;
}

void Image::ClearSample(int sx, int sy)
{
	getPixelData(sx, sy) = PixelData();
}

void Image::Clear()
{
	m_pixelBuffer.Memset(0);
	ThrowCudaErrors(cudaMemset(m_filteredColorsDevice, 0, sizeof(RGBE) * xResolution * yResolution));
	ThrowCudaErrors(cudaMemset(m_viewTarget, 0, sizeof(RGBCOL) * xResolution * yResolution));
}

CUDA_DEVICE int g_minLum;
CUDA_DEVICE int g_maxLum;
CUDA_DEVICE float g_avgLum;
CUDA_DEVICE float g_avgLogLum;
CUDA_DEVICE Spectrum g_avgColor;
CUDA_GLOBAL void computeLuminanceInfo(Image img)
{
	CUDA_SHARED int s_minLum;
	CUDA_SHARED int s_maxLum;
	CUDA_SHARED float s_avgLum;
	CUDA_SHARED float s_avgLogLum;
	CUDA_SHARED Spectrum s_avgColor;

	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		s_maxLum = s_avgLum = s_avgLogLum = 0;
		s_minLum = INT_MAX;
		__syncthreads();
		Spectrum L_w;
		L_w.fromRGBE(img.getFilteredData(x, y));
		float Y = L_w.getLuminance();
		auto iY = floatToOrderedInt(Y);

		atomicMin(&s_minLum, iY);
		atomicMax(&s_maxLum, iY);
		atomicAdd(&s_avgLum, Y);
		atomicAdd(&s_avgLogLum, math::log(2.3e-5f + Y));
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			atomicAdd(&s_avgColor[i], L_w[i]);
	}

	__syncthreads();
	if (!threadIdx.x && !threadIdx.y)
	{
		atomicMin(&g_minLum, s_minLum);
		atomicMax(&g_maxLum, s_maxLum);
		atomicAdd(&g_avgLum, s_avgLum);
		atomicAdd(&g_avgLogLum, s_avgLogLum);
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			atomicAdd(&g_avgColor[i], s_avgColor[i]);
	}
}

void Image::ComputeLuminanceInfo(Spectrum& avgColor, float& minLum, float& maxLum, float& avgLum, float& avgLogLum)
{
	//host side implementation for correctness checking
	/*RGBE* hostData = (RGBE*)&m_pixelBuffer[0];
	cudaMemcpy(hostData, m_filteredColorsDevice, xResolution * yResolution * sizeof(RGBE), cudaMemcpyDeviceToHost);
	minLum = FLT_MAX; maxLum = avgLum = avgLogLum = 0;
	for(auto x = 0u; x < xResolution; x++)
		for (auto y = 0u; y < yResolution; y++)
		{
			Spectrum L_w;
			L_w.fromRGBE(hostData[y * xResolution + x]);
			float Y = L_w.getLuminance();

			minLum = min(minLum, Y);
			maxLum = max(maxLum, Y);
			avgLum += Y;
			avgLogLum += math::log(2.3e-5f + Y);
		}
	avgLum /= xResolution * yResolution;
	avgLogLum /= xResolution * yResolution;
	setOnGPU();
	Synchronize();
	return;*/

	const int block = 32;
	int iMinLum = INT_MAX, iMaxLum;
	ZeroSymbol(g_maxLum) ZeroSymbol(g_avgLum) ZeroSymbol(g_avgLogLum) CopyToSymbol(g_minLum, iMinLum) ZeroSymbol(g_avgColor)
	computeLuminanceInfo << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(*this);

	CopyFromSymbol(iMinLum, g_minLum) CopyFromSymbol(iMaxLum, g_maxLum) CopyFromSymbol(avgLum, g_avgLum) CopyFromSymbol(avgLogLum, g_avgLogLum) CopyFromSymbol(avgColor, g_avgColor)
	minLum = orderedIntToFloat(iMinLum);
	maxLum = orderedIntToFloat(iMaxLum);
	avgLum /= xResolution * yResolution;
	avgLogLum /= xResolution * yResolution;
	avgColor /= (float)xResolution * yResolution;
	avgLogLum = math::exp(avgLogLum);
}

}
