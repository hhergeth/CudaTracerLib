#include "Image.h"
#include <cuda_surface_types.h>
#include <Base/Platform.h>
#include <Math/Vector.h>

#ifdef ISWINDOWS
#include <windows.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>
#endif

namespace CudaTracerLib {

void Image::AddSample(float sx, float sy, const Spectrum &_L)
{
	Spectrum L = _L;
	L.clampNegative();
	int x = math::Floor2Int(sx), y = math::Floor2Int(sy);
	if (x < 0 || x >= xResolution || y < 0 || y >= yResolution || L.isNaN() || !L.isValid())
		return;
	float rgb[3];
	L.toLinearRGB(rgb[0], rgb[1], rgb[2]);
	Pixel* pixel = getPixel(y * xResolution + x);
#ifdef ISCUDA
	for (int i = 0; i < 3; i++)
		atomicAdd(pixel->rgb + i, rgb[i]);
	atomicAdd(&pixel->weightSum, 1.0f);
#else
	for (int i = 0; i < 3; i++)
		pixel->rgb[i] += rgb[i];
	pixel->weightSum += 1.0f;
#endif
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
	Pixel* pixel = getPixel(y * xResolution + x);
	float rgb[3];
	L.toLinearRGB(rgb[0], rgb[1], rgb[2]);
#ifdef ISCUDA
	for(int i = 0; i < 3; i++)
		atomicAdd(pixel->rgbSplat + i, rgb[i]);
#else
	for(int i = 0; i < 3; i++)
		pixel->rgbSplat[i] += rgb[i];
#endif
}

void Image::SetSample(int x, int y, RGBCOL c)
{
	viewTarget[y * xResolution + x] = c;
}

CUDA_FUNC_IN Spectrum evalFilter(const Filter& filter, Image::Pixel* P, float splatScale, unsigned int _x, unsigned int _y, unsigned int w, unsigned int h)
{
	//return P[_y * w + _x].toSpectrum(splatScale);
	float dimageX = _x - 0.5f;
	float dimageY = _y - 0.5f;
	int x0 = math::Ceil2Int(dimageX - filter.As<FilterBase>()->xWidth);
	int x1 = math::Floor2Int(dimageX + filter.As<FilterBase>()->xWidth);
	int y0 = math::Ceil2Int(dimageY - filter.As<FilterBase>()->yWidth);
	int y1 = math::Floor2Int(dimageY + filter.As<FilterBase>()->yWidth);
	x0 = max(x0, 0);
	x1 = min(x1, 0 + int(w) - 1);
	y0 = max(y0, 0);
	y1 = min(y1, 0 + int(h) - 1);
	if ((x1 - x0) < 0 || (y1 - y0) < 0)
		return Spectrum(0.0f);
	Spectrum acc(0.0f);
	float accFilter = 0;
	for (int y = y0; y <= y1; ++y)
	{
		for (int x = x0; x <= x1; ++x)
		{
			float filterWt = filter.Evaluate(math::abs(x - dimageX), math::abs(y - dimageY));
			acc += P[y * w + x].toSpectrum(splatScale).saturate() * filterWt;
			accFilter += filterWt;
		}
	}
	return acc / accFilter;
}

CUDA_FUNC_IN RGBCOL gammaCorrecture(const Spectrum& c, float f)
{
	Spectrum c2;
	c.toSRGB(c2[0], c2[1], c2[2]);
	return Spectrum(c2 * f).toRGBCOL();
}

CUDA_GLOBAL void rtm_Copy(Image::Pixel* P, RGBCOL* viewTarget, unsigned int w, unsigned int h, float splatScale, Filter filter, float scale)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < w && y < h)
	{
		Spectrum c = evalFilter(filter, P, splatScale, x, y, w, h);
		viewTarget[y * w + x] = gammaCorrecture(c, scale);
	}
}

void Image::InternalUpdateDisplay()
{
	if(usedHostPixels)
	{
		ThrowCudaErrors(cudaMemcpy(cudaPixels, hostPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyHostToDevice));
	}

	int block = 32;
	rtm_Copy << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(cudaPixels, viewTarget, xResolution, yResolution, lastSplatVal, filter, m_fOutScale);

	ThrowCudaErrors(cudaThreadSynchronize());
}

void Image::Clear()
{
	usedHostPixels = false;
	Platform::SetMemory(hostPixels, sizeof(Pixel) * xResolution * yResolution);
	ThrowCudaErrors(cudaMemset(cudaPixels, 0, sizeof(Pixel) * xResolution * yResolution));
	ThrowCudaErrors(cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xResolution * yResolution));
}

}
