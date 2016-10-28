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

void Image::SetSample(int x, int y, RGBCOL c)
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

}
