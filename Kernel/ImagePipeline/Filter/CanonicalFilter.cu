#include "CanonicalFilter.h"

namespace CudaTracerLib
{

CUDA_FUNC_IN Spectrum evalFilter(const Filter& filter, PixelData* P, float splatScale, int _x, int _y, int w, int h)
{
	int x0 = max(0, math::Ceil2Int(_x - filter.As<FilterBase>()->xWidth));
	int x1 = min(w - 1, math::Floor2Int(_x + filter.As<FilterBase>()->xWidth));
	int y0 = max(0, math::Ceil2Int(_y - filter.As<FilterBase>()->yWidth));
	int y1 = min(h - 1, math::Floor2Int(_y + filter.As<FilterBase>()->yWidth));
	if ((x1 - x0) < 0 || (y1 - y0) < 0)
		return Spectrum(0.0f);
	Spectrum acc(0.0f);
	float accFilter = 0;
	for (int y = y0; y <= y1; ++y)
	{
		for (int x = x0; x <= x1; ++x)
		{
			float filterWt = filter.Evaluate((float)math::abs(x - _x), (float)math::abs(y - _y));
			acc += P[y * w + x].toSpectrum(splatScale) * filterWt;
			accFilter += filterWt;
		}
	}
	return acc / accFilter;
}

CUDA_GLOBAL void rtm_Copy(Image img, int w, int h, float splatScale, Filter filter)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		Spectrum c = evalFilter(filter, &img.getPixelData(0, 0), splatScale, x, y, w, h);
		img.getFilteredData(x, y) = c.toRGBE();
	}
}

void CanonicalFilter::Apply(Image& img, int numPasses, float splatScale, const PixelVarianceBuffer& varBuffer)
{
	int block = 16, xResolution = img.getWidth(), yResolution = img.getHeight();
	rtm_Copy << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, (int)xResolution, (int)yResolution, splatScale, m_filter);

	ThrowCudaErrors(cudaThreadSynchronize());
}

}