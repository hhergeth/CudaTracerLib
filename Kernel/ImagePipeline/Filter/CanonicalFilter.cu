#include "CanonicalFilter.h"

namespace CudaTracerLib
{

CUDA_FUNC_IN Spectrum evalFilter(const Filter& filter, PixelData* P, float splatScale, unsigned int _x, unsigned int _y, unsigned int w, unsigned int h)
{
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
			acc += P[y * w + x].toSpectrum(splatScale) * filterWt;
			accFilter += filterWt;
		}
	}
	return acc / accFilter;
}

CUDA_GLOBAL void rtm_Copy(Image img, unsigned int w, unsigned int h, float splatScale, Filter filter)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		Spectrum c = evalFilter(filter, &img.getPixelData(0, 0), splatScale, x, y, w, h);
		img.getFilteredData(x, y) = c.toRGBE();
	}
}

void CanonicalFilter::Apply(Image& img, int numPasses, float splatScale)
{
	int block = 32, xResolution = img.getWidth(), yResolution = img.getHeight();
	rtm_Copy << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, splatScale, m_filter);

	ThrowCudaErrors(cudaThreadSynchronize());
}

}