#include "ToneMapPostProcess.h"

namespace CudaTracerLib
{

CUDA_GLOBAL void Reinhard05Kernel(Image img, float scale, float invWp2)
{
	unsigned int _x = threadIdx.x + blockDim.x * blockIdx.x, _y = threadIdx.y + blockDim.y * blockIdx.y;
	if (_x < img.getWidth() && _y < img.getHeight())
	{
		Spectrum color;
		color.fromRGBE(img.getFilteredData(_x, _y));

		float x, y, Y;
		color.toYxy(Y, x, y);

		float Lp = scale * Y;
		Y = Lp * (1.0f + Lp*invWp2) / (1.0f + Lp);

		color.fromYxy(Y, x, y);

		img.getProcessedData(_x, _y) = color.toRGBCOL();
	}
}

void ToneMapPostProcess::Apply(Image& img, int numPasses)
{
	Spectrum Cav;
	float logAvgLuminance, minLum, maxLuminance, Lav;
	img.ComputeLuminanceInfo(Cav, minLum, maxLuminance, Lav, logAvgLuminance);

	
	float scale = m_key / logAvgLuminance,
		  Lwhite = maxLuminance * scale;
	auto burn = min(1.0f, max(1e-8f, 1.0f - m_burn));
	float invWp2 = 1 / (Lwhite * Lwhite * std::pow(burn, 4.0f));
	int block = 32;
	Reinhard05Kernel << <dim3(img.getWidth() / block + 1, img.getHeight() / block + 1), dim3(block, block) >> > (img, scale, invWp2);
}

}