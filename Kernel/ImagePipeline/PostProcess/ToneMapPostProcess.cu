#include "ToneMapPostProcess.h"

namespace CudaTracerLib
{

CUDA_FUNC_IN unsigned int FloatToUInt(float f)
{
	int mask = -int(*(unsigned int*)&f >> 31) | 0x80000000;
	return (*(unsigned int*)&f) ^ mask;
}

CUDA_FUNC_IN float UIntToFloat(unsigned int f)
{
	unsigned int mask = ((f >> 31) - 1) | 0x80000000, q = f ^ mask;
	return *(float*)&q;
}

///Reinhard Tone Mapping Operator
CUDA_DEVICE float g_LogLum;
CUDA_DEVICE unsigned int g_MaxLum;
CUDA_GLOBAL void rtm_SumLogLum(Image img, unsigned int w, unsigned int h)
{
	CUDA_SHARED float s_LogLum;
	CUDA_SHARED unsigned int s_MaxLum;
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		s_LogLum = s_MaxLum = 0;
		__syncthreads();
		Spectrum L_w;
		L_w.fromRGBE(img.getFilteredData(x, y));
		float f2 = L_w.getLuminance();
		float logLum = logf(0.0001f + f2);
		atomicAdd(&g_LogLum, logLum);
		atomicMax(&g_MaxLum, FloatToUInt(f2));
		__syncthreads();
		if (!threadIdx.x && !threadIdx.y)
		{
			atomicAdd(&g_LogLum, s_LogLum);
			atomicMax(&g_MaxLum, s_MaxLum);
		}
	}
}

CUDA_GLOBAL void rtm_Scale(Image img, unsigned int w, unsigned int h, float L_w, float alpha, float L_white2)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		Spectrum col;
		col.fromRGBE(img.getFilteredData(x, y));
		Vec3f yxy;
		col.toYxy(yxy.x, yxy.y, yxy.z);
		if (yxy.x < 1e-3f)
			return;
		float L = alpha / L_w * yxy.x;
		float L_d = (L * (1.0f + L / L_white2)) / (1.0f + L);
		yxy.x = L_d;
		col.fromYxy(yxy.x, yxy.y, yxy.z);
		img.getProcessedData(x, y) = col.toRGBCOL();
	}
}

void ToneMapPostProcess::Apply(Image& img, int numPasses)
{
	int block = 32, xResolution = img.getWidth(), yResolution = img.getHeight();
	float Lum_avg = 0;
	unsigned int val = FloatToUInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_LogLum, &Lum_avg, sizeof(Lum_avg)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxLum, &val, sizeof(unsigned int)));
	rtm_SumLogLum << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	ThrowCudaErrors(cudaThreadSynchronize());
	ThrowCudaErrors(cudaMemcpyFromSymbol(&Lum_avg, g_LogLum, sizeof(Lum_avg)));
	unsigned int mLum;
	ThrowCudaErrors(cudaMemcpyFromSymbol(&mLum, g_MaxLum, sizeof(unsigned int)));
	float maxLum = UIntToFloat(mLum);
	float L_w = exp(Lum_avg / float(xResolution * yResolution));
	//float middleGrey = 1.03f - 2.0f / (2.0f + log10(L_w + 1.0f));
	float alpha = 0.18f, lumWhite2 = max(maxLum * maxLum, 0.1f);
	rtm_Scale << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, L_w, alpha, lumWhite2);
}

}