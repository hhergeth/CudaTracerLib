#include "e_Image.h"
#include "..\Base\FrameworkInterop.h"

CUDA_FUNC_IN unsigned int FloatToUInt(float f)
{
	unsigned int mask = -unsigned int(*(unsigned int*)&f >> 31) | 0x80000000;
	return (*(unsigned int*)&f) ^ mask;
}

CUDA_FUNC_IN float UIntToFloat(unsigned int f)
{
	unsigned int mask = ((f >> 31) - 1) | 0x80000000, q = f ^ mask;
	return *(float*)&q;
}

CUDA_FUNC_IN float3 COL(e_Image::Pixel* P, unsigned int i, float splatScale)
{
	float weightSum = P[i].weightSum;
	float3 rgb = P[i].rgb;
	if(weightSum != 0)
		rgb = fmaxf(make_float3(0), rgb / P[i].weightSum);
	rgb += splatScale * P[i].splatRgb;
	return fmaxf(rgb, make_float3(0.01f));
}

///Reinhard Tone Mapping Operator
CUDA_ALIGN(16) CUDA_DEVICE float g_LogLum;
CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_MaxLum;

CUDA_ALIGN(16) CUDA_SHARED float s_LogLum;
CUDA_ALIGN(16) CUDA_SHARED unsigned int s_MaxLum;
CUDA_GLOBAL void rtm_SumLogLum(e_Image::Pixel* P, RGBCOL* T, unsigned int w, unsigned int h, float splatScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int i = y * w + x, j = !(blockIdx.x | blockIdx.y);
		if(j)
		{
			s_LogLum = 0;
			s_MaxLum = 0;
		}
		float3 L_w = COL(P, i, splatScale);
		float f2 = ::y(L_w);
		float logLum = logf(0.01f + f2);
		atomicAdd(&g_LogLum, logLum);
		atomicMax(&g_MaxLum, FloatToUInt(f2));
		__syncthreads();
		if(j)
		{
			atomicAdd(&g_LogLum, s_LogLum);
			atomicMax(&g_MaxLum, s_MaxLum);
		}
	}
}

CUDA_GLOBAL void rtm_Scale(e_Image::Pixel* P, RGBCOL* T, unsigned int w, unsigned int h, float splatScale, float lumAvg, float alpha, float lumWhite2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int i = y * w + x;
		T[i] = Float3ToCOLORREF(COL(P, i, splatScale)); return;
		float3 yxy = XYZToYxy(RGBToXYZ(COL(P, i, splatScale)));
		float L = alpha / lumAvg * yxy.x;
		float L_d = (L * (1.0f + L / lumWhite2)) / (1.0f + L);
		yxy.x = L_d;
		T[i] = Float3ToCOLORREF(XYZToRGB(YxyToXYZ(yxy)));
	}
}

void e_Image::UpdateDisplay(float splatScale)
{
	CUDA_ALIGN(16) float Lum_avg = 0;
	unsigned int val = FloatToUInt(0);
	cudaError_t r = cudaMemcpyToSymbol(g_LogLum, &Lum_avg, sizeof(Lum_avg));
	r = cudaMemcpyToSymbol(g_MaxLum, &val, sizeof(unsigned int));
	rtm_SumLogLum<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, target, xResolution, yResolution, splatScale);
	r = cudaThreadSynchronize();
	r = cudaMemcpyFromSymbol(&Lum_avg, g_LogLum, sizeof(Lum_avg));
	unsigned int mLum;
	r = cudaMemcpyFromSymbol(&mLum, g_MaxLum, sizeof(unsigned int));
	float maxLum = UIntToFloat(mLum);
	float L_w = exp(Lum_avg / float(xResolution * yResolution));
	//float middleGrey = 1.03f - 2.0f / (2.0f + log10(L_w + 1.0f));
	float alpha = 0.35, lumWhite2 = MAX(maxLum * maxLum, 0.1f);
	rtm_Scale<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, target, xResolution, yResolution, splatScale, L_w, alpha, lumWhite2);
}