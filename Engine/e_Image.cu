#include "e_Image.h"

CUDA_FUNC_IN float3 COL(e_Image::Pixel* P, unsigned int i, float splatScale)
{
	float weightSum = P[i].weightSum;
	float3 rgb = P[i].rgb;
	if(weightSum != 0)
		rgb = fmaxf(make_float3(0), rgb / P[i].weightSum);
	rgb += splatScale * P[i].splatRgb;
	return rgb;
}

///Reinhard Tone Mapping Operator
CUDA_DEVICE float g_LogLum;

__global__ void rtm_SumLogLum(e_Image::Pixel* P, RGBCOL* T, unsigned int w, unsigned int h, float splatScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int i = y * w + x;
		CUDA_SHARED float s_LogLum;
		if(!blockIdx.x && !blockIdx.y)
			s_LogLum = 0;
		float logLum = logf(0.01f + ::y(COL(P, i, splatScale)));
		atomicAdd(&s_LogLum, logLum);
		if(!blockIdx.x && !blockIdx.y)
			atomicAdd(&g_LogLum, s_LogLum);
	}
}

__global__ void rtm_Scale(e_Image::Pixel* P, RGBCOL* T, unsigned int w, unsigned int h, float splatScale, float lumAvg, float middleGrey, float lumWhite2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int i = y * w + x;
		float3 c = COL(P, i, splatScale);
		float lumScaled = ::y(c) * middleGrey / lumAvg;
		float lumCompress = (lumScaled * (1.0f + lumScaled / lumWhite2)) / (1.0f + lumScaled);
		float3 yxy = XYZToYxy(RGBToXYZ(c));
		yxy.x = lumCompress;
		c = XYZToRGB(YxyToXYZ(yxy));
		T[i] = Float3ToCOLORREF(c);
	}
}

void e_Image::UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale)
{
	float Lum_avg = 0;
	cudaMemcpyToSymbol(g_LogLum, &Lum_avg, 4);
	rtm_SumLogLum<<<dim3(xResolution / 32, yResolution / 32), dim3(32, 32)>>>(cudaPixels, target, xResolution, yResolution, splatScale);
	cudaMemcpyFromSymbol(&Lum_avg, g_LogLum, 4);
	Lum_avg = exp(Lum_avg / float(xResolution * yResolution));
	float middleGrey = 1.03f - 2.0f / (2.0f + log10(Lum_avg + 1.0f));
	rtm_Scale<<<dim3(xResolution / 32, yResolution / 32), dim3(32, 32)>>>(cudaPixels, target, xResolution, yResolution, splatScale, Lum_avg, middleGrey, 4);
}