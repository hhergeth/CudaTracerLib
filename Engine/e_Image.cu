#include "e_Image.h"
#include "..\Base\FrameworkInterop.h"

void e_Image::AddSample(const CameraSample &sample, const Spectrum &L)
{
	const bool SPLAT = true;
	if(SPLAT)
	{
		float dimageX = sample.imageX - 0.5f;
		float dimageY = sample.imageY - 0.5f;
		int x0 = Ceil2Int (dimageX - filter.As<e_KernelFilterBase>()->xWidth);
		int x1 = Floor2Int(dimageX + filter.As<e_KernelFilterBase>()->xWidth);
		int y0 = Ceil2Int (dimageY - filter.As<e_KernelFilterBase>()->yWidth);
		int y1 = Floor2Int(dimageY + filter.As<e_KernelFilterBase>()->yWidth);
		x0 = MAX(x0, xPixelStart);
		x1 = MIN(x1, xPixelStart + xPixelCount - 1);
		y0 = MAX(y0, yPixelStart);
		y1 = MIN(y1, yPixelStart + yPixelCount - 1);
		if ((x1-x0) < 0 || (y1-y0) < 0)
			return;
		float invX = filter.As<e_KernelFilterBase>()->invXWidth, invY = filter.As<e_KernelFilterBase>()->invYWidth;
		for (int y = y0; y <= y1; ++y)
		{
			for (int x = x0; x <= x1; ++x)
			{
				// Evaluate filter value at $(x,y)$ pixel
				float fx = fabsf((x - dimageX) * invX * FILTER_TABLE_SIZE);
				float fy = fabsf((y - dimageY) * invY * FILTER_TABLE_SIZE);
				int ify = MIN(Floor2Int(fx), FILTER_TABLE_SIZE-1);
				int ifx = MIN(Floor2Int(fy), FILTER_TABLE_SIZE-1);
				int offset = ify * FILTER_TABLE_SIZE + ifx;
				float filterWt = filterTable[offset];

				// Update pixel values with filtered sample contribution
				Pixel* pixel = getPixel((y - yPixelStart) * xPixelCount + (x - xPixelStart));
#ifdef ISCUDA
				for(int i = 0;i < SPECTRUM_SAMPLES; i++)
					atomicAdd(&pixel->rgb[i], filterWt * L[i]);
				atomicAdd(&pixel->weightSum, filterWt);
#else
				pixel->rgb += filterWt * L;
				pixel->weightSum += filterWt;
#endif
			}
		}
	}
	else
	{
		int x = (int)sample.imageX, y = (int)sample.imageY;
		Pixel* pixel = getPixel((y - yPixelStart) * xPixelCount + (x - xPixelStart));
		pixel->rgb += L;
		pixel->weightSum += 1;
	}
}

void e_Image::Splat(const CameraSample &sample, const Spectrum &L)
{
	int x = Floor2Int(sample.imageX), y = Floor2Int(sample.imageY);
	if (x < xPixelStart || x - xPixelStart >= xPixelCount || y < yPixelStart || y - yPixelStart >= yPixelCount)
		return;
	Pixel* pixel = getPixel((y - yPixelStart) * xPixelCount + (x - xPixelStart));
#ifdef ISCUDA
	for(int i = 0;i < SPECTRUM_SAMPLES; i++)
		atomicAdd(&pixel->splatRgb[i], L[i]);
#else
	pixel->splatRgb += L;
#endif
}

void e_Image::SetSampleDirect(const CameraSample &sample, const Spectrum &L)
{
	int x = Floor2Int(sample.imageX), y = Floor2Int(sample.imageY);
	if (x < xPixelStart || x - xPixelStart >= xPixelCount || y < yPixelStart || y - yPixelStart >= yPixelCount)
		return;
	unsigned int off = (y - yPixelStart) * xPixelCount + (x - xPixelStart);
	Pixel* pixel = getPixel(off);
	pixel->rgb = L;
	pixel->weightSum = 1;
	pixel->splatRgb = Spectrum(0.0f);
	target[off] = L.toRGBCOL();
}

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

CUDA_FUNC_IN Spectrum COL(e_Image::Pixel* P, unsigned int i, float splatScale)
{
	float weightSum = P[i].weightSum;
	Spectrum rgb = P[i].rgb;
	if(weightSum != 0)
		rgb = fmaxf(Spectrum(0.0f), rgb / P[i].weightSum);
	rgb += splatScale * P[i].splatRgb;
	return fmaxf(rgb, Spectrum(0.01f));
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
		Spectrum L_w = COL(P, i, splatScale);
		float f2 = L_w.getLuminance();
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
		T[i] = COL(P, i, splatScale).toRGBCOL();
		/*
		float3 yxy;
		COL(P, i, splatScale).toYxy(yxy.x, yxy.y, yxy.z);
		float L = alpha / lumAvg * yxy.x;
		float L_d = (L * (1.0f + L / lumWhite2)) / (1.0f + L);
		yxy.x = L_d;
		Spectrum s;
		s.fromYxy(yxy.x, yxy.y, yxy.z);
		T[i] = s.toRGBCOL();;*/
	}
}

void e_Image::UpdateDisplay(float splatScale)
{
	if(usedHostPixels)
	{
		cudaMemcpy(cudaPixels, hostPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyHostToDevice);
	}
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