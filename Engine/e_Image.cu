#include "e_Image.h"

Spectrum e_Image::Pixel::toSpectrum(float splat)
{
	Spectrum s;
	s.fromXYZ(xyz[0], xyz[1], xyz[2]);

	if(weightSum != 0.0f)
		s = fmaxf(Spectrum(0.0f), s / weightSum);
	Spectrum s2;
	s2.fromXYZ(xyzSplat[0], xyzSplat[1], xyzSplat[2]);
	return s + s2 * splat;
}

void e_Image::AddSample(int sx, int sy, const Spectrum &_L)
{
	Spectrum L = _L;
	L.clampNegative();
	if(L.isNaN())
		return;
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
	const bool SPLAT = true;
	if(SPLAT)
	{
		float dimageX = sx - 0.5f;
		float dimageY = sy - 0.5f;
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
				float filterWt = filterTable[ifx][ify];

				// Update pixel values with filtered sample contribution
				Pixel* pixel = getPixel((y - yPixelStart) * xPixelCount + (x - xPixelStart));
#ifdef ISCUDA
				for(int i = 0; i < 3; i++)
					atomicAdd(pixel->xyz + i, filterWt * xyz[i]);
				atomicAdd(&pixel->weightSum, filterWt);
#else
				for(int i = 0; i < 3; i++)
					pixel->xyz[i] += filterWt * xyz[i];
				pixel->weightSum += filterWt;
#endif
			}
		}
	}
	else
	{
		int x = sx, y = sy;
		Pixel* pixel = getPixel((y - yPixelStart) * xPixelCount + (x - xPixelStart));
		for(int i = 0; i < 3; i++)
			pixel->xyz[i] += xyz[i];
		pixel->weightSum += 1;
	}
}

void e_Image::Splat(int sx, int sy, const Spectrum &L)
{
	if (sx < xPixelStart || sx - xPixelStart >= xPixelCount || sy < yPixelStart || sy - yPixelStart >= yPixelCount)
		return;
	Pixel* pixel = getPixel((sy - yPixelStart) * xPixelCount + (sx - xPixelStart));
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
#ifdef ISCUDA
	for(int i = 0; i < 3; i++)
		atomicAdd(pixel->xyzSplat + i, xyz[i]);
#else
	for(int i = 0; i < 3; i++)
		pixel->xyzSplat[i] += xyz[i];
#endif
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
		Spectrum L_w = P[i].toSpectrum(splatScale);
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
	unsigned int x = threadId % w, y = threadId / w;
	if(x < w && y < h)
	{
		unsigned int i = y * w + x;
		float3 yxy;
		P[i].toSpectrum(splatScale).toYxy(yxy.x, yxy.y, yxy.z);
		float L = alpha / lumAvg * yxy.x;
		float L_d = (L * (1.0f + L / lumWhite2)) / (1.0f + L);
		yxy.x = L_d;
		Spectrum s;
		s.fromYxy(yxy.x, yxy.y, yxy.z);
		T[i] = s.toRGBCOL();
	}
}

CUDA_GLOBAL void rtm_Copy(e_Image::Pixel* P, RGBCOL* T, unsigned int w, unsigned int h, float splatScale)
{
	unsigned int x = threadId % w, y = threadId / w;
	if(x < w && y < h)
		T[y * w + x] = P[y * w + x].toSpectrum(splatScale).toRGBCOL();
}

void e_Image::UpdateDisplay(bool forceHDR, float splatScale)
{
	if(!target)
		return;
	if(usedHostPixels)
	{
		cudaMemcpy(cudaPixels, hostPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyHostToDevice);
	}
	if(forceHDR || doHDR)
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
	else rtm_Copy<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, target, xResolution, yResolution, splatScale);
}