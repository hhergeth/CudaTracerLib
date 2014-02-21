#include "e_Image.h"
#include <cuda_surface_types.h>

//#define FAST_ADD_SAMPLE

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

#ifndef FAST_ADD_SAMPLE
void e_Image::AddSample(int sx, int sy, const Spectrum &_L)
{
	Spectrum L = _L;
	L.clampNegative();
	if(L.isNaN())
		return;
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
	float avg = L.average();
	float dimageX = sx - 0.5f;
	float dimageY = sy - 0.5f;
	int x0 = Ceil2Int (dimageX - filter.As<e_KernelFilterBase>()->xWidth);
	int x1 = Floor2Int(dimageX + filter.As<e_KernelFilterBase>()->xWidth);
	int y0 = Ceil2Int (dimageY - filter.As<e_KernelFilterBase>()->yWidth);
	int y1 = Floor2Int(dimageY + filter.As<e_KernelFilterBase>()->yWidth);
	x0 = MAX(x0, 0);
	x1 = MIN(x1, 0 + xResolution - 1);
	y0 = MAX(y0, 0);
	y1 = MIN(y1, 0 + yResolution - 1);
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
			//filterWt = filter.Evaluate(x - dimageX, y - dimageY);

			// Update pixel values with filtered sample contribution
			Pixel* pixel = getPixel((y - 0) * xResolution + (x - 0));
#ifdef ISCUDA
			for(int i = 0; i < 3; i++)
				atomicAdd(pixel->xyz + i, filterWt * xyz[i]);
			atomicAdd(&pixel->weightSum, filterWt);
#else
			for(int i = 0; i < 3; i++)
				pixel->xyz[i] += filterWt * xyz[i];
			pixel->weightSum += filterWt;
#endif
			float wh = filterWt * avg;
			pixel->I += wh;
			pixel->I2 += wh * wh;
		}
	}
}
#else
void e_Image::AddSample(int x, int y, const Spectrum &L)
{
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
	Pixel* pixel = getPixel(y * xResolution + x);
	const float filterWt = 1.0f;
#ifdef ISCUDA
		for(int i = 0; i < 3; i++)
			atomicAdd(pixel->xyz + i, filterWt * xyz[i]);
		atomicAdd(&pixel->weightSum, filterWt);
#else
		for(int i = 0; i < 3; i++)
			pixel->xyz[i] += filterWt * xyz[i];
		pixel->weightSum += filterWt;
#endif
	float f = filterWt * L.average();
	pixel->I += f;
	pixel->I2 += f * f;
}
#endif

void e_Image::Splat(int sx, int sy, const Spectrum &L)
{
	if (sx >= xResolution || sy >= yResolution)
		return;
	Pixel* pixel = getPixel(sy * xResolution + sx);
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
/*#ifdef ISCUDA
	for(int i = 0; i < 3; i++)
		atomicAdd(pixel->xyzSplat + i, xyz[i]);
#else*/
	for(int i = 0; i < 3; i++)
		pixel->xyzSplat[i] += xyz[i];
//#endif
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
CUDA_GLOBAL void rtm_SumLogLum(e_Image::Pixel* P, unsigned int w, unsigned int h, float splatScale)
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

struct memTarget
{
	RGBCOL* viewTarget;
	int w;

	CUDA_FUNC_IN  void operator()(int x, int y, RGBCOL c)
	{
		viewTarget[y * w + x] = c;
	}
};

struct texTarget
{
	cudaSurfaceObject_t viewCudaSurfaceObject;
	int w;

	CUDA_ONLY_FUNC void operator()(int x, int y, RGBCOL c)
	{
		surf2Dwrite(c, viewCudaSurfaceObject, x * 4, y);
	}
};

void e_Image::SetSample(int x, int y, RGBCOL c)
{
	if(outState == 1)
#ifdef ISCUDA
		surf2Dwrite(c, viewCudaSurfaceObject, x * 4, y);
#else
		;
#endif
	else viewTarget[y * xResolution + x] = c;
}

template<typename TARGET> CUDA_GLOBAL void rtm_Scale(e_Image::Pixel* P, TARGET T, unsigned int w, unsigned int h, float splatScale, float lumAvg, float alpha, float lumWhite2)
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
		T(x, y, s.toRGBCOL());
	}
}

template<typename TARGET> CUDA_GLOBAL void rtm_Copy(e_Image::Pixel* P, TARGET T, unsigned int w, unsigned int h, float splatScale)
{
	unsigned int x = threadId % w, y = threadId / w;
	if(x < w && y < h)
	{
		Spectrum c = P[y * w + x].toSpectrum(splatScale);
		T(x, y, c.toRGBCOL());
		float W = P[y * w + x].weightSum, i = P[y * w + x].I / W, i2 = P[y * w + x].I2 / W;
		float var = i2 - i * i;
		//T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(abs(var))));
	}
}

void e_Image::InternalUpdateDisplay(float splatScale)
{
	if(outState > 2)
		return;
	if(usedHostPixels)
	{
		cudaMemcpy(cudaPixels, hostPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyHostToDevice);
	}
	memTarget T1;
	texTarget T2;
	T1.w = T2.w = xResolution;
	T1.viewTarget = viewTarget;
	T2.viewCudaSurfaceObject = viewCudaSurfaceObject;
	if(doHDR)
	{
		CUDA_ALIGN(16) float Lum_avg = 0;
		unsigned int val = FloatToUInt(0);
		cudaError_t r = cudaMemcpyToSymbol(g_LogLum, &Lum_avg, sizeof(Lum_avg));
		r = cudaMemcpyToSymbol(g_MaxLum, &val, sizeof(unsigned int));
		rtm_SumLogLum<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, xResolution, yResolution, splatScale);
		r = cudaThreadSynchronize();
		r = cudaMemcpyFromSymbol(&Lum_avg, g_LogLum, sizeof(Lum_avg));
		unsigned int mLum;
		r = cudaMemcpyFromSymbol(&mLum, g_MaxLum, sizeof(unsigned int));
		float maxLum = UIntToFloat(mLum);
		float L_w = exp(Lum_avg / float(xResolution * yResolution));
		//float middleGrey = 1.03f - 2.0f / (2.0f + log10(L_w + 1.0f));
		float alpha = 0.35, lumWhite2 = MAX(maxLum * maxLum, 0.1f);
		if(outState == 1)
			rtm_Scale<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, T2, xResolution, yResolution, splatScale, L_w, alpha, lumWhite2);
		else rtm_Scale<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, T1, xResolution, yResolution, splatScale, L_w, alpha, lumWhite2);
	}
	else
	{
		if(outState == 1)
			rtm_Copy<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, T2, xResolution, yResolution, splatScale);
		else rtm_Copy<<<dim3(xResolution / 32 + 1, yResolution / 32 + 1), dim3(32, 32)>>>(cudaPixels, T1, xResolution, yResolution, splatScale);
	}
}

void e_Image::Clear()
{
	usedHostPixels = false;
	Platform::SetMemory(hostPixels, sizeof(Pixel) * xResolution * yResolution);
	cudaMemset(cudaPixels, 0, sizeof(Pixel) * xResolution * yResolution);
	if(outState == 2)
		cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xResolution * yResolution);
	else if(outState == 1)
		cudaMemcpyToArray(viewCudaArray, 0, 0, viewTarget, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToDevice);
}