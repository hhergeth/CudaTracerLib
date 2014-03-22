#include "e_Image.h"
#include <cuda_surface_types.h>

#define BASE 20

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
	getPixel(sy * xResolution + sx)->N++;
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
			Pixel* pixel = getPixel(y * xResolution + x);
			float wh = filterWt * avg;
#ifdef ISCUDA
			for(int i = 0; i < 3; i++)
				atomicAdd(pixel->xyz + i, filterWt * xyz[i]);
			atomicAdd(&pixel->weightSum, filterWt);
			atomicAdd(&pixel->I, wh);
			atomicAdd(&pixel->I2, wh * wh);
#else
			for(int i = 0; i < 3; i++)
				pixel->xyz[i] += filterWt * xyz[i];
			pixel->weightSum += filterWt;
			pixel->I += wh;
			pixel->I2 += wh * wh;
#endif
		}
	}
}
#else
void e_Image::AddSample(int x, int y, const Spectrum &L)
{
	float xyz[3];
	L.toXYZ(xyz[0], xyz[1], xyz[2]);
	Pixel* pixel = getPixel(y * xResolution + x);
	pixel->N++;
	const float filterWt = 1.0f, wh = filterWt * L.average();
#ifdef ISCUDA
		for(int i = 0; i < 3; i++)
			atomicAdd(pixel->xyz + i, filterWt * xyz[i]);
		atomicAdd(&pixel->weightSum, filterWt);
		atomicAdd(&pixel->I, wh);
		atomicAdd(&pixel->I2, wh * wh);
#else
		for(int i = 0; i < 3; i++)
			pixel->xyz[i] += filterWt * xyz[i];
		pixel->weightSum += filterWt;
		pixel->I += wh;
		pixel->I2 += wh * wh;
#endif
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
	//int mask = -int(*(unsigned int*)&f >> 31) | 0x80000000;
	//return (*(unsigned int*)&f) ^ mask;
	return unsigned int(clamp(f, 0.0f, 100.0f) * 1000000);
}

CUDA_FUNC_IN float UIntToFloat(unsigned int f)
{
	//unsigned int mask = ((f >> 31) - 1) | 0x80000000, q = f ^ mask;
	//return *(float*)&q;
	return float(f) / 1000000.0f;
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
		s_LogLum = s_MaxLum = 0;
		__syncthreads();
		Spectrum L_w = P[y * w + x].toSpectrum(splatScale);
		float avg = L_w.average();
		float& E = P[y * w + x].E, &E2 = P[y * w + x].E2;
		if(P[y * w + x].N > BASE)
		{
			E += avg;
			E2 += avg * avg;
		}
		float f2 = L_w.getLuminance();
		float logLum = logf(0.0001f + f2);
		atomicAdd(&g_LogLum, logLum);
		atomicMax(&g_MaxLum, FloatToUInt(f2));
		__syncthreads();
		if(!threadIdx.x && !threadIdx.y)
		{
			atomicAdd(&g_LogLum, s_LogLum);
			atomicMax(&g_MaxLum, s_MaxLum);
		}
	}
}

struct memTarget
{
	RGBCOL* viewTarget;
	int w, h;

	CUDA_FUNC_IN  void operator()(int x, int y, RGBCOL c)
	{
		viewTarget[(h - y - 1) * w + x] = c;
	}
};

struct texTarget
{
	cudaSurfaceObject_t viewCudaSurfaceObject;
	int w, h;

	CUDA_ONLY_FUNC void operator()(int x, int y, RGBCOL c)
	{
		surf2Dwrite(c, viewCudaSurfaceObject, x * 4, h - y - 1);
	}
};

void e_Image::SetSample(int x, int y, RGBCOL c)
{
	if(outState == 1)
#ifdef ISCUDA
		surf2Dwrite(c, viewCudaSurfaceObject, x * 4, yResolution - y - 1);
#else
		;
#endif
	else viewTarget[(yResolution - y - 1) * xResolution + x] = c;
}

template<typename TARGET> CUDA_GLOBAL void rtm_Scale(e_Image::Pixel* P, TARGET T, unsigned int w, unsigned int h, float splatScale, float L_w, float alpha, float L_white2)
{
	unsigned int x = threadId % w, y = threadId / w;
	if(x < w && y < h)
	{
		float3 yxy;
		P[y * w + x].toSpectrum(splatScale).toYxy(yxy.x, yxy.y, yxy.z);
		float L = alpha / L_w * yxy.x;
		float L_d = (L * (1.0f + L / L_white2)) / (1.0f + L);
		yxy.x = L_d;
		Spectrum c;
		c.fromYxy(yxy.x, yxy.y, yxy.z);	
		T(x, y, c.toRGBCOL());
	}
}

template<typename TARGET> CUDA_GLOBAL void rtm_Copy(e_Image::Pixel* P, TARGET T, unsigned int w, unsigned int h, float splatScale, ImageDrawType TYPE, unsigned int NumFrame)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	CUDA_SHARED float sumI, sumI2;
	sumI = sumI2 = 0.0f;
	__syncthreads();
	if(x < w && y < h)
	{
		Spectrum c = P[y * w + x].toSpectrum(splatScale);
		float avg = c.average();
		unsigned int N = P[y * w + x].N;

		float& E = P[y * w + x].E, &E2 = P[y * w + x].E2;
		if(N > BASE)
		{
			E += avg;
			E2 += avg * avg;
		}

		if(TYPE == ImageDrawType::Normal)
		{
			T(x, y, c.toRGBCOL());
		}
		else if(TYPE == ImageDrawType::BlockVariance)
		{
			Spectrum c = P[y * w + x].toSpectrum(splatScale);
			atomicAdd(&sumI, avg);
			atomicAdd(&sumI2, avg * avg);
			__syncthreads();
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(variance(sumI, sumI2, blockDim.x * blockDim.y))));
		}
		else if(TYPE == ImageDrawType::PixelVariance)
		{
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(variance(P[y * w + x].I2, P[y * w + x].I, P[y * w + x].weightSum))));
		}
		else if(TYPE == ImageDrawType::BlockPixelVariance)
		{
			float var = variance(P[y * w + x].I2, P[y * w + x].I, P[y * w + x].weightSum);
			atomicAdd(&sumI, var);
			__syncthreads();
			float f = sumI / float(blockDim.x * blockDim.y);
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(f)));
		}
		else if(TYPE == ImageDrawType::AverageVariance)
		{
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(variance(E, E2, N - BASE) * 100)));
		}
		else if(TYPE == ImageDrawType::BlockAverageVariance)
		{
			atomicAdd(&sumI, variance(E, E2, N - BASE));
			__syncthreads();
			float f = sumI / float(blockDim.x * blockDim.y) * 100;
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(f)));
		}
		else if(TYPE == ImageDrawType::NumSamples)
		{
			float f = float(N) / float(NumFrame);
			T(x, y, SpectrumConverter::Float3ToCOLORREF(make_float3(f)));
		}
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
	T1.h = T2.h = yResolution;
	T1.viewTarget = viewTarget;
	T2.viewCudaSurfaceObject = viewCudaSurfaceObject;
	int block = 32;
	NumFrame++;
	if(drawStyle == ImageDrawType::HDR)
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
		float alpha = 0.18, lumWhite2 = MAX(maxLum * maxLum, 0.1f);
		if(outState == 1)
			rtm_Scale<<<dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block)>>>(cudaPixels, T2, xResolution, yResolution, splatScale, L_w, alpha, lumWhite2);
		else rtm_Scale<<<dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block)>>>(cudaPixels, T1, xResolution, yResolution, splatScale, L_w, alpha, lumWhite2);
	}
	else
	{
		if(outState == 1)
			rtm_Copy<<<dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block)>>>(cudaPixels, T2, xResolution, yResolution, splatScale, drawStyle, NumFrame);
		else rtm_Copy<<<dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block)>>>(cudaPixels, T1, xResolution, yResolution, splatScale, drawStyle, NumFrame);
	}
}

void e_Image::Clear()
{
	NumFrame = 0;
	usedHostPixels = false;
	Platform::SetMemory(hostPixels, sizeof(Pixel) * xResolution * yResolution);
	cudaMemset(cudaPixels, 0, sizeof(Pixel) * xResolution * yResolution);
	if(outState == 2)
		cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xResolution * yResolution);
	else if(outState == 1)
		cudaMemcpyToArray(viewCudaArray, 0, 0, viewTarget, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToDevice);
}

CUDA_GLOBAL void rtm_VarBuffer(e_Image::Pixel* P, float* T, unsigned int w, unsigned int h, float splatScale)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	CUDA_SHARED float sumI;
	sumI = 0.0f;
	if(x < w && y < h)
	{
		float var = variance(P[y * w + x].E, P[y * w + x].E2, P[y * w + x].N - BASE);
		atomicAdd(&sumI, var);
		__syncthreads();
		float f = sumI / float(blockDim.x * blockDim.y);
		if(threadIdx.x == 0 && threadIdx.y == 0)
			T[blockIdx.y * blockDim.x + blockIdx.x] = (f);
	}
}

static float* hostData = new float[2048 * 2048];
void calcVar(int block, float splatScale, float* deviceBuffer, e_Image::Pixel* deviceP, e_Image::Pixel* hostP, int w, int h)
{
	cudaMemcpy(hostP, deviceP, w * h * sizeof(e_Image::Pixel), cudaMemcpyDeviceToHost);
	int BlockDimX = int(ceilf(float(w) / block));
	int blockDimY = int(ceilf(float(h) / block));
	for(int i = 0; i < BlockDimX; i++)
		for(int j = 0; j < blockDimY; j++)
		{
			float sumVar = 0;
			for(int x = 0; x < block; x++)
				for(int y = 0; y < block; y++)
				{
					int cx = i * block + x, cy = j * block + y;
					if(cx >= w || cy >= h)
						continue;
					e_Image::Pixel& p = hostP[cx * w + cy];
					float var = p.E2 - p.E * p.E / float(p.N - BASE);
					//float e = p.E / float(p.N - BASE), e2 = p.E2 / float(p.N - BASE);
					//float var = abs(e2 - e * e);
					sumVar += var;
				}
			hostData[j * BlockDimX + i] = sumVar / float(block * block);
		}
		cudaMemcpy(deviceBuffer, hostData, BlockDimX * blockDimY * sizeof(float), cudaMemcpyHostToDevice);
}

bool e_Image::calculateBlockVariance(int block, float splatScale, float* deviceBuffer) const
{
	if(NumFrame <= BASE)
		return false;
	if(block > 32)
		throw new std::exception("block size <= 32");
	//calcVar(block, splatScale, deviceBuffer, cudaPixels, hostPixels, xResolution, yResolution); return true;
	rtm_VarBuffer<<<dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block)>>>(cudaPixels, deviceBuffer, xResolution, yResolution, splatScale);
	return true;
}