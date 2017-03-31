#pragma once

#include "Vector.h"
#include "Spectrum.h"

namespace CudaTracerLib {

struct CudaRNG;

//Implementation of most methods copied from Mitsuba, some are PBRT material too.

class MonteCarlo
{
public:
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void RejectionSampleDisk(float *x, float *y, CudaRNG &rng);

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void StratifiedSample1D(float *samples, int nSamples, CudaRNG &rng, bool jitter = true);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void StratifiedSample2D(float *samples, int nx, int ny, CudaRNG &rng, bool jitter = true);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void stratifiedSample1D(CudaRNG& random, float *dest, int count, bool jitter);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void stratifiedSample2D(CudaRNG& random, Vec2f *dest, int countX, int countY, bool jitter);

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void latinHypercube(CudaRNG& random, float *dest, unsigned int nSamples, size_t nDim);

	CUDA_FUNC_IN static float BalanceHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		return (nf * fPdf) / (nf * fPdf + ng * gPdf);
	}

	CUDA_FUNC_IN static float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf, g = ng * gPdf;
		return (f*f) / (f*f + g*g);
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int sampleReuse(float *cdf, unsigned int size, float &sample, float& pdf);

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static void sampleReuse(unsigned int N, float& pdf, unsigned int& slot);
};

}