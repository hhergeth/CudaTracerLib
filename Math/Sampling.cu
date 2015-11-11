#include "Sampling.h"
#include <Base/CudaRandom.h>
#include <Base/STL.h>

unsigned int MonteCarlo::sampleReuse(float *cdf, unsigned int size, float &sample, float& pdf)
{
	const float *entry = STL_lower_bound(cdf, cdf + size + 1, sample);
	unsigned int index = (unsigned int)min(max(0, int(entry - cdf) - 1), int(size - 1));
	pdf = cdf[index + 1] - cdf[index];
	sample = (sample - cdf[index]) / pdf;
	return index;
}

void MonteCarlo::stratifiedSample1D(CudaRNG& random, float *dest, int count, bool jitter)
{
	float invCount = 1.0f / count;

	for (int i = 0; i<count; i++) {
		float offset = jitter ? random.randomFloat() : 0.5f;
		*dest++ = (i + offset) * invCount;
	}
}

void MonteCarlo::stratifiedSample2D(CudaRNG& random, Vec2f *dest, int countX, int countY, bool jitter)
{
	float invCountX = 1.0f / countX;
	float invCountY = 1.0f / countY;

	for (int x = 0; x<countX; x++) {
		for (int y = 0; y<countY; y++) {
			float offsetX = jitter ? random.randomFloat() : 0.5f;
			float offsetY = jitter ? random.randomFloat() : 0.5f;
			*dest++ = Vec2f(
				(x + offsetX) * invCountX,
				(y + offsetY) * invCountY
				);
		}
	}
}

void MonteCarlo::RejectionSampleDisk(float *x, float *y, CudaRNG &rng)
{
	float sx, sy;
	do {
		sx = 1.f - 2.f * rng.randomFloat();
		sy = 1.f - 2.f * rng.randomFloat();
	} while (sx*sx + sy*sy > 1.f);
	*x = sx;
	*y = sy;
}

void MonteCarlo::StratifiedSample1D(float *samples, int nSamples, CudaRNG &rng, bool jitter)
{
	float invTot = 1.f / nSamples;
	for (int i = 0; i < nSamples; ++i)
	{
		float delta = jitter ? rng.randomFloat() : 0.5f;
		*samples++ = min((i + delta) * invTot, ONE_minUS_EPS);
	}
}

void MonteCarlo::StratifiedSample2D(float *samples, int nx, int ny, CudaRNG &rng, bool jitter)
{
	float dx = 1.f / nx, dy = 1.f / ny;
	for (int y = 0; y < ny; ++y)
		for (int x = 0; x < nx; ++x)
		{
			float jx = jitter ? rng.randomFloat() : 0.5f;
			float jy = jitter ? rng.randomFloat() : 0.5f;
			*samples++ = min((x + jx) * dx, ONE_minUS_EPS);
			*samples++ = min((y + jy) * dy, ONE_minUS_EPS);
		}
}

void MonteCarlo::latinHypercube(CudaRNG& random, float *dest, unsigned int nSamples, size_t nDim)
{
	float delta = 1 / (float)nSamples;
	for (size_t i = 0; i < nSamples; ++i)
		for (size_t j = 0; j < nDim; ++j)
			dest[nDim * i + j] = (i + random.randomFloat()) * delta;
	for (size_t i = 0; i < nDim; ++i) {
		for (size_t j = 0; j < nSamples; ++j) {
			unsigned int other = math::Floor2Int(float(nSamples) * random.randomFloat());
			swapk(dest + nDim * j + i, dest + nDim * other + i);
		}
	}
}