#include "ImagePipeline.h"
#include <Kernel/Tracer.h>

namespace CudaTracerLib
{

CUDA_FUNC_IN RGBCOL gammaCorrecture(const Spectrum& c)
{
	Spectrum c2;
	c.toSRGB(c2[0], c2[1], c2[2]);
	return Spectrum(c2).toRGBCOL();
}

CUDA_GLOBAL void copySamplesToOutput(Image img, int w, int h, float splatScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		img.getProcessedData(x, y) = gammaCorrecture(img.getPixelData(x, y).toSpectrum(splatScale));
	}
}

CUDA_GLOBAL void copySamplesToFiltered(Image img, int w, int h, float splatScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		img.getFilteredData(x, y) = img.getPixelData(x, y).toSpectrum(splatScale).toRGBE();
	}
}

CUDA_GLOBAL void copyFilteredToOutput(Image img, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		Spectrum s;
		s.fromRGBE(img.getFilteredData(x, y));
		img.getProcessedData(x, y) = gammaCorrecture(s);
	}
}

CUDA_GLOBAL void applyGammaCorrectureToOutput(Image img, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		Spectrum s;
		s.fromRGBCOL(img.getProcessedData(x, y));
		img.getProcessedData(x, y) = gammaCorrecture(s);
	}
}

void applyImagePipeline(const TracerBase& tracer, Image& img, ImageSamplesFilter* filter, PostProcess* process)
{
	auto numPasses = tracer.getNumPassesDone();
	auto splatScale = tracer.getSplatScale();
	auto pixelVarianceBuffer = tracer.getPixelVarianceBuffer();

	int block = 16, xResolution = img.getWidth(), yResolution = img.getHeight();
	if (!filter && !process) //copy directly to output
	{
		copySamplesToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, splatScale);
	}
	else if(!filter) //copy to filtered data (Stage 2)
	{
		copySamplesToFiltered << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, splatScale);
		process->Apply(img, numPasses, pixelVarianceBuffer);
		applyGammaCorrectureToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}
	else if(!process) //only use filter, copy to output
	{
		filter->Apply(img, numPasses, splatScale, pixelVarianceBuffer);
		copyFilteredToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}
	else //use both
	{
		filter->Apply(img, numPasses, splatScale, pixelVarianceBuffer);
		process->Apply(img, numPasses, pixelVarianceBuffer);
		applyGammaCorrectureToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}

	ThrowCudaErrors(cudaThreadSynchronize());
}

}