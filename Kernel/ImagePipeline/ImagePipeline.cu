#include "ImagePipeline.h"
#include <Kernel/Tracer.h>

namespace CudaTracerLib
{

void useFilter(ImagePipelineInfo info, Image& img, ImageSamplesFilter* filter)
{
	filter->Apply(img, info.numPasses, info.splatScale);
}

void useAllProcesses(ImagePipelineInfo info, Image& img, const std::vector<PostProcess*>& postProcesses)
{
	for (auto& pP : postProcesses)
	{
		pP->Apply(img, info.numPasses);
	}
}

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

void applyImagePipeline(ImagePipelineInfo info, Image& img, ImageSamplesFilter* filter, const std::vector<PostProcess*>& postProcesses)
{
	int block = 32, xResolution = img.getWidth(), yResolution = img.getHeight();
	if (!filter && postProcesses.size() == 0) //copy directly to output
	{
		copySamplesToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, info.splatScale);
	}
	else if(!filter) //copy to filtered data (Stage 2)
	{
		copySamplesToFiltered << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution, info.splatScale);
		useAllProcesses(info, img, postProcesses);
		applyGammaCorrectureToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}
	else if(postProcesses.size() == 0) //only use filter, copy to output
	{
		useFilter(info, img, filter);
		copyFilteredToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}
	else //use both
	{
		useFilter(info, img, filter);
		useAllProcesses(info, img, postProcesses);
		applyGammaCorrectureToOutput << <dim3(xResolution / block + 1, yResolution / block + 1), dim3(block, block) >> >(img, xResolution, yResolution);
	}

	ThrowCudaErrors(cudaThreadSynchronize());
}

ImagePipelineInfo constructImagePipelineInfo(const TracerBase& tracer)
{
	ImagePipelineInfo info = { tracer.getNumPassesDone() , tracer.getSplatScale()};

	return info;
}

}