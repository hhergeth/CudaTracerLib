#include "PixelVarianceBuffer.h"
#include "Tracer.h"

namespace CudaTracerLib
{

CUDA_GLOBAL void updateVarianceBuffer(PixelVarianceInfo* data, Image img, unsigned int numPasses, float splatScale)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < img.getWidth() && y < img.getHeight())
	{
		data[y * img.getWidth() + x].updateMoments(img.getPixelData(x, y), numPasses, splatScale);
	}
}

void PixelVarianceBuffer::AddPass(Image& img, float splatScale)
{
	m_numPasses++;

	const int cBlock = 32;
	updateVarianceBuffer << <dim3(img.getWidth() / cBlock + 1, img.getHeight() / cBlock + 1), dim3(cBlock, cBlock) >> > (m_pixelBuffer.getDevicePtr(), img, m_numPasses, splatScale);
	m_pixelBuffer.setOnGPU();
}

}