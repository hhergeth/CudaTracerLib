#include "PixelDebugVisualizer.h"
#include <Engine/Image.h>
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

CUDA_GLOBAL void flagKernel(unsigned int w, unsigned int h, IPixelDebugVisualizer::FeatureVisualizer features, SynchronizedBuffer<char> flagBuf)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		auto prim_ray = g_SceneData.GenerateSensorRay(x, y);
		auto res = traceRay(prim_ray);
		if (res.hasHit())
		{
			DifferentialGeometry dg;
			res.fillDG(dg);

			auto bary = Vec3f(dg.bary, 1 - dg.bary.x - dg.bary.y);
			bool is_feature = false;

			if (features == IPixelDebugVisualizer::FeatureVisualizer::Edge)
			{
				const float th = 1e-3f;
				if (bary.x < th || bary.y < th || bary.z < th)
					is_feature = true;
			}
			else if (features == IPixelDebugVisualizer::FeatureVisualizer::Vertex)
			{
				const float th = 1e-3f;
				if ((bary.x < th && bary.y < th) || (bary.y < th && bary.z < th) || (bary.x < th && bary.z < th))
					is_feature = true;
			}

			if(is_feature)
				flagBuf[y * w + x] = 1;
		}
	}
}

void IPixelDebugVisualizer::VisualizeFeatures(const IDebugDrawer& drawer, IPixelDebugVisualizer::FeatureVisualizer features)
{
	static SynchronizedBuffer<char> flagBuf(4096*4096);

	flagBuf.Memset((unsigned char)0);
	const int block = 16;
	flagKernel << < dim3(m_width / block + 1, m_height / block + 1), dim3(block, block) >> >(m_width, m_height, features, flagBuf);
	flagBuf.setOnGPU();
	flagBuf.Synchronize();

	for(unsigned int x = 0; x < m_width; x++)
		for(unsigned int y = 0; y < m_height; y++)
			if (flagBuf[y * m_width + x])
			{
				VisualizePixel(x, y, drawer);
			}
}

}