#include "Vec3fPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"

namespace CudaTracerLib {

struct Vec3f_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<Vec3f>& buffer, bool arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg)
			col = Spectrum((val.x + 1) / 2, (val.y + 1) / 2, (val.z + 1) / 2);
		else col = Spectrum(val.x, val.y, val.z);

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<Vec3f>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	Launch(img, *this, m_normalize, Vec3f_op());
}


void PixelDebugVisualizer<Vec3f>::VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer)
{

}

}