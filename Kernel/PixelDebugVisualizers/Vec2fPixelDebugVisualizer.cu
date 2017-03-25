#include "Vec2fPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"

namespace CudaTracerLib {

struct Vec2f_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<Vec2f>& buffer, bool arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg)
			col = Spectrum((val.x + 1) / 2, (val.y + 1) / 2, 0.0f);
		else col = Spectrum(val.x, val.y, 0.0f);

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<Vec2f>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	Launch(img, *this, m_normalize, Vec2f_op());
}


void PixelDebugVisualizer<Vec2f>::VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer)
{

}

}