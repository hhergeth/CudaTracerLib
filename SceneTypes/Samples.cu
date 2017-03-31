#include "Samples.h"
#include <Engine/DifferentialGeometry.h>

namespace CudaTracerLib {

NormalizedT<Vec3f> BSDFSamplingRecord::getOutgoing() const
{
	return dg.sys.toWorld(wo);
}

}