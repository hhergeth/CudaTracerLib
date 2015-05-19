#include "e_Samples.h"
#include "e_DifferentialGeometry.h"

Vec3f BSDFSamplingRecord::getOutgoing()
{
	return normalize(dg.toWorld(wo));
}

