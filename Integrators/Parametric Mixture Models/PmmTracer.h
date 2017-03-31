#pragma once

#include <Kernel/Tracer.h>
#include <Engine/SpatialStructures/SpatialGrid.h>
#include "PmmHelper.h"

namespace CudaTracerLib {

void plotPoints(Vec3f* dirs, unsigned int N);

struct SpatialEntry;
struct DirectionModel;

//Incomplete implementation of "On-line Learning of Parametric Mixture Models for Light Transport Simulation"

class PmmTracer : public Tracer<true>
{
public:
	CTL_EXPORT PmmTracer();
	~PmmTracer()
	{
		sMap.Free();
		dMap.Free();
	}
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
private:
	int passIteration;
	SpatialLinkedMap<SpatialEntry> sMap;
	SpatialSet<DirectionModel> dMap;
};

}