#pragma once

#include <Kernel/Tracer.h>
#include <Engine/SpatialGrid.h>
#include "PmmHelper.h"

namespace CudaTracerLib {

void plotPoints(Vec3f* dirs, unsigned int N);

struct SpatialEntry;
struct DirectionModel;

//Incomplete implementation of "On-line Learning of Parametric Mixture Models for Light Transport Simulation"

class PmmTracer : public Tracer<false, true>
{
public:
	PmmTracer();
	~PmmTracer()
	{
		sMap.Free();
		dMap.Free();
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
protected:
	virtual void DoRender(Image* I);
	virtual void StartNewTrace(Image* I);
private:
	int passIteration;
	SpatialLinkedMap<SpatialEntry> sMap;
	SpatialSet<DirectionModel> dMap;
};

}