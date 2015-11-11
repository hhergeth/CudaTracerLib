#pragma once

#include <Kernel/k_Tracer.h>
#include <Engine/e_SpatialGrid.h>
#include "k_PmmHelper.h"

void plotPoints(Vec3f* dirs, unsigned int N);

struct SpatialEntry;
struct DirectionModel;

//Incomplete implementation of "On-line Learning of Parametric Mixture Models for Light Transport Simulation"

class k_PmmTracer : public k_Tracer<false, true>
{
public:
	k_PmmTracer();
	~k_PmmTracer()
	{
		sMap.Free();
		dMap.Free();
	}
	virtual void Debug(e_Image* I, const Vec2i& pixel);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
private:
	int passIteration;
	e_SpatialLinkedMap<SpatialEntry> sMap;
	e_SpatialSet<DirectionModel> dMap;
};