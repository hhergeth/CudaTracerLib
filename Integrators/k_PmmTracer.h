#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_SpatialGrid.h"

void plotPoints(Vec3f* dirs, unsigned int N);

struct SpatialEntry;
struct DirectionModel;

class k_PmmTracer : public k_Tracer<false, true>
{
public:
	k_PmmTracer();
	virtual void Debug(e_Image* I, const Vec2i& pixel);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
private:
	int passIteration;
	e_SpatialLinkedMap<SpatialEntry> sMap;
	e_SpatialSet<DirectionModel> dMap;
};