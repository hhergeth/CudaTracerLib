#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_SpatialGrid.h"

void plotPoints(float3* dirs, unsigned int N);

struct SpatialEntry;
struct DirectionModel;

class k_PmmTracer : public k_ProgressiveTracer
{
public:
	k_PmmTracer();
	virtual void Debug(int2 pixel);
	virtual void Resize(unsigned int _w, unsigned int _h)
	{

	}
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
private:
	bool doTrace;
	e_SpatialLinkedMap<SpatialEntry> sMap;
	e_SpatialSet<DirectionModel> dMap;
};