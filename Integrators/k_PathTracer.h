#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PathTracer : public k_Tracer<true, true>
{
public:
	bool m_Direct;
	k_PathTracer(bool direct = false)
		: m_Direct(direct)
	{
	}
	virtual void Debug(e_Image* I, const Vec2i& pixel, ITracerDebugger* debugger = 0);
protected:
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
};