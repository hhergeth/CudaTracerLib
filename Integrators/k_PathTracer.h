#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PathTracer : public k_Tracer<true, true>
{
public:
	bool m_Direct, m_Regularization;
	k_PathTracer(bool direct = false, bool regularization = false)
		: m_Direct(direct), m_Regularization(regularization)
	{
	}
	virtual void Debug(e_Image* I, const Vec2i& pixel);
protected:
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
};