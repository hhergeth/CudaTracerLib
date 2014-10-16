#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PathTracer : public k_ProgressiveTracer
{
public:
	bool m_Direct;
	k_PathTracer(bool direct = false)
		: m_Direct(direct)
	{
	}
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
};

#include "..\Kernel\k_BlockSampler.h"

class k_BlockPathTracer : public k_ProgressiveTracer
{
public:
	k_BlockSampler sampler;
	bool m_Direct;
	k_BlockPathTracer(bool direct = false)
		: m_Direct(direct)
	{
	}
protected:
	virtual void StartNewTrace(e_Image* I)
	{
		k_ProgressiveTracer::StartNewTrace(I);
		unsigned int _w, _h;
		I->getExtent(_w, _h);
		sampler.Initialize(_w, _h);
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		k_ProgressiveTracer::Resize(_w, _h);
		sampler.Initialize(_w, _h);
	}
	virtual void DoRender(e_Image* I);
};