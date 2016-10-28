#pragma once

#include "Filter.h"
#include <Engine/Filter.h>

namespace CudaTracerLib
{

class CanonicalFilter : public ImageSamplesFilter
{
private:
	Filter m_filter;
public:
	CanonicalFilter(const Filter& f)
		: m_filter(f)
	{
	}
	virtual void Free()
	{

	}
	virtual void Resize(int xRes, int yRes)
	{

	}
	virtual void Apply(Image& img, int numPasses, float splatScale);
	const Filter& getFilter() const
	{
		return m_filter;
	}
	void setFilter(const Filter& f)
	{
		m_filter = f;
	}
};

}