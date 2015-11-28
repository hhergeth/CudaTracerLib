#include <StdAfx.h>
#include "TracerSettings.h"

namespace CudaTracerLib {

TracerParameterCollection& operator<<(TracerParameterCollection& lhs, const std::string& name)
{
	if (lhs.lastName.size() != 0)
		throw std::runtime_error("Invalid state of TracerParameterCollection. Shift Parameter after name!");
	lhs.lastName = name;
	return lhs;
}

TracerParameterCollection& operator<<(TracerParameterCollection& lhs, ITracerParameter* para)
{
	if (lhs.lastName.size() == 0)
		throw std::runtime_error("Invalid state of TracerParameterCollection. Shift Parameter after name!");
	lhs.parameter[lhs.lastName] = std::unique_ptr<ITracerParameter>(para);
	lhs.lastName = "";
	return lhs;
}

}