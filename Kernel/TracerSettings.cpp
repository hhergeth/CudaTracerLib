#include <StdAfx.h>
#include "TracerSettings.h"

namespace CudaTracerLib {

TracerParameterCollection::InitHelper operator<<(TracerParameterCollection& lhs, const std::string& name)
{
	return TracerParameterCollection::InitHelper(lhs, name);
}

TracerParameterCollection& operator<<(TracerParameterCollection::InitHelper& lhs, ITracerParameter* para)
{
	lhs.add(para);
	return lhs.settings;
}

}