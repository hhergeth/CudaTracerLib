#pragma once
#include <string>
#include <Defines.h>

namespace CudaTracerLib {

CTL_EXPORT void InitializeCuda4Tracer(const std::string& dataPath);

CTL_EXPORT void DeInitializeCuda4Tracer();

}