#pragma once
#include <string>

namespace CudaTracerLib {

void InitializeCuda4Tracer(const std::string& dataPath);

void DeInitializeCuda4Tracer();

}