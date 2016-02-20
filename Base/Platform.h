#pragma once
#include <stdexcept>
#include <Defines.h>
#include <string>
#include <cstdarg>

namespace CudaTracerLib {

class Platform
{
public:
	//mimics cuda atomic functions as atomicInc, atomicAdd, atomicExch, atomicAdd

	CUDA_DEVICE CUDA_HOST static unsigned int Increment(unsigned int* add);
	CUDA_DEVICE CUDA_HOST static unsigned int Add(unsigned int* add, unsigned int val);
	CUDA_DEVICE CUDA_HOST static unsigned int Exchange(unsigned int* add, unsigned int val);

	CUDA_DEVICE CUDA_HOST static float Add(float* add, float val);

	CUDA_HOST static void SetMemory(void* dest, size_t length, unsigned char val = 0);
	CUDA_HOST static void OutputDebug(const std::string& msg);
};

#define ZERO_MEM(ref) Platform::SetMemory(&ref, sizeof(ref), 0)

std::string vformat(const char *fmt, va_list ap);

inline std::string format(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	std::string buf = vformat(fmt, ap);
	va_end(ap);
	return buf;
}

}
