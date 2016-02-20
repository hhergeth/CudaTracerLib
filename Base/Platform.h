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

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Increment(unsigned int* add);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Add(unsigned int* add, unsigned int val);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Exchange(unsigned int* add, unsigned int val);

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float Add(float* add, float val);

	//standard x86 memset which uses a 8 bit as value
	CTL_EXPORT CUDA_HOST static void SetMemory(void* dest, size_t length, unsigned char val = 0);
	//memset which uses 32 bit as value
	CTL_EXPORT CUDA_HOST static void SetMemoryExt(void* dest, size_t length, unsigned int val = 0);
	CTL_EXPORT CUDA_HOST static void OutputDebug(const std::string& msg);
};

#define ZERO_MEM(ref) Platform::SetMemory(&ref, sizeof(ref), 0)

CTL_EXPORT std::string vformat(const char *fmt, va_list ap);

inline std::string format(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	std::string buf = vformat(fmt, ap);
	va_end(ap);
	return buf;
}

}
