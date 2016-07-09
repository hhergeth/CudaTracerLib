#pragma once
#include <stdexcept>
#include <Defines.h>
#include <string>
#include <memory>

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

//code is from this great answer : http://stackoverflow.com/a/26221725/1715849
template<typename ... Args> std::string format(const std::string& format, Args ... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

}