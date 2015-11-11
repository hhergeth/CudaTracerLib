#pragma once
#include <Defines.h>
#include <string>
#include <cstdarg>

class Platform
{
public:
	CUDA_DEVICE CUDA_HOST static unsigned int Increment(unsigned int* add);
	CUDA_DEVICE CUDA_HOST static unsigned int Add(unsigned int* add, unsigned int val);
	CUDA_DEVICE CUDA_HOST static unsigned int Exchange(unsigned int* add, unsigned int val);

	CUDA_DEVICE CUDA_HOST static float Add(float* add, float val);

	CUDA_HOST static void SetMemory(void* dest, unsigned long long length, unsigned int val = 0);
	CUDA_HOST static void OutputDebug(const std::string& msg);
};

#define ZERO_MEM(ref) Platform::SetMemory(&ref, sizeof(ref), 0)

inline std::string vformat(const char *fmt, va_list ap)
{
	int l = _vscprintf(fmt, ap) + 1;
	std::string str;
	str.resize(l);
	vsprintf((char*)str.c_str(), fmt, ap);
	return str;
}

inline std::string format(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	std::string buf = vformat(fmt, ap);
	va_end(ap);
	return buf;
}