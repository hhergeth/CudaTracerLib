#include <StdAfx.h>
#include "Platform.h"
#include <string.h>

#ifdef ISWINDOWS
#include <Windows.h>
#elif ISUNIX

#endif

namespace CudaTracerLib {

unsigned int Platform::Increment(unsigned int* add)
{
#if defined(ISCUDA)
	return atomicInc(add, UINT_MAX);
#elif defined(ISWINDOWS)
	return InterlockedIncrement(add);
#elif defined(ISUNIX)
	unsigned int v = *add;
	*add++;
	return v;
#endif
}

unsigned int Platform::Add(unsigned int* add, unsigned int val)
{
#if defined(ISCUDA)
	return atomicAdd(add, val);
#elif defined(ISWINDOWS)
	return InterlockedAdd((long*)add, val);
#elif defined(ISUNIX)
	unsigned int v = *add;
	*add += val;
	return v;
#endif
}

unsigned int Platform::Exchange(unsigned int* add, unsigned int val)
{
#if defined(ISCUDA)
	return atomicExch(add, val);
#elif defined(ISWINDOWS)
	return InterlockedExchange(add, val);
#elif defined(ISUNIX)
	unsigned int old = *add;
	*add = val;
	return old;
#endif
}

float Platform::Add(float* add, float val)
{
#if defined(ISCUDA)
	return atomicAdd(add, val);
#else
	float f = *add;
	*add = f + val;
	return f;
#endif
}

void Platform::SetMemory(void* dest, unsigned long long length, unsigned int val)
{
	memset(dest, val, length);
}

void Platform::OutputDebug(const std::string& msg)
{
#if defined(ISWINDOWS)
	OutputDebugString(msg.c_str());
#elif defined(ISUNIX)

#endif
}

std::string vformat(const char *fmt, va_list ap)
{
	int l = vsnprintf(0, 0, fmt, ap);
	std::string str;
	str.resize(l);
	int n = vsnprintf((char*)str.c_str(), l, fmt, ap);
	if (n != l)
		throw std::runtime_error("Error formating string!");
	return str;
}

}