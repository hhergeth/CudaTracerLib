#include <StdAfx.h>
#include "Platform.h"
#include <string.h>
#include <cstdint>

#ifdef ISWINDOWS
#include <Windows.h>
#endif

namespace CudaTracerLib {

unsigned int Platform::Increment(unsigned int* add)
{
#if defined(ISCUDA)
	return atomicInc(add, UINT_MAX);
#elif defined(ISWINDOWS)
	return InterlockedExchangeAdd(add, 1);
#else
	return __sync_fetch_and_add(add, 1);
#endif
}

unsigned int Platform::Add(unsigned int* add, unsigned int val)
{
#if defined(ISCUDA)
	return atomicAdd(add, val);
#elif defined(ISWINDOWS)
	return InterlockedExchangeAdd(add, val);
#else
	return __sync_fetch_and_add(add, val);
#endif
}

unsigned int Platform::Exchange(unsigned int* add, unsigned int val)
{
#if defined(ISCUDA)
	return atomicExch(add, val);
#elif defined(ISWINDOWS)
	return InterlockedExchange(add, val);
#else
	return __atomic_exchange_n(add, val, __ATOMIC_SEQ_CST);
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

void Platform::SetMemory(void* dest, size_t length, unsigned char val)
{
	if (dest && length)
		memset(dest, val, length);
}

void Platform::SetMemoryExt(void* p, size_t n, unsigned int c)
{
	if (!p || !n)
		return;

	//code from here : http://www.xs-labs.com/en/blog/2013/08/06/optimising-memset/

	uint8_t  * sp;
	uint64_t * lp;
	uint64_t   u64;
	uint8_t    u8;

	u8 = (uint8_t)c;
	u64 = (uint64_t)c;
	u64 = (u64 << 32) | u64;
	sp = (uint8_t *)p;

	while (n-- && (((uint64_t)sp & (uint64_t)-8) < (uint64_t)sp))
	{
		*(sp++) = u8;
	}

	lp = (uint64_t *)((void *)sp);

	while ((n / 8) > 0)
	{
		*(lp++) = u64;
		n -= 8;
	}

	sp = (uint8_t *)((void *)lp);

	while (n--)
	{
		*(sp++) = u8;
	}
}

void Platform::OutputDebug(const std::string& msg)
{
#if defined(ISWINDOWS)
	OutputDebugString(msg.c_str());
#else

#endif
}

std::string vformat(const char *fmt, va_list ap)
{
	va_list ap2;
	va_copy(ap2, ap);
	int l = vsnprintf(0, 0, fmt, ap);
	std::string str;
	str.resize(l);
	int n = vsnprintf((char*)str.c_str(), l, fmt, ap2);
	if (n != l)
		throw std::runtime_error("Error formating string!");
	return str;
}

}
