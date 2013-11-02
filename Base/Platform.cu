#include <StdAfx.h>
#include "Platform.h"

#ifdef ISWINDOWS
#include <Windows.h>
#elif ISUNIX

#endif

unsigned int Platform::Increment(unsigned int* add)
{
#if defined(ISCUDA)
	return atomicInc(add, 0xffffffff);
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

void Platform::SetMemory(void* dest, unsigned long long length, unsigned int val)
{
#if defined(ISWINDOWS)
	ZeroMemory(dest, length);
#elif defined(ISUNIX)

#endif
}

void Platform::OutputDebug(const char* msg)
{
#if defined(ISWINDOWS)
	OutputDebugString(msg);
#elif defined(ISUNIX)

#endif
}