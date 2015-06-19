#include "StdAfx.h"
#include "Defines.h"
#include <stdio.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>

#ifdef ISWINDOWS
#include <Windows.h>
#endif

void fail(const char* format, ...)
{

	va_list arglist;
	va_start(arglist, format);
	vprintf(format, arglist);
	va_end(arglist);

#ifdef ISWINDOWS
	if (IsDebuggerPresent())
		__debugbreak();
#endif

	// Kill the app.

	throw std::runtime_error("");
}

void ThrowCudaErrors()
{
	cudaError r = cudaGetLastError();
	if (r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}

void ThrowCudaErrors(cudaError r)
{
	if (r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}

void CudaSetToZero(void* dest, size_t length)
{
	static void* zeroBuf = 0;
	static size_t zeroBufLength = 0;
	if (!zeroBuf || zeroBufLength < length)
	{
		if (zeroBuf)
			free(zeroBuf);
		zeroBufLength = RND_16(Dmax2(length, zeroBufLength));
		zeroBuf = malloc(zeroBufLength);
		for (int i = 0; i < zeroBufLength / 8; i++)
			*((unsigned long long*)zeroBuf + i) = 0;
	}
	cudaMemcpy(dest, zeroBuf, length, cudaMemcpyHostToDevice);
}