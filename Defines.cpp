#include "StdAfx.h"
#include "Defines.h"
#include <stdio.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include <Base/Platform.h>

#ifdef ISWINDOWS
#include <Windows.h>
#endif

namespace CudaTracerLib {

void fail(const char* err_str)
{
#ifdef ISWINDOWS
	if (IsDebuggerPresent())
		__debugbreak();
#endif

	throw std::runtime_error(err_str);
}

void __ThrowCudaErrors__(const char* file, int line, ...)
{
	va_list ap;
	va_start(ap, line);
	int arg = va_arg(ap, int);
	va_end(ap);
	cudaError_t r = arg == -1 ? cudaGetLastError() : (cudaError_t)arg;
	if (r)
	{
		const char* msg = cudaGetErrorString(r);
		auto er = format("In file '%s' at line %d : %s\n", file, line, msg);
		std::cout << er;
		throw std::runtime_error(er);
	}
}

static void* zeroBuf = 0;
static size_t zeroBufLength = 0;

void CudaSetToZero_FreeBuffer()
{
	if (zeroBuf)
		free(zeroBuf);
}

void CudaSetToZero(void* dest, size_t length)
{
	if (!zeroBuf || zeroBufLength < length)
	{
		if (zeroBuf)
			free(zeroBuf);
		zeroBufLength = RND_16(Dmax2(length, zeroBufLength));
		zeroBuf = malloc(zeroBufLength);
		for (int i = 0; i < zeroBufLength / 8; i++)
			*((unsigned long long*)zeroBuf + i) = 0;
	}
	ThrowCudaErrors(cudaMemcpy(dest, zeroBuf, length, cudaMemcpyHostToDevice));
}

}