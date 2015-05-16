#include <StdAfx.h>
#include "e_Core.h"
#include "..\MathTypes.h"

#include "e_Core.h"

#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"

#define FREEIMAGE_LIB
#include <FreeImage.h>

#include "e_FileTexture.h"
#include <crtdbg.h>

#include "../Kernel/k_Tracer.h"

void InitializeCuda4Tracer(const std::string& dataPath)
{
#ifndef NDEBUG
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetDbgFlag(_CrtSetDbgFlag(0) | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
	cudaError er = CUDA_FREE(0);
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	e_RoughTransmittanceManager::StaticInitialize(dataPath);
}

void DeInitializeCuda4Tracer()
{
	k_TracerBase::g_sRngs.Free();
	FreeImage_DeInitialise();
	SpectrumHelper::StaticDeinitialize();
	e_RoughTransmittanceManager::StaticDeinitialize();
#ifndef NDEBUG
	//_CrtDumpMemoryLeaks();
#endif
}

void ThrowCudaErrors()
{
	cudaError r = cudaGetLastError();
	if(r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}

void ThrowCudaErrors(cudaError r)
{
	if(r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}

std::map<void*, CudaMemoryEntry> CudaMemoryManager::alloced_entries;
std::vector<CudaMemoryEntry> CudaMemoryManager::freed_entries;

cudaError_t CudaMemoryManager::Cuda_malloc_managed(void** v, size_t i, const std::string& callig_func)
{
	cudaError_t r = cudaMalloc(v, i);
	if(r == CUDA_SUCCESS)
	{
		CudaMemoryEntry e;
		e.address = *v;
		e.free_func = "";
		e.length = i;
		e.malloc_func = callig_func;
		alloced_entries[*v] = e;
	}
	else ThrowCudaErrors(r);
	return r;
}

cudaError_t CudaMemoryManager::Cuda_free_managed(void* v, const std::string& callig_func)
{
	if(alloced_entries.count(v))
	{
		CudaMemoryEntry e = alloced_entries[v];
		//alloced_entries.erase(v);
		e.free_func = callig_func;
		freed_entries.push_back(e);
	}
	cudaError_t r = cudaFree(v);
	ThrowCudaErrors(r);
	return r;
}