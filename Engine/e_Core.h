#pragma once

#include "cuda_runtime.h"

void InitializeCuda4Tracer(const char* dataPath);

void ThrowCudaErrors();

void ThrowCudaErrors(cudaError_t r);

#include <map>
#include <vector>
struct CudaMemoryEntry
{
	void* address;
	size_t length;
	const char* malloc_func;
	const char* free_func;
};
class CudaMemoryManager
{
public:
	static std::map<void*, CudaMemoryEntry> alloced_entries;
	static std::vector<CudaMemoryEntry> freed_entries;
public:
	static cudaError_t Cuda_malloc_managed(void** v, size_t i, const char* callig_func);
	template<typename T> static cudaError_t Cuda_malloc_managed(T** v, size_t i, const char* callig_func)
	{
		return Cuda_malloc_managed((void**)v, i, callig_func);
	}
	static cudaError_t Cuda_free_managed(void* v, const char* callig_func);
	template<typename T> static cudaError_t Cuda_free_managed(T* v, const char* callig_func)
	{
		return Cuda_free_managed((void*)v, callig_func);
	}
};

#define CUDA_MALLOC(v,i) CudaMemoryManager::Cuda_malloc_managed(v, i, __FUNCSIG__)
#define CUDA_FREE(v) CudaMemoryManager::Cuda_free_managed(v, __FUNCSIG__)