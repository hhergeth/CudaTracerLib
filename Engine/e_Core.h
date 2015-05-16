#pragma once

#include "cuda_runtime.h"

#include <string>

void InitializeCuda4Tracer(const std::string& dataPath);

void DeInitializeCuda4Tracer();

void ThrowCudaErrors();

void ThrowCudaErrors(cudaError_t r);

#include <map>
#include <vector>
#include <string>
struct CudaMemoryEntry
{
	void* address;
	size_t length;
	std::string malloc_func;
	std::string free_func;
};
class CudaMemoryManager
{
public:
	static std::map<void*, CudaMemoryEntry> alloced_entries;
	static std::vector<CudaMemoryEntry> freed_entries;
public:
	static cudaError_t Cuda_malloc_managed(void** v, size_t i, const std::string& callig_func);
	template<typename T> static cudaError_t Cuda_malloc_managed(T** v, size_t i, const std::string& callig_func)
	{
		return Cuda_malloc_managed((void**)v, i, callig_func);
	}
	static cudaError_t Cuda_free_managed(void* v, const std::string& callig_func);
	template<typename T> static cudaError_t Cuda_free_managed(T* v, const std::string&callig_func)
	{
		return Cuda_free_managed((void*)v, callig_func);
	}
};

#define CUDA_MALLOC(v,i) CudaMemoryManager::Cuda_malloc_managed(v, i, std::string(__FUNCSIG__))
#define CUDA_FREE(v) CudaMemoryManager::Cuda_free_managed(v, std::string(__FUNCSIG__))