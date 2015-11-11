#include <StdAfx.h>
#include "CudaMemoryManager.h"
#include <Kernel/k_Tracer.h>

std::map<void*, CudaMemoryEntry> CudaMemoryManager::alloced_entries;
std::vector<CudaMemoryEntry> CudaMemoryManager::freed_entries;

cudaError_t CudaMemoryManager::Cuda_malloc_managed(void** v, size_t i, const std::string& callig_func)
{
	cudaError_t r = cudaMalloc(v, i);
	if (r == CUDA_SUCCESS)
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
	if (alloced_entries.count(v))
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