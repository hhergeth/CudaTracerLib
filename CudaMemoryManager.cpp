#include <StdAfx.h>
#include "CudaMemoryManager.h"

namespace CudaTracerLib {

std::map<void*, CudaMemoryEntry> CudaMemoryManager::alloced_entries;
std::vector<CudaMemoryEntry> CudaMemoryManager::freed_entries;

cudaError_t CudaMemoryManager::Cuda_malloc_managed(void** v, size_t i, const std::string& callig_func)
{
	cudaError_t r = cudaMalloc(v, i);
	if (r == cudaError_t::cudaSuccess)
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
	cudaError_t r = cudaError_t::cudaSuccess;
	if (alloced_entries.count(v))
	{
		CudaMemoryEntry e = alloced_entries[v];
		if (e.address == 0)
			throw std::runtime_error("Trying to free cuda memory multiple times!");
		//alloced_entries.erase(v);
		e.free_func = callig_func;
		freed_entries.push_back(e);
		r = cudaFree(v);
		ThrowCudaErrors(r);
	}
	else throw std::runtime_error("Trying to free cuda memory without allocating it correctly!");
	return r;
}

}
