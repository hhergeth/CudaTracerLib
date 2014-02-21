#pragma once

#include "cuda_runtime.h"

void InitializeCuda4Tracer();

void ThrowCudaErrors();

void ThrowCudaErrors(cudaError_t r);

#define CUDA_MALLOC cudaMalloc