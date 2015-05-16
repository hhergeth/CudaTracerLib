#pragma once
#include "..\Defines.h"
#include <string>

class Platform
{
public:
	CUDA_DEVICE CUDA_HOST static unsigned int Increment(unsigned int* add);
	CUDA_DEVICE CUDA_HOST static unsigned int Add(unsigned int* add, unsigned int val);
	CUDA_DEVICE CUDA_HOST static unsigned int Exchange(unsigned int* add, unsigned int val);

	CUDA_DEVICE CUDA_HOST static float Add(float* add, float val);

	CUDA_HOST static void SetMemory(void* dest, unsigned long long length, unsigned int val = 0);
	CUDA_HOST static void OutputDebug(const std::string& msg);
};

#define ZERO_MEM(ref) Platform::SetMemory(&ref, sizeof(ref), 0)