#pragma once

template<typename T> struct e_KernelBuffer
{
	T* Data;
	unsigned int UsedCount;
	unsigned int Length;
	CUDA_FUNC_IN T& operator[](unsigned int i) const
	{
		return Data[i];
	}
};