#pragma once

template<typename T> CUDA_FUNC_IN void STL_Sort(T* a_Array, unsigned int a_Length, int (*cmp)(T*, T*))
{
	for (int i = 0; i < a_Length -1; ++i)
		for (int j = 0; j < a_Length - i - 1; ++j)
			if (cmp(a_Array[j], a_Array[j + 1]) > 0) 
				swapk(a_Array[j], a_Array[j + 1]);
}

template<typename T> CUDA_FUNC_IN void STL_Sort(T* a_Array, unsigned int a_Length)
{
	for (int i = 0; i < a_Length -1; ++i)
		for (int j = 0; j < a_Length - i - 1; ++j)
			if (a_Array[j] > a_Array[j + 1]) 
				swapk(a_Array[j], a_Array[j + 1]);
}

template<typename T> CUDA_FUNC_IN const T* STL_upper_bound(const T* first, const T* last, const T& value)
{
	const T* f = first;
	do
	{
		if(*f > value)
			return f;
	}
	while(f++ <= last);
	return first;
}