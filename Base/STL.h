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

template<typename T> CUDA_FUNC_IN const T* STL_upper_bound(const T* _First, const T* _Last, const T& _Val)
{
	unsigned int _Count = _Last - _First;
	for (; 0 < _Count; )
	{
		unsigned int _Count2 = _Count / 2;
		const T* _Mid = _First + _Count2;

		if (!(_Val < *_Mid))
		{
			_First = ++_Mid;
			_Count -= _Count2 + 1;
		}
		else _Count = _Count2;
	}
	return (_First);
}

template<typename T> CUDA_FUNC_IN const T* STL_lower_bound(const T* _First, const T* _Last, const T& _Val)
{
	unsigned int _Count = _Last - _First;

	for (; 0 < _Count; )
	{
		unsigned int _Count2 = _Count / 2;
		const T* _Mid = _First + _Count2;

		if (*_Mid < _Val)
		{	
			_First = ++_Mid;
			_Count -= _Count2 + 1;
		}
		else _Count = _Count2;
	}
	return (_First);
}