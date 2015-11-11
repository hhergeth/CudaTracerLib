#pragma once

#include <string>
#include "Platform.h"
#include <vector>

template<typename T, int LENGTH, bool RESET = true, unsigned char RESET_VALUE = 0> struct FixedSizeArray
{
	FixedSizeArray()
	{
		length = 0;
		if (RESET)
			Platform::SetMemory(buffer, LENGTH * sizeof(T), RESET_VALUE);
	}

	FixedSizeArray(const T* data, size_t dataLength)
	{
		set(data, dataLength);
	}

	FixedSizeArray(const std::vector<T>& data)
	{
		set(&data[0], data.size());
	}

	void set(const T* data, size_t dataLength)
	{
		if (dataLength > LENGTH)
			throw std::runtime_error(format("Trying to copy %ul entries to fixed size array of length %d.", dataLength, LENGTH));
		length = (unsigned int)dataLength;
		for (unsigned int i = 0; i < length; i++)
			operator()(i) = data[i];
		if (RESET && LENGTH - length > 0)
			Platform::SetMemory(buffer + length, (LENGTH - length) * sizeof(T), RESET_VALUE);
	}

	std::vector<T> toVector() const
	{
		std::vector<T> res(length);
		for (unsigned int l = 0; l < length; l++)
			res.push_back(operator()(l));
		return res;
	}

	CUDA_FUNC_IN bool isFull() const
	{
		return size() == LENGTH;
	}

	CUDA_FUNC_IN T* ptr()
	{
		return buffer;
	}

	CUDA_FUNC_IN const T* ptr() const
	{
		return buffer;
	}

	CUDA_FUNC_IN size_t size() const
	{
		return length;
	}

	CUDA_FUNC_IN T& operator()(size_t idx)
	{
		CT_ASSERT(idx < length);
		return buffer[idx];
	}

	CUDA_FUNC_IN const T& operator()(size_t idx) const
	{
		CT_ASSERT(idx < length);
		return buffer[idx];
	}

	template <int LENGTH2, bool RESET2, unsigned char RESET_VALUE2> FixedSizeArray<T, LENGTH, RESET, RESET_VALUE>& operator=(const FixedSizeArray<T, LENGTH2, RESET2, RESET_VALUE2>& other)
	{
		static_assert(LENGTH2 <= LENGTH, "FixedSizeArray too long to copy!");
		if (this != &other)
			set(other.ptr(), other.size());
		return *this;
	}

	FixedSizeArray<T, LENGTH, RESET, RESET_VALUE>& operator=(const std::vector<T>& other)
	{
		set(&other[0], other.size());
		return *this;
	}

	void push_back(const T& val)
	{
		if (length == LENGTH)
			throw std::runtime_error("FixedSizeArray not long enough for push_back.");
		buffer[length++] = val;
	}

	void resize(size_t l)
	{
		if (l > LENGTH)
			throw std::runtime_error(format("FixedSizeArray is not long enough for resize of %ul.", l));
		length = (unsigned int)l;
	}
protected:
	unsigned int length;
	T buffer[LENGTH];
};