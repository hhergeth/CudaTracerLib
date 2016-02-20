#pragma once

#include "FixedSizeArray.h"
#include <string.h>

namespace CudaTracerLib {

//stores null terminated strings
template<int LENGTH> struct FixedString : public FixedSizeArray<char, LENGTH, true, 0>
{
	FixedString()
		: FixedSizeArray<char, LENGTH>()
	{
	}

	FixedString(const std::string& s)
		: FixedSizeArray<char, LENGTH>(s.c_str(), s.length())
	{
		FixedSizeArray<char, LENGTH, true, 0>::push_back('\0');
	}

	FixedString(const char* s)
		: FixedSizeArray<char, LENGTH>(s, strlen(s) + 1)
	{

	}

	void resize(size_t l)
	{
		FixedSizeArray<char, LENGTH, true, 0>::resize(l);
		auto l2 = FixedSizeArray<char, LENGTH, true, 0>::length;
		if (LENGTH - l2 > 0)
			Platform::SetMemory(FixedSizeArray<char, LENGTH, true, 0>::buffer + l2, (LENGTH - l2));
	}

	operator std::string() const
	{
		return std::string(FixedSizeArray<char, LENGTH>::buffer);
	}

	const char* c_str() const
	{
		return FixedSizeArray<char, LENGTH>::buffer;
	}

};

}
