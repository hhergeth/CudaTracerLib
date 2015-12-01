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
		: FixedSizeArray<char, LENGTH>(s.c_str(), s.length() + 1)
	{
		FixedSizeArray<char, LENGTH>::buffer[FixedSizeArray<char, LENGTH>::length - 1] = 0;
	}

	FixedString(const char* s)
		: FixedSizeArray<char, LENGTH>(s, strlen(s) + 1)
	{

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
