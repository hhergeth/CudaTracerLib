#pragma once

#include "FixedSizeArray.h"

template<int LENGTH> struct FixedString : public FixedSizeArray<char, LENGTH>
{
	FixedString()
		: FixedSizeArray<char, LENGTH>()
	{
	}

	FixedString(const std::string& s)
		: FixedSizeArray<char, LENGTH>(s.c_str(), s.length())
	{
	}

	FixedString(const char* s)
		: FixedSizeArray<char, LENGTH>(s, strlen(s))
	{

	}

	operator std::string() const
	{
		std::string str(buffer, buffer + length);
		return str;
	}

	const char* c_str() const
	{
		return buffer;
	}
	
};