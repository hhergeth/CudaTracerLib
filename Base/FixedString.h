#pragma once

#include <string>
#include "Platform.h"

template<int LENGTH> struct FixedString
{
	FixedString()
	{
		length = 0;
		Platform::SetMemory(buffer, LENGTH);
	}
	FixedString(const std::string& s)
	{
		if (s.size() > LENGTH)
			throw std::runtime_error("Fixed string buffer not long enough!");
		Platform::SetMemory(buffer, LENGTH);
		memcpy(buffer, s.c_str(), s.size());
		length = (unsigned short)s.size();
	}
	FixedString(const char* s)
	{
		Platform::SetMemory(buffer, LENGTH);
		length = (unsigned short)strlen(s);
		if (length > LENGTH)
			throw std::runtime_error("Fixed string buffer not long enough!");
		memcpy(buffer, s, length);
	}
	operator std::string() const
	{
		std::string str(buffer, buffer + length);
		return str;
	}
	size_t size() const
	{
		return length;
	}
	const char* c_str() const
	{
		return buffer;
	}
	template <int N2>
	FixedString<LENGTH>& operator=(const FixedString<N2>& other)
	{
		if (c_str() != other.c_str())
		{
			if (other.size() > LENGTH)
				throw std::runtime_error("Invalid fixed string assignment!");
			length = (unsigned short)other.size();
			memcpy(buffer, other.c_str(), length);
			Platform::SetMemory(buffer + length, LENGTH - length);
		}
		return *this;
	}
private:
	unsigned short length;
	char buffer[LENGTH];
};