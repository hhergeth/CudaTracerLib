#pragma once
#include <string>
#include <cstdarg>
#include <vector>
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>

inline std::string vformat (const char *fmt, va_list ap)
{
	int l = _vscprintf(fmt, ap) + 1;
	std::string str;
	str.resize(l);
	vsprintf_s((char*)str.c_str(), l, fmt, ap);
	return str;
}

inline std::string format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}