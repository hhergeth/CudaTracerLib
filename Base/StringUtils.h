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
	int l = _vscprintf(fmt, ap);
	char* c = new char[l];
	vsnprintf_s(c, l, l, fmt, ap);
	return std::string(c);
}

inline std::string format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

inline std::string getFileName(const std::string& s)
{
	char drive[_MAX_DRIVE]; 
	char dir[_MAX_DIR]; 
	char fname[_MAX_FNAME]; 
	char ext[_MAX_EXT];
	_splitpath_s(s.c_str(), drive, dir, fname, ext); 
	return std::string(fname);
}

inline std::string getDirName(const std::string& s)
{
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	_splitpath_s(s.c_str(), drive, dir, fname, ext); 
	return std::string(drive) + std::string(dir);
}

inline void toLower(std::string& s)
{
	std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

inline std::string toLower(const std::string& s)
{
	std::string r = s;
	toLower(r);
	return r;
}

inline bool endsWith (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

inline bool startsWith(std::string const &fullString, std::string const &prefix)
{
	return fullString.substr(0, prefix.size()) == prefix;
}

inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

inline std::string& trim(std::string &s)
{/*
	int n = 0, e = s.size() - 1;
	while(n < s.size() && (s[n] == ' ' || s[n] == '\t'))
		n++;
	while(e >= n && (s[e] == ' ' || s[e] == '\t'))
		e--;
	if(n < e)
		return s.substr(n, e - n);
	else return std::string();*/
	
	return ltrim(rtrim(s));
	/*
	std::string str = s;
	std::string::size_type pos = std::min(str.find_last_not_of(' '), str.find_last_not_of('\t'));
	if(pos != std::string::npos)
	{
		str.erase(pos + 1);
		pos = std::max(str.find_first_not_of(' '), str.find_first_not_of('\t'));
		if(pos != std::string::npos)
			str.erase(0, pos);
	}
	else str.erase(str.begin(), str.end());
	return str;*/
}

inline bool parseSpace(const char*& ptr)
{
    while (*ptr == ' ' || *ptr == '\t')
        ptr++;
    return true;
}

inline bool parseChar(const char*& ptr, char chr)
{
    if (*ptr != chr)
        return false;
    ptr++;
    return true;
}

inline bool parseLiteral(const char*& ptr, const char* str)
{
    const char* tmp = ptr;

    while (*str && *tmp == *str)
    {
        tmp++;
        str++;
    }
    if (*str)
        return false;

    ptr = tmp;
    return true;
}

inline bool parseInt(const char*& ptr, int& value)
{
    const char* tmp = ptr;
    int v = 0;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
    if (*tmp < '0' || *tmp > '9')
        return false;
    while (*tmp >= '0' && *tmp <= '9')
        v = v * 10 + *tmp++ - '0';

    value = (neg) ? -v : v;
    ptr = tmp;
    return true;
}

inline bool parseInt(const char*& ptr, long long& value)
{
    const char* tmp = ptr;
    long long v = 0;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
    if (*tmp < '0' || *tmp > '9')
        return false;
    while (*tmp >= '0' && *tmp <= '9')
        v = v * 10 + *tmp++ - '0';

    value = (neg) ? -v : v;
    ptr = tmp;
    return true;
}

inline bool parseHex(const char*& ptr, unsigned int& value)
{
    const char* tmp = ptr;
    unsigned int v = 0;
    for (;;)
    {
        if (*tmp >= '0' && *tmp <= '9')         v = v * 16 + *tmp++ - '0';
        else if (*tmp >= 'A' && *tmp <= 'F')    v = v * 16 + *tmp++ - 'A' + 10;
        else if (*tmp >= 'a' && *tmp <= 'f')    v = v * 16 + *tmp++ - 'a' + 10;
        else                                    break;
    }

    if (tmp == ptr)
        return false;

    value = v;
    ptr = tmp;
    return true;
}

inline bool parseFloat(const char*& ptr, float& value)
{
#define bitsToFloat(x) (*(float*)&x)
    const char* tmp = ptr;
    bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));

    float v = 0.0f;
    int numDigits = 0;
    while (*tmp >= '0' && *tmp <= '9')
    {
        v = v * 10.0f + (float)(*tmp++ - '0');
        numDigits++;
    }
    if (parseChar(tmp, '.'))
    {
        float scale = 1.0f;
        while (*tmp >= '0' && *tmp <= '9')
        {
            scale *= 0.1f;
            v += scale * (float)(*tmp++ - '0');
            numDigits++;
        }
    }
    if (!numDigits)
        return false;

    ptr = tmp;
    if (*ptr == '#')
    {
		unsigned int v = 0;
		if(parseLiteral(ptr, "#INF"))
			v = 0x7F800000;
		else if(parseLiteral(ptr, "#SNAN"))
			v = 0xFF800001;
		else if(parseLiteral(ptr, "#QNAN"))
			v = 0xFFC00001;
		else if(parseLiteral(ptr, "#IND"))
			v = 0xFFC00000;
		if(v)
		{
			v |= neg << 31;
			value = *(float*)&v;
			return true;
		}
		else return false;
    }

    int e = 0;
    if ((parseChar(tmp, 'e') || parseChar(tmp, 'E')) && parseInt(tmp, e))
    {
        ptr = tmp;
        if (e)
            v *= pow(10.0f, (float)e);
    }
    value = (neg) ? -v : v;
    return true;
#undef bitsToFloat
}