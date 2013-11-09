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
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 1024;
    char stackbuf[1024];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];

    while (1) {
        // Try to vsnprintf into our buffer.
        int needed = vsnprintf (buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0) {
            // It fit fine so we're done.
            return std::string (buf, (size_t) needed);
        }

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize (size);
        buf = &dynamicbuf[0];
    }
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
	_splitpath(s.c_str(), drive, dir, fname, ext); 
	return std::string(fname);
}

inline std::string getDirName(const std::string& s)
{
	char drive[_MAX_DRIVE]; 
	char dir[_MAX_DIR]; 
	char fname[_MAX_FNAME]; 
	char ext[_MAX_EXT];
	_splitpath(s.c_str(), drive, dir, fname, ext); 
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

std::string &trim(std::string &s);

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
        if (parseLiteral(ptr, "#INF"))
        {
            value = bitsToFloat((neg) ? 0xFF800000 : 0x7F800000);
            return true;
        }
        if (parseLiteral(ptr, "#SNAN"))
        {
            value = bitsToFloat((neg) ? 0xFF800001 : 0x7F800001);
            return true;
        }
        if (parseLiteral(ptr, "#QNAN"))
        {
            value = bitsToFloat((neg) ? 0xFFC00001 : 0x7FC00001);
            return true;
        }
        if (parseLiteral(ptr, "#IND"))
        {
            value = bitsToFloat((neg) ? 0xFFC00000 : 0x7FC00000);
            return true;
        }
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