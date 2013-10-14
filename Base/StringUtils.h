#pragma once
#include <string>
#include <cstdarg>
#include <vector>
#include <string>
#include <algorithm>

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
	return std::string(dir);
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