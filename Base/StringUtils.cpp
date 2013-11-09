#include <StdAfx.h>
#include "StringUtils.h"

std::string &trim(std::string &s)
{/*
	int n = 0, e = s.size() - 1;
	while(n < s.size() && (s[n] == ' ' || s[n] == '\t'))
		n++;
	while(e >= n && (s[e] == ' ' || s[e] == '\t'))
		e--;
	if(n < e)
		return s.substr(n, e - n);
	else return std::string();*/return ltrim(rtrim(s));

	std::string str = s;
std::string::size_type pos = std::min(str.find_last_not_of(' '), str.find_last_not_of('\t'));
  if(pos != std::string::npos) {
    str.erase(pos + 1);
    pos = std::max(str.find_first_not_of(' '), str.find_first_not_of('\t'));
    if(pos != std::string::npos) str.erase(0, pos);
  }
  else str.erase(str.begin(), str.end());
  return str;
}