#pragma once

#include <functional>
#include <string>

namespace CudaTracerLib
{

#define MACROSTR(k) std::string(#k),
#define MACROID(k) k,

template<typename E> struct EnumConverter
{

};

#define ENUMIZE(NAME, A) \
enum NAME { \
	A(MACROID) \
}; \
template<> struct EnumConverter<NAME> \
{ \
  static const std::string& ToString(NAME val) \
  { \
	static std::string ARR[] = { A(MACROSTR) }; \
	return ARR[(int)val];\
  } \
  static NAME FromString(const std::string& val2) \
  { \
	static std::string ARR[] = { A(MACROSTR) }; \
	for(int i = 0; i < sizeof(ARR)/sizeof(ARR[0]); i++) if(ARR[i] == val2) return (NAME)i; \
	throw 1; \
  }  \
  static void enumerateEntries(const std::function<void(NAME, const std::string&)>& f) \
  { \
	static std::string ARR[] = { A(MACROSTR) }; \
	for(int i = 0; i < sizeof(ARR)/sizeof(ARR[0]); i++) \
		f((NAME)i, ARR[i]); \
  } \
};

}