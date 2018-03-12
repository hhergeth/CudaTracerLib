#pragma once
#include <stdexcept>
#include <Defines.h>
#include <string>
#include <memory>
#include <algorithm>
#include <cctype>
#include <locale>

namespace CudaTracerLib {

class Platform
{
public:
	//mimics cuda atomic functions as atomicInc, atomicAdd, atomicExch, atomicAdd

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Increment(unsigned int* add);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Add(unsigned int* add, unsigned int val);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static unsigned int Exchange(unsigned int* add, unsigned int val);

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float Add(float* add, float val);

	//standard x86 memset which uses a 8 bit as value
	CTL_EXPORT CUDA_HOST static void SetMemory(void* dest, size_t length, unsigned char val = 0);
	//memset which uses 32 bit as value
	CTL_EXPORT CUDA_HOST static void SetMemoryExt(void* dest, size_t length, unsigned int val = 0);
	CTL_EXPORT CUDA_HOST static void OutputDebug(const std::string& msg);
};

#define ZERO_MEM(ref) Platform::SetMemory(&ref, sizeof(ref), 0)

inline std::string to_lower(const std::string& _data)
{
	std::string data = _data;
	std::transform(data.begin(), data.end(), data.begin(), ::tolower);
	return data;
}

//https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c/2072890#2072890
inline bool ends_with(const std::string& value, const std::string& ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool starts_with(const std::string& value, const std::string& beginning)
{
	if (beginning.size() > value.size()) return false;
	return std::equal(beginning.begin(), beginning.end(), value.begin());
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
		return !std::isspace(ch);
	}));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
		return !std::isspace(ch);
	}).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
	ltrim(s);
	rtrim(s);
}

}