#pragma once

#include <stdexcept>

#if defined(ISCUDA)
	#define BAD_EXCEPTION(...) { }
#else
	#define BAD_EXCEPTION(...) \
		{ \
			char* buf = new char[256]; \
			sprintf(buf, ##__VA_ARGS__); \
			std::cout << buf; \
			throw std::runtime_error(buf); \
		}
#endif

#define BAD_CUDA_ALLOC(_SIZE_) BAD_EXCEPTION("Cuda memory allocation(%lld bytes) failure.", long long int(_SIZE_))

#define BAD_HOST_ALLOC(_SIZE_) BAD_EXCEPTION("Memory allocation(%lld bytes) failure.", long long int(_SIZE_))

#define BAD_HOST_DEVICE_COPY(_START_, _SIZE_) BAD_EXCEPTION("Host to device memcpy failure, start at : %lld, size : %lld", long long int(_START_), long long int(_SIZE_))

#define BAD_DEVICE_HOST_COPY(_START_, _SIZE_) BAD_EXCEPTION("Device to host memcpy failure, start at : %lld, size : %lld", long long int(_START_), long long int(_SIZE_))

#define CHECK_CUDA_CODE(_ERR_, _MES_) if(_ERR_ != CUDA_SUCCESS) BAD_EXCEPTION("%s \nCuda Message : %s", _MES_, cudaGetErrorString(_ERR_))

#define CHECK_BAD_CUDA_CALL(_MES_) CHECK_CUDA_CODE(cudaGetLastError(), "Cuda error.")

#define CHECK_BAD_CUDA_CALLD CHECK_BAD_CUDA_CALL("Cuda error.")

#define SYNC_BAD_CUDA_CALL(_MES_) CHECK_CUDA_CODE(cudaDeviceSynchronize(), _MES_)

#define SYNC_BAD_CUDA_CALLD SYNC_BAD_CUDA_CALL("Cuda error.")