#pragma once

#include <CudaMemoryManager.h>
#include <vector>

namespace CudaTracerLib {

enum DataLocation
{
	Invalid = 0,
	GPU = 1,
	CPU = 2,
	Synchronized = GPU | CPU,
};

class ISynchronizedBuffer
{
protected:
	DataLocation m_location;
public:
	ISynchronizedBuffer()
		: m_location(DataLocation::Synchronized)
	{
		
	}
	virtual ~ISynchronizedBuffer()
	{
		
	}
	CUDA_FUNC_IN bool isOnCPU() const
	{
		return (m_location & DataLocation::CPU) == DataLocation::CPU;
	}
	CUDA_FUNC_IN bool isOnGPU() const
	{
		return (m_location & DataLocation::GPU) == DataLocation::GPU;
	}
	virtual void Synchronize() const = 0;
	virtual void setOnCPU()
	{
		m_location = DataLocation::CPU;
	}
	virtual void setOnGPU()
	{
		m_location = DataLocation::GPU;
	}
};

template<typename T> class SynchronizedBuffer : public ISynchronizedBuffer
{
	unsigned int m_length;
	T* m_hostData;
	T* m_deviceData;
public:
	SynchronizedBuffer(unsigned int length)
		: m_length(length)
	{
		m_hostData = new T[m_length];
		CUDA_MALLOC(&m_deviceData, m_length * sizeof(T));
	}
	virtual ~SynchronizedBuffer()
	{
		if (m_hostData == 0 || m_deviceData == 0)
			throw std::runtime_error("Invalid call to destructor!");
		delete[] m_hostData;
		CUDA_FREE(m_deviceData);
		m_length = 0;
		m_hostData = m_deviceData = 0;
	}
	virtual void Resize(unsigned int newLength)
	{
		Synchronize();
		CUDA_FREE(m_deviceData);
		CUDA_MALLOC(&m_deviceData, newLength * sizeof(T));
		auto l = Dmin2(newLength, m_length);
		CUDA_MEMCPY_TO_DEVICE(m_deviceData, m_hostData, l * sizeof(T));
		m_location = DataLocation::Synchronized;
		T* newHostData = new T[newLength];
		memcpy(newHostData, m_hostData, l * sizeof(T));
		delete[] m_hostData;
		m_hostData = newHostData;
		m_length = newLength;
	}
	CUDA_FUNC_IN const T& operator[](unsigned int idx) const
	{
#ifdef ISCUDA
		return m_deviceData[idx];
#else
		return m_hostData[idx];
#endif
	}
	CUDA_FUNC_IN T& operator[](unsigned int idx)
	{
#ifdef ISCUDA
		return m_deviceData[idx];
#else
		return m_hostData[idx];
#endif
	}
	CUDA_FUNC_IN unsigned int getLength() const
	{
		return m_length;
	}
	virtual void Synchronize() const override
	{
		if (m_location == DataLocation::CPU)
			CUDA_MEMCPY_TO_DEVICE(m_deviceData, m_hostData, m_length * sizeof(T));
		else if (m_location == DataLocation::GPU)
			CUDA_MEMCPY_TO_HOST(m_hostData, m_deviceData, m_length * sizeof(T));
	}
};

//out of class for ease of use
template<typename T> static void CreateOrResize(SynchronizedBuffer<T>*& buf, unsigned int newLength)
{
	if (buf)
		buf->Resize(newLength);
	else buf = new SynchronizedBuffer<T>(newLength);
}

class ISynchronizedBufferParent : public ISynchronizedBuffer
{
	template<typename T> static void checkType(const SynchronizedBuffer<T>& arg)
	{

	}
	template<typename T> static void checkType(const T& arg)
	{
		static_assert(false, "Type passed to ISynchronizedBufferParent is not derived from SynchronizedBuffer!");
	}

	std::vector<ISynchronizedBuffer*> m_buffers;
	template<typename ARG, typename... ARGS> void iterateTypes(ARG& arg, ARGS&... args)
	{
		iterateTypes(arg);
		iterateTypes(args...);
	}
	template<typename ARG> void iterateTypes(ARG& arg)
	{
		checkType(arg);
		m_buffers.push_back(&arg);
	}
public:
	template<typename... ARGS> ISynchronizedBufferParent(ARGS&... args)
	{
		iterateTypes(args...);
	}
	virtual void Synchronize() const override
	{
		for (auto buf : m_buffers)
			buf->Synchronize();
	}
	virtual void setOnCPU() override
	{
		ISynchronizedBuffer::setOnCPU();
		for (auto buf : m_buffers)
			buf->setOnCPU();
	}
	virtual void setOnGPU() override
	{
		ISynchronizedBuffer::setOnGPU();
		for (auto buf : m_buffers)
			buf->setOnGPU();
	}
};

}