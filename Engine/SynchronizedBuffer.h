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
	virtual void Free() = 0;
	CUDA_FUNC_IN bool isOnCPU() const
	{
		return (m_location & DataLocation::CPU) == DataLocation::CPU;
	}
	CUDA_FUNC_IN bool isOnGPU() const
	{
		return (m_location & DataLocation::GPU) == DataLocation::GPU;
	}
	virtual void Synchronize() = 0;
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
		CUDA_MEMCPY_TO_DEVICE(m_deviceData, m_hostData, m_length * sizeof(T));
	}
	virtual void Free() override
	{
		if (m_hostData == 0 || m_deviceData == 0)
			throw std::runtime_error("Invalid call to Free!");
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
		CTL_ASSERT(isOnGPU());
		return m_deviceData[idx];
#else
		CTL_ASSERT(isOnCPU());
		return m_hostData[idx];
#endif
	}
	CUDA_FUNC_IN T& operator[](unsigned int idx)
	{
#ifdef ISCUDA
		CTL_ASSERT(isOnGPU());
		return m_deviceData[idx];
#else
		CTL_ASSERT(isOnCPU());
		return m_hostData[idx];
#endif
	}
	CUDA_FUNC_IN const T* getDevicePtr() const
	{
		return m_deviceData;
	}
	CUDA_FUNC_IN T* getDevicePtr()
	{
		return m_deviceData;
	}
	CUDA_FUNC_IN unsigned int getLength() const
	{
		return m_length;
	}
	virtual void Synchronize() override
	{
		if (m_location == DataLocation::CPU)
			CUDA_MEMCPY_TO_DEVICE(m_deviceData, m_hostData, m_length * sizeof(T));
		else if (m_location == DataLocation::GPU)
			CUDA_MEMCPY_TO_HOST(m_hostData, m_deviceData, m_length * sizeof(T));
		m_location = DataLocation::Synchronized;
	}
	virtual void Memset(unsigned char val)
	{
		Platform::SetMemory(m_hostData, m_length * sizeof(T), val);
		cudaMemset(m_deviceData, val, m_length * sizeof(T));
		m_location = DataLocation::Synchronized;
	}
	virtual void Memset(const T& val)
	{
		for (unsigned int i = 0; i < m_length; i++)
			m_hostData[i] = val;
		m_location = DataLocation::CPU;
		Synchronize();
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
	std::vector<ISynchronizedBuffer*> m_buffers;
	void iterateTypes()
	{
		
	}
	template<typename ARG, typename... ARGS> void iterateTypes(ARG& arg, ARGS&... args)
	{
		iterateTypes(arg);
		iterateTypes(args...);
	}
	template<typename ARG> void iterateTypes(ARG& arg)
	{
		m_buffers.push_back(&arg);
	}
public:
	template<typename... ARGS> ISynchronizedBufferParent(ARGS&... args)
	{
		iterateTypes(args...);
	}
	virtual void Free() override
	{
		for (auto buf : m_buffers)
			buf->Free();
	}
	virtual void Synchronize() override
	{
		for (auto buf : m_buffers)
			buf->Synchronize();
		m_location = DataLocation::Synchronized;
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