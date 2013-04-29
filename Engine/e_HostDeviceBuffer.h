#pragma once

#include "..\Defines.h"
#include "..\Math\vector.h"

enum HostDeviceBufferRefresh
{
	HostDeviceBufferRefresh_Immediate,
	HostDeviceBufferRefresh_Buffered,
};

template<typename T> struct e_KernelHostDeviceBuffer
{
	T* Data;
	unsigned int UsedCount;
	unsigned int Length;
	CUDA_ONLY_FUNC T& operator[](unsigned int i) const
	{
		return Data[i];
	}
};


template<typename H, typename D> class e_HostDeviceBufferReference;
template<typename H, typename D> class e_HostDeviceBuffer
{
	H* host;
	D* device;
	D* deviceMapped;
	unsigned int m_uPos;
	unsigned int m_uLength;
	unsigned int m_uHostBlockSize;
private:
	typedef std::pair<unsigned int, unsigned int> part;
	std::vector<part> m_sInvalidated;
public:
	e_HostDeviceBuffer(unsigned int a_NumElements, unsigned int a_ElementSize = -1)
	{
		if(a_ElementSize == -1)
			a_ElementSize = sizeof(H);
		m_uHostBlockSize = a_ElementSize;
		m_uPos = 0;
		m_uLength = a_NumElements;
		host = (H*)::malloc(m_uHostBlockSize * a_NumElements);
		cudaMalloc(&device, sizeof(D) * a_NumElements);
		cudaMemset(device, 0, sizeof(D) * a_NumElements);
		deviceMapped = new D[a_NumElements];
		memset(deviceMapped, 0, sizeof(D) * a_NumElements);
	}
	e_HostDeviceBufferReference<H, D> malloc(int a_Num)
	{
		if(m_uPos + a_Num >= m_uLength)
			throw 1;
		unsigned int i = m_uPos;
		m_uPos += a_Num;
		return e_HostDeviceBufferReference<H, D>(i, a_Num, this);
	}
	void Free()
	{
		free(host);
		cudaFree(device);
	}
	void Invalidate()
	{
		m_sInvalidated.clear();
		m_sInvalidated.push_back(std::make_pair<unsigned int, unsigned int>(0, numElements));
	}
	void Invalidate(e_HostDeviceBufferReference<H, D> a_Ref)
	{	
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			if(m_sInvalidated[i].first <= a_Ref.getIndex() && m_sInvalidated[i].second + m_sInvalidated[i].first >= a_Ref.getIndex() + a_Ref.getLength())
				return;
			else if(m_sInvalidated[i].first <= a_Ref.getIndex() && m_sInvalidated[i].first + m_sInvalidated[i].second >= a_Ref.getIndex())
			{
				m_sInvalidated[i].second = a_Ref.getIndex() + a_Ref.getLength() - m_sInvalidated[i].first;
				return;
			}
		}
		m_sInvalidated.push_back(std::make_pair<unsigned int, unsigned int>(a_Ref.getIndex(), a_Ref.getLength()));
	}
	void UpdateInvalidated()
	{
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			for(unsigned int j = 0; j < m_sInvalidated[i].second; j++)
			{
				unsigned int k = j + m_sInvalidated[i].first;
				deviceMapped[k] = getHost(k)->getKernelData();
			}
		}
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			int q = sizeof(D), f = m_sInvalidated[i].first, n = q * m_sInvalidated[i].second;
			cudaMemcpy(device + f, deviceMapped + f, n, cudaMemcpyHostToDevice);
		}
		m_sInvalidated.clear();
	}
	H* getHost(unsigned int i = 0)
	{
		return (H*)((char*)host + i * m_uHostBlockSize);
	}
	D* getDevice(unsigned int i = 0)
	{
		return device + i;
	}
	unsigned int getLength()
	{
		return m_uLength;
	}
	H* operator()(unsigned int i)
	{
		if (i >= m_uLength)
			BAD_EXCEPTION("Invalid index of %d, length is %d", i, m_uLength)
		return getHost(i);
	}
	e_HostDeviceBufferReference<H, D> operator()(H* val)
	{
		return e_HostDeviceBufferReference<H, D>(((unsigned long long)val - (unsigned long long)host) / m_uHostBlockSize, 1, this);
	}
	unsigned int UsedElements()
	{
		return m_uPos;
	}
	unsigned int getSizeInBytes()
	{
		return m_uLength * sizeof(D);
	}
	e_KernelHostDeviceBuffer<D> getKernelData()
	{
		e_KernelHostDeviceBuffer<D> r;
		r.Data = device;
		r.Length = m_uLength;
		r.UsedCount = m_uPos;
		return r;
	}
};

template<typename H, typename D> class e_HostDeviceBufferReference
{
private:
	unsigned int m_uIndex;
	unsigned int m_uLength;
	e_HostDeviceBuffer<H, D>* m_pStream;
public:
	e_HostDeviceBufferReference(){m_uIndex = -1; m_uLength = -1;}
	e_HostDeviceBufferReference(unsigned int i, unsigned int c, e_HostDeviceBuffer<H, D>* s)
	{
		m_uIndex = i;
		m_uLength = c;
		m_pStream = s;
	}
	unsigned int getIndex()
	{
		return m_uIndex;
	}
	unsigned int getLength()
	{
		return m_uLength;
	}
	unsigned int getSizeInBytes()
	{
		return m_uLength * sizeof(H);
	}
	H* operator()(unsigned int i = 0)
	{
		return m_pStream[0](m_uIndex + i);
	}
	template<typename U> U* operator()(unsigned int i = 0)
	{
		return (U*)m_pStream[0](m_uIndex + i);
	}
	void Invalidate()
	{
		m_pStream->Invalidate(*this);
	}
	D* getDevice(unsigned int i = 0)
	{
		return m_pStream->getDevice(m_uIndex + i);
	}
	H* getHost(unsigned int i = 0)
	{
		return m_pStream->getHost(m_uIndex + i);
	}
};

template<typename H, typename D> class e_CachedHostDeviceBuffer : public e_HostDeviceBuffer<H, D>
{
	inline bool strcmpare(const char* a, const char* b)
	{
		int al = strlen(a), bl = strlen(b);
		if(al != bl)
			return false;
		else for(int i = 0; i < al; i++)
			if(a[i] != b[i])
				return false;
		return true;
	}
private:
	struct entry
	{
		unsigned int count;
		char file[256];
	};
	std::vector<entry> m_sEntries;
public:
	e_CachedHostDeviceBuffer(unsigned int a_Count, unsigned int a_ElementSize = -1)
		: e_HostDeviceBuffer<H, D>(a_Count, a_ElementSize)
	{
		
	}
	e_HostDeviceBufferReference<H, D> LoadCached(const char* file, bool* load)
	{
		for(unsigned int i = 0; i < m_sEntries.size(); i++)
			if(strcmpare(m_sEntries[i].file, file))
			{
				m_sEntries[i].count++;
				*load = false;
				return e_HostDeviceBufferReference<H, D>(i, 1, this);
			}
		*load = true;
		entry e;
		e.count = 1;
		ZeroMemory(e.file, sizeof(e.file));
		memcpy(e.file, file, strlen(file));
		m_sEntries.push_back(e);
		return this->malloc(1);
	}
	void Release(char* file)
	{/*
		for(unsigned int i = 0; i < m_sEntries.size(); i++)
			if(strcmp(m_sEntries[i].file, file))
			{
				m_sEntries[i].count--;
				if(!m_sEntries[i].count)
				{
					this->dealloc(e_DataStreamReference<T>(i, 1, this));
					m_sData.erase(i, i);
				}
				break;
			}*/
	}
};
