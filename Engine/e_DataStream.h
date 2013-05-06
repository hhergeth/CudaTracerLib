#pragma once

#include "..\Defines.h"
#include "cuda_runtime.h"

template<typename T> struct e_KernelDataStream
{
	T* Data;
	unsigned int UsedCount;
	unsigned int Length;
	CUDA_ONLY_FUNC T& operator[](unsigned int i) const
	{
		return Data[i];
	}
};

#include "..\Base\FileStream.h"
#include <vector>
#include <exception>
#include "e_ErrorHandler.h"

enum DataStreamRefresh
{
	DataStreamRefresh_Immediate,
	DataStreamRefresh_Buffered,
};

template<typename T> class e_DataStreamReference;
template<typename T> class e_DataStream
{
	int RoundUp(int n, int roundTo)
	{
		// fails on negative?  What does that mean?
		if (roundTo == 0) return 0;
		return ((n + roundTo - 1) / roundTo) * roundTo; // edit - fixed error
	}
private:
	T* host;
	T* device;
	unsigned int numElements;
private:
	typedef std::pair<unsigned int, unsigned int> part;
	std::vector<part> m_sDeallocated;
	unsigned int m_uPos;
	std::vector<part> m_sInvalidated;
public:
	e_DataStream(unsigned int a_Count)
	{
		numElements = a_Count;
		unsigned int si = a_Count * sizeof(T);
		cudaError_t r = cudaMalloc(&device, si);
		if(r)
			BAD_CUDA_ALLOC(si)
		host = new T[a_Count];
		m_uPos = 0;
		Invalidate(DataStreamRefresh_Buffered);
	}
	e_DataStream(InputStream& a_In)
	{
		a_In >> numElements;
		a_In >> m_uPos;
		unsigned int l;
		a_In >> l;
		if(cudaMalloc(&device, numElements * sizeof(T)))
			BAD_CUDA_ALLOC(numElements * sizeof(T))
		host = new T[numElements];
		m_sDeallocated.resize(l);
		a_In.Read(host, numElements * sizeof(T));
		if(l)
			a_In.Read(&m_sDeallocated[0], l * sizeof(part));
		Invalidate(DataStreamRefresh_Immediate);
	}
	~e_DataStream()
	{
		Free();
	}
	void Serialize(OutputStream& a_Out)
	{
		a_Out << numElements;
		a_Out << m_uPos;
		unsigned int l = m_sDeallocated.size();
		a_Out << l;
		a_Out.Write(host, numElements * sizeof(T));
		if(l)
			a_Out.Write(&m_sDeallocated[0], l * sizeof(part));
	}
	void Free()
	{
		cudaFree(device);
		delete [] host;
		m_sDeallocated.~vector();
		m_sInvalidated.~vector();
	}
	e_DataStreamReference<T> malloc(unsigned int a_Count)
	{
		for(unsigned int i = 0; i < m_sDeallocated.size(); i++)
			if(m_sDeallocated[i].second >= a_Count)
			{
				part& p = m_sDeallocated[i];
				if(p.second == a_Count)
				{
					m_sDeallocated.erase(m_sDeallocated.begin() + i);
					return e_DataStreamReference<T>(p.first, a_Count, this);
				}
				else
				{
					p.second = p.second - a_Count;
					unsigned int s = p.first;
					p.first += a_Count;
					return e_DataStreamReference<T>(s, a_Count, this);
				}
			}
		if(a_Count <= numElements - m_uPos)
		{
			m_uPos += a_Count;//RoundUp(a_Count, 16)
			return e_DataStreamReference<T>(m_uPos - a_Count, a_Count, this);
		}
		else
		{
			BAD_EXCEPTION("Cuda data stream malloc failure, %d elements requested, %d available.", a_Count, numElements - m_uPos)
			return e_DataStreamReference<T>();
		}
	}
	e_DataStreamReference<T> malloc(e_DataStreamReference<T> a_Ref)
	{
		e_DataStreamReference<T> R = malloc(a_Ref.getLength());
		cudaMemcpy(getDevice(R.getIndex()), getDevice(a_Ref.getIndex()), R.getSizeInBytes(), cudaMemcpyDeviceToDevice);
		memcpy(getHost(R.getIndex()), getHost(a_Ref.getIndex()), R.getSizeInBytes());
		return R;
	}
	void dealloc(e_DataStreamReference<T> a_Ref)
	{
		if(a_Ref.getIndex() + a_Ref.getLength() == m_uPos)
			m_uPos -= a_Ref.getLength();
		else
		{
			for(unsigned int i = 0; i < m_sDeallocated.size(); i++)
				if(m_sDeallocated[i].first == a_Ref.getIndex() + a_Ref.getLength())
				{
					m_sDeallocated[i].first = a_Ref.getIndex();
					m_sDeallocated[i].second += a_Ref.getLength();
					return;
				}
			m_sDeallocated.push_back(std::make_pair<unsigned int, unsigned int>(a_Ref.getIndex(), a_Ref.getLength()));
		}
	}
	void Invalidate(DataStreamRefresh r)
	{
		if(r == DataStreamRefresh_Immediate)
			cudaMemcpy(device, host, numElements * sizeof(T), cudaMemcpyHostToDevice);
		else
		{
			m_sInvalidated.clear();
			m_sInvalidated.push_back(std::make_pair<unsigned int, unsigned int>(0, numElements));
		}
	}
	void Invalidate(DataStreamRefresh r, e_DataStreamReference<T> a_Ref)
	{
		if(r == DataStreamRefresh_Immediate)
			cudaMemcpy(device + a_Ref.getIndex(), host + a_Ref.getIndex(), a_Ref.getLength() * sizeof(T), cudaMemcpyHostToDevice);
		else 
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
	}
	void Invalidate(DataStreamRefresh r, unsigned int i)
	{
		Invalidate(r, this->operator()(this->operator()(i)));
	}
	void UpdateInvalidated()
	{
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			int f = m_sInvalidated[i].first, l = m_sInvalidated[i].second * sizeof(T);
			cudaError_t r = cudaMemcpy(device + f, host + f, l, cudaMemcpyHostToDevice);
			if(r)
			{
				const char* c = cudaGetErrorString(r);
				OutputDebugString(c);
				OutputDebugString("\n");
			}
		}
		m_sInvalidated.clear();
	}
	void MemSetHost(unsigned int val)
	{
		memset(host, val, numElements * sizeof(T));
	}
	void MemSetDevice(unsigned int val)
	{
		cudaMemset(device, val, numElements * sizeof(T));
	}
	void CopyDeviceToHost(unsigned int start, unsigned int count)
	{
		cudaMemcpy(host + start, device + start, count * sizeof(T), cudaMemcpyDeviceToHost);
	}
	T* getHost(unsigned int i = 0)
	{
		return host + i;
	}
	T* getDevice(unsigned int i = 0)
	{
		return device + i;
	}
	T* operator()(unsigned int i)
	{
		if (i >= numElements)
			BAD_EXCEPTION("Invalid index of %d, length is %d", i, numElements)
		return host + i;
	}
	unsigned int NumUsedElements()
	{
		return m_uPos;
	}
	e_DataStreamReference<T> UsedElements()
	{
		return e_DataStreamReference<T>(0, m_uPos, this);
	}
	e_DataStreamReference<T> operator()(T* val)
	{
		return e_DataStreamReference<T>(((unsigned long long)val - (unsigned long long)host) / sizeof(T), 1, this);
	}
	unsigned int getSizeInBytes()
	{
		return numElements * sizeof(T);
	}
	unsigned int getLength()
	{
		return numElements;
	}
	e_DataStreamReference<T> translatePointer(const T* val)
	{
		unsigned long long t0 = ((unsigned long long)val - (unsigned long long)host) / sizeof(T), t1 = ((unsigned long long)val - (unsigned long long)device) / sizeof(T);
		unsigned int i = t0 < numElements ? t0 : t1;
		return e_DataStreamReference<T>(i, 1, this);
	}
	e_KernelDataStream<T> getKernelData()
	{
		e_KernelDataStream<T> r;
		r.Data = device;
		r.Length = numElements;
		r.UsedCount = m_uPos;
		return r;
	}
};

template<typename T> class e_CachedDataStream : public e_DataStream<T>
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
		const char* file;
	};
	std::vector<entry> m_sEntries;
public:
	e_CachedDataStream(unsigned int a_Count)
		: e_DataStream<T>(a_Count)
	{
		
	}
	e_DataStreamReference<T> LoadCached(const char* file, bool* load)
	{
		for(unsigned int i = 0; i < m_sEntries.size(); i++)
			if(strcmpare(m_sEntries[i].file, file))
			{
				m_sEntries[i].count++;
				*load = false;
				return e_DataStreamReference<T>(i, 1, this);
			}
		*load = true;
		entry e;
		e.count = 1;
		e.file = file;
		m_sEntries.push_back(e);
		return this->malloc(1);
	}
	void Release(char* file)
	{
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
			}
	}
};

template<typename T> class e_DataStreamReference
{
private:
	unsigned int m_uIndex;
	unsigned int m_uLength;
	e_DataStream<T>* m_pStream;
public:
	e_DataStreamReference(){m_uIndex = -1; m_uLength = -1;}
	e_DataStreamReference(unsigned int i, unsigned int c, e_DataStream<T>* s)
	{
		m_uIndex = i;
		m_uLength = c;
		m_pStream = s;
	}
	unsigned int getIndex() const
	{
		return m_uIndex;
	}
	unsigned int getLength() const
	{
		return m_uLength;
	}
	unsigned int getSizeInBytes() const
	{
		return m_uLength * sizeof(T);
	}
	T* operator()(unsigned int i = 0) const
	{
		return m_pStream[0](m_uIndex + i);
	}
	template<typename U> U* operator()(unsigned int i = 0) const
	{
		return (U*)m_pStream[0](m_uIndex + i);
	}
	T* getDevice(unsigned int i = 0) const
	{
		return m_pStream->getDevice(m_uIndex + i);
	}
	T* getHost(unsigned int i = 0) const
	{
		return m_pStream->getHost(m_uIndex + i);
	}
	void Invalidate() const
	{
		m_pStream->Invalidate(DataStreamRefresh_Buffered, *this);
	}
};

template<typename T> struct REF
{
	typedef e_DataStreamReference<T> type;
};