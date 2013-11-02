#include <StdAfx.h>
#include <vector>
#include "Engine\e_ErrorHandler.h"
#include "Base\FileStream.h"

template<typename T> struct e_KernelBuffer
{
	T* Data;
	unsigned int UsedCount;
	unsigned int Length;
	CUDA_FUNC_IN T& operator[](unsigned int i) const
	{
		return Data[i];
	}
};

template<typename H, typename D> class e_BufferReference;
template<typename H, typename D> class e_BufferIterator;
template<typename H, typename D> class e_Buffer
{
	template<typename A, typename B> struct templateEquality
	{
		bool operator()()
		{
			return false;
		}
	};
	template<typename A> struct templateEquality<A, A>
	{
		bool operator()()
		{
			return true;
		}
	};

	friend e_BufferReference<H, D>;
protected:
	H* host;
	D* device;
	D* deviceMapped;
	unsigned int m_uPos;
	unsigned int m_uLength;
	unsigned int m_uHostBlockSize;
	typedef std::pair<unsigned int, unsigned int> range;
	std::vector<range> m_sInvalidated;//indices into the total buffer, not the entries!
	struct entry
	{
		unsigned int start;
		unsigned int end;
	};
	//entries ordered by start!
	std::vector<entry> m_sEntries;
	//convenience function which returns H == D
	bool isEqual()
	{
		return templateEquality()();
	}
public:
	e_Buffer(unsigned int a_NumElements, unsigned int a_ElementSize = 0xffffffff)
	{
		if(a_ElementSize == 0xffffffff)
			a_ElementSize = sizeof(H);
		m_uHostBlockSize = a_ElementSize;
		m_uPos = 0;
		m_uLength = a_NumElements;
		host = (H*)::malloc(m_uHostBlockSize * a_NumElements);
		cudaMalloc(&device, sizeof(D) * a_NumElements);
		cudaMemset(device, 0, sizeof(D) * a_NumElements);
		if(!isEqual())
			deviceMapped = (D*)::malloc(a_NumElements * sizeof(D));
		memset(deviceMapped, 0, sizeof(D) * a_NumElements);
	}
	void Free()
	{
		free(host);
		if(deviceMapped)
			free(deviceMapped);
		cudaFree(device);
	}
	e_Buffer(InputStream& a_In)
	{
		a_In >> m_uLength;
		a_In >> m_uPos;
		a_In >> m_uHostBlockSize;
		if(cudaMalloc(&device, sizeof(D) * m_uLength))
			BAD_CUDA_ALLOC(sizeof(D) * m_uLength)
		host = (H*)::malloc(m_uHostBlockSize * m_uLength);
		a_In.Read(host, m_uPos * m_uHostBlockSize);
		if(!isEqual())
			deviceMapped = (D*)::malloc(a_NumElements * sizeof(D));
		unsigned int Q;
		a_In >> Q;
		m_sEntries.resize(deN);
		entry e;
		for(int i = 0; i < Q; i++)
		{
			a_In >> e;
			m_sEntries.push_back(e);
		}
		Invalidate();
	}
	void Serialize(OutputStream& a_Out) const
	{
		a_Out << m_uLength;
		a_Out << m_uPos;
		a_Out << m_uHostBlockSize;
		a_Out.Write(host, m_uPos * m_uHostBlockSize);
		a_Out << (unsigned int)m_sEntries.size();
		if(m_sEntries.size())
			a_Out.Write(&m_sEntries[0], m_sEntries.size() * sizeof(entry));
	}
	e_BufferReference<H, D> malloc(unsigned int a_Count)
	{
		for(unsigned int i = 0; i < m_sDeallocated.size(); i++)
			if(m_sDeallocated[i].second >= a_Count)
			{
				part& p = m_sDeallocated[i];
				if(p.second == a_Count)
				{
					m_sDeallocated.erase(m_sDeallocated.begin() + i);
					return e_BufferReference<H, D>(this, p.first, a_Count);
				}
				else
				{
					p.second = p.second - a_Count;
					unsigned int s = p.first;
					p.first += a_Count;
					return e_BufferReference<H, D>(this, s, a_Count);
				}
			}
		if(a_Count <= m_uLength - m_uPos)
		{
			m_uPos += a_Count;//RoundUp(a_Count, 16)
			return e_BufferReference<H, D>(this, m_uPos - a_Count, a_Count);
		}
		else
		{
			BAD_EXCEPTION("Cuda data stream malloc failure, %d elements requested, %d available.", a_Count, m_uLength - m_uPos)
			return e_BufferReference<H, D>();
		}
	}
	e_BufferReference<H, D> malloc(e_BufferReference<H, D> r)
	{
		return malloc(r.getLength());
	}
	void dealloc(e_BufferReference<H, D> a_Ref)
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
			m_sDeallocated.push_back(std::make_pair(a_Ref.getIndex(), a_Ref.getLength()));
		}
	}
	void dealloc(unsigned int i, unsigned int j = 1)
	{
		dealloc(operator()(i, j));
	}
	void Invalidate()
	{
		m_sInvalidated.clear();
		m_sInvalidated.push_back(std::make_pair(0U, m_uPos));
	}
	void Invalidate(e_BufferReference<H, D> a_Ref)
	{	
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			//a_Ref in m_sInvalidated[i]
			if(m_sInvalidated[i].first <= a_Ref.getIndex() && m_sInvalidated[i].second + m_sInvalidated[i].first >= a_Ref.getIndex() + a_Ref.getLength())
				return;
			//extend m_sInvalidated[i] to the end of a_Ref
			else if(m_sInvalidated[i].first <= a_Ref.getIndex() && m_sInvalidated[i].first + m_sInvalidated[i].second >= a_Ref.getIndex())
			{
				m_sInvalidated[i].second = a_Ref.getIndex() + a_Ref.getLength() - m_sInvalidated[i].first;
				return;
			}
		}
		m_sInvalidated.push_back(std::make_pair(a_Ref.getIndex(), a_Ref.getLength()));
	}
	void Invalidate(unsigned int i, unsigned int j = 1)
	{
		Invalidate(operator()(i, j));
	}
	void UpdateInvalidated()
	{
		struct id
		{
			operator()(e_BufferReference<H, D> ref)
			{
			}
		};
		UpdateInvalidatedCB(id());
	}
	template<typename CBT> void UpdateInvalidatedCB(CBT& f)
	{
		if(!isEqual())
			for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
			{
				for(unsigned int j = 0; j < m_sInvalidated[i].second; j++)
				{
					unsigned int k = j + m_sInvalidated[i].first;
					f(operator()(k));//do that BEFORE, so the user has a chance to modify the data which will be copied...
					deviceMapped[k] = operator()(k)->getKernelData();
				}
			}
		for(unsigned int i = 0; i < m_sInvalidated.size(); i++)
		{
			int q = sizeof(D), f = m_sInvalidated[i].first, n = q * m_sInvalidated[i].second;
			cudaMemcpy(device + f, (isEqual() ? host : deviceMapped) + f, n, cudaMemcpyHostToDevice);
		}
		m_sInvalidated.clear();
	}
	float getPercenateUsed() const
	{
		return (float)m_uPos / (float)m_uLength;
	}
	/*unsigned int getLength() const
	{
		return m_uLength;
	}
	unsigned int NumUsedElements() const
	{
		return m_uPos;
	}*/
	unsigned int getSizeInBytes() const
	{
		return m_uLength * sizeof(D);
	}
	e_KernelBuffer<D> getKernelData(bool devicePointer = true) const
	{
		e_KernelBuffer<D> r;
		r.Data = devicePointer ? device : deviceMapped;
		r.Length = m_uLength;
		r.UsedCount = m_uPos;
		return r;
	}
	e_BufferReference<H, D> translate(const H* val) const
	{
		unsigned long long t0 = ((unsigned long long)val - (unsigned long long)host) / m_uHostBlockSize, t1 = ((unsigned long long)val - (unsigned long long)device) / sizeof(D);
		unsigned int i = t0 < m_uLength ? t0 : t1;
		return e_BufferReference<H, D>(this, i, 1);
	}
	e_BufferReference<H, D> translate(const D* val) const
	{
		unsigned long long t0 = ((unsigned long long)val - (unsigned long long)host) / sizeof(H), t1 = ((unsigned long long)val - (unsigned long long)device) / sizeof(D);
		unsigned int i = t0 < m_uLength ? t0 : t1;
		return e_BufferReference<H, D>(this, i, 1);
	}
};