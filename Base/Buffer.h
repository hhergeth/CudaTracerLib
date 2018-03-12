//WARNING
//This header can not be included in any *.cu file due to a bug in nvcc (CUDA 7.5)
//WARNING

#pragma once
#include <stdio.h>
#include <cstring>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <type_traits>
#include <boost/icl/interval_set.hpp>
#include "Buffer_device.h"
#include <Base/Platform.h>
#include <Base/CudaMemoryManager.h>
#include <Base/VirtualFuncType.h>

namespace CudaTracerLib {

namespace __buffer_internal__
{

}

template<typename H, typename D> class BufferIterator;
template<typename H, typename D> class BufferRange
{
public:
	virtual BufferIterator<H, D> begin() = 0;

	virtual BufferIterator<H, D> end() = 0;

	virtual BufferReference<H, D> operator()(size_t i, size_t l = 1) = 0;

	bool hasElements()
	{
		return hasMoreThanElements(0);
	}

	virtual bool hasMoreThanElements(size_t i)
	{
		return num_in_iterator(*this, i) > i;
	}

	virtual size_t numElements()
	{
		return num_in_iterator(*this, MINUS_ONE);
	}

	virtual BufferReference<H, D> translate(e_Variable<D> var) = 0;
};
template<typename T> using StreamRange = BufferRange<T, T>;

template<typename H, typename D> class BufferBase : public BufferRange<H, D>
{
protected:
	typedef boost::icl::interval_set<size_t, ICL_COMPARE_INSTANCE(std::less, size_t), boost::icl::right_open_interval<size_t>> range_set_t;
	typedef typename range_set_t::interval_type ival;

	H* host;
	D* device;
	size_t m_uPos;
	size_t m_uLength;
	size_t m_uBlockSize;
	bool m_bUpdateElement;
	range_set_t* m_uInvalidated;
	range_set_t* m_uDeallocated;

	virtual void updateElement(size_t i) = 0;
	virtual void copyRange(size_t i, size_t l) = 0;
	virtual void reallocAfterResize(){}
	virtual D* getDeviceMappedData() = 0;

	template<typename H2, typename D2> friend class BufferIterator;
	template<typename H2, typename D2> friend class BufferReference;

protected:

	BufferReference<H, D> malloc_internal(size_t a_Length)
	{
		BufferReference<H, D> res;
		for (range_set_t::iterator it = m_uDeallocated->begin(); it != m_uDeallocated->end(); ++it)
			if (it->upper() - it->lower() >= a_Length)
			{
				ival i(it->lower(), it->lower() + a_Length);
				res = BufferReference<H, D>(this, it->lower(), a_Length);
				m_uDeallocated->erase(i);
				break;
			}
		if (a_Length <= m_uLength - m_uPos)
		{
			m_uPos += a_Length;
			res = BufferReference<H, D>(this, m_uPos - a_Length, a_Length);
		}
		else
		{
			//BAD_EXCEPTION("Cuda data stream malloc failure, %d elements requested, %d available.", a_Count, m_uLength - m_uPos)
			size_t newLength = std::max(m_uPos + a_Length, m_uLength + m_uLength / 2);
			std::cout << __FUNCTION__ << " :: Resizing buffer from " << m_uLength << " to " << newLength << " elements" << std::endl;
			CUDA_FREE(device);
			H* newHost = (H*)::malloc(m_uBlockSize * newLength);
			::memcpy(newHost, host, m_uPos * m_uBlockSize);
			free(host);
			host = newHost;
			m_uLength = newLength;
			CUDA_MALLOC(&device, sizeof(D) * newLength);
			cudaMemset(device, 0, sizeof(D) * newLength);
			reallocAfterResize();
			Invalidate(0, m_uPos);
			return malloc_internal(a_Length);

		}
		Invalidate(res);
		return res;
	}

	template<bool CALL_F, typename CLB> void __UpdateInvalidated_internal(const CLB& f)
	{
		for (range_set_t::iterator it = m_uInvalidated->begin(); it != m_uInvalidated->end(); ++it)
		{
			if (CALL_F || m_bUpdateElement)
			{
				for (size_t i = it->lower(); i < it->upper(); i++)
				{
					if (CALL_F)
						f(BufferBase<H, D>::operator()(i, 1));
					if (m_bUpdateElement)
						updateElement(i);
				}
			}
			copyRange(it->lower(), it->upper() - it->lower());
		}
		m_uInvalidated->clear();
	}

public:

	typedef BufferIterator<H, D> iterator;

	BufferBase(size_t a_NumElements, size_t a_ElementSize, bool callUpdateElement)
		: m_uPos(0), m_uLength(a_NumElements), m_uBlockSize(a_ElementSize != MINUS_ONE ? a_ElementSize : sizeof(H)), m_bUpdateElement(callUpdateElement)
	{
		host = (H*)::malloc(m_uBlockSize * m_uLength);
		Platform::SetMemory(host, m_uBlockSize * a_NumElements);
		CUDA_MALLOC(&device, sizeof(D) * a_NumElements);
		cudaMemset(device, 0, sizeof(D) * a_NumElements);
		m_uInvalidated = new range_set_t();
		m_uDeallocated = new range_set_t();
	}

	virtual ~BufferBase()
	{
		if (device == 0)
		{
			std::cout << "Trying to destruct buffer multiple times!" << std::endl;
			return;
		}
		for (auto it : *this)
			(*it).~H();
		free(host);
		CUDA_FREE(device);
		delete m_uInvalidated;
		delete m_uDeallocated;
		device = 0;
		host = 0;
		m_uInvalidated = m_uDeallocated = 0;
	}

	size_t getBufferLength() const
	{
		return m_uLength;
	}

	BufferReference<H, D> malloc(size_t a_Length)
	{
		static_assert(!std::is_same<H, char>::value, "Please use malloc_aligned instead of malloc on a char buffer to guarantee alignment!");
		return malloc_internal(a_Length);
	}

	BufferReference<char, char> malloc_aligned(unsigned int a_Count, unsigned int a_Alignment)
	{
		static_assert(std::is_same<H, char>::value, "Do not use malloc_aligned on non char buffers! Alignement is guaranteed automatically");
		BufferReference<char, char> ref = malloc_internal(a_Count + a_Alignment * 2);
		uintptr_t ptr = (uintptr_t)ref.getDevice();
		unsigned int diff = ptr % a_Alignment, off = a_Alignment - diff;
		if (diff)
		{
			unsigned int end = ref.getIndex() + off + a_Count;
			BufferReference<char, char> refFreeFront = BufferReference<char, char>(this, ref.getIndex(), off),
										refFreeTail = BufferReference<char, char>(this, end, ref.getLength() - off - a_Count);
			dealloc(refFreeFront);
			if (refFreeTail.getLength() != 0)
				dealloc(refFreeTail);
			return BufferReference<char, char>(this, ref.getIndex() + off, a_Count);
		}
		else
		{
			BufferReference<char, char> refFreeTail = BufferReference<char, char>(this, ref.getIndex() + a_Count, a_Alignment * 2);
			dealloc(refFreeTail);
			return BufferReference<char, char>(this, ref.getIndex(), a_Count);
		}
	}

	//count is in bytes!
	template<typename T> StreamReference<char> malloc_aligned(unsigned int a_Count)
	{
		CTL_ASSERT((a_Count % sizeof(T)) == 0);
		return malloc_aligned(a_Count, std::alignment_of<T>::value);
	}

	BufferReference<H, D> malloc(BufferReference<H, D> r, bool copyToNew = true)
	{
		BufferReference<H, D> r2 = malloc(r.l);
		if (copyToNew)
			memcpy(r2, r);
		return r2;
	}

	void memcpy(BufferReference<H, D> dest, BufferReference<H, D> source)
	{
		if (source.l > dest.l || source.buf != this || dest.buf != this)
			throw std::runtime_error(__FUNCTION__);
		::memcpy(dest(), source(), m_uBlockSize * dest.l);
	}

	void dealloc(BufferReference<H, D> ref)
	{
		dealloc(ref.p, ref.l);
	}

	void dealloc(size_t p, size_t l)
	{
		m_uInvalidated->erase(ival(p, p + l));
		for (size_t i = p; i < p + l; ++i)
			host[i].~H();
		if (p + l == m_uPos)
		{
			m_uPos -= l;
		}
		else
		{
			m_uDeallocated->insert(ival(p, p + l));
		}

		if (m_uPos > 0)
		{
			range_set_t::const_iterator it = m_uDeallocated->find(m_uPos - 1);
			if (it != m_uDeallocated->end())
			{
				m_uPos -= it->upper() - it->lower();
				m_uDeallocated->erase(it);
			}
		}
	}

	void Invalidate()
	{
		m_uInvalidated->insert(ival(0U, m_uPos));
	}

	void Invalidate(BufferReference<H, D> ref)
	{
		Invalidate(ref.p, ref.l);
	}

	void Invalidate(size_t idx, size_t l)
	{
		m_uInvalidated->insert(ival(idx, idx + l));
	}

	void UpdateInvalidated()
	{
		struct f{
			void operator()(BufferReference<H, D> r) const
			{
			}
		};
		f _f;
		__UpdateInvalidated_internal<false>(_f);
	}

	template<typename CLB> void UpdateInvalidated(const CLB& f)
	{
		__UpdateInvalidated_internal<true>(f);
	}

	void CopyFromDevice(BufferReference<H, D> ref)
	{
		static_assert(std::is_same<H, D>::value, "H != T");
		ThrowCudaErrors(cudaMemcpy(this->host + ref.p, device + ref.p, sizeof(H) * ref.l, cudaMemcpyDeviceToHost));
	}

	virtual BufferReference<H, D> operator()(size_t i, size_t l = 1)
	{
		if (i >= m_uPos)
			throw std::runtime_error("Invalid idx!");
		return BufferReference<H, D>(this, i, l);
	}

	virtual BufferIterator<H, D> begin()
	{
		return BufferIterator<H, D>(*this, 0, false);
	}

	virtual BufferIterator<H, D> end()
	{
		return BufferIterator<H, D>(*this, m_uPos, true);
	}

	virtual size_t numElements()
	{
		if (m_uDeallocated->empty())
			return m_uPos;
		size_t p = m_uPos;
		for (range_set_t::iterator it = m_uDeallocated->begin(); it != m_uDeallocated->end(); it++)
		{
			size_t n = it->upper() - it->lower();
			p -= n;
		}
		return p;
	}

	virtual bool hasMoreThanElements(size_t i)
	{
		return numElements() > i;
	}

	size_t getDeviceSizeInBytes() const
	{
		return m_uLength * sizeof(D);
	}

	virtual KernelBuffer<D> getKernelData(bool devicePointer = true) const = 0;

	virtual BufferReference<H, D> translate(e_Variable<D> var)
	{
		size_t idx = var.device - device;
		return BufferReference<H, D>(this, idx, 1);
	}

	template<typename T> BufferReference<H, D> translate(e_Variable<T> var)
	{
		size_t idx = (D*)var.device - device;
		return BufferReference<H, D>(this, idx, (unsigned int)(sizeof(T) / sizeof(D)));
	}
};

template<typename H, typename D> class BufferIterator
{
	BufferBase<H, D>& buf;
	size_t idx;
	typename BufferBase<H, D>::range_set_t::iterator next_interval;
public:
	BufferIterator(BufferBase<H, D>& b, size_t i, bool isEnd)
		: buf(b), idx(i)
	{
		if (!isEnd)
		{
			next_interval = buf.m_uDeallocated->find(idx);
			if (next_interval != buf.m_uDeallocated->end())
			{
				idx = next_interval->upper();
				next_interval = buf.m_uDeallocated->lower_bound(typename BufferBase<H, D>::ival(idx, idx + 1));
			}
		}
	}

	BufferReference<H, D> operator*() const
	{
		return buf(idx, 1);
	}

	H* operator->() const
	{
		return buf.host + idx;
	}

	BufferIterator<H, D>& operator++()
	{
		if (next_interval != buf.m_uDeallocated->end() && idx + 1 == next_interval->lower())
		{
			idx = next_interval->upper();
			next_interval = buf.m_uDeallocated->lower_bound(typename BufferBase<H, D>::ival(idx, idx + 1));
		}
		else idx++;
		if (idx > buf.m_uLength)
			throw std::runtime_error("Out of bounds!");
		return *this;
	}

	bool operator==(const BufferIterator<H, D>& rhs) const
	{
		return &buf == &rhs.buf && idx == rhs.idx;
	}

	bool operator!=(const BufferIterator<H, D>& rhs) const
	{
		return !(*this == rhs);
	}
};

template<typename H, typename D> size_t num_in_iterator(BufferRange<H, D>& range, size_t i)
{
	size_t n = 0;
	for (auto it : range)
	{
		n++;
		if (n > i)
			return n;
	}
	return n;
}

template<typename H, typename D> class Buffer : public BufferBase<H, D>
{
	D* deviceMapped;
protected:
	virtual void updateElement(size_t i)
	{
		deviceMapped[i] = BufferBase<H, D>::operator()(i)->getKernelData();
	}
	virtual void copyRange(size_t i, size_t l)
	{
		cudaMemcpy(this->device + i, deviceMapped + i, l * sizeof(D), cudaMemcpyHostToDevice);
	}
	virtual void reallocAfterResize()
	{
		free(deviceMapped);
		deviceMapped = (D*)::malloc(BufferBase<H, D>::m_uLength * sizeof(D));
		memset(deviceMapped, 0, sizeof(D) * BufferBase<H, D>::m_uLength);
	}
	virtual D* getDeviceMappedData()
	{
		return deviceMapped;
	}
public:
	explicit Buffer(size_t a_NumElements, size_t a_ElementSize = MINUS_ONE)
		: BufferBase<H, D>(a_NumElements, a_ElementSize, true)
	{
		deviceMapped = (D*)::malloc(a_NumElements * sizeof(D));
		memset(deviceMapped, 0, sizeof(D) * a_NumElements);
	}
	virtual ~Buffer()
	{
		free(deviceMapped);
	}
	virtual KernelBuffer<D> getKernelData(bool devicePointer = true) const
	{
		KernelBuffer<D> r;
		r.Data = devicePointer ? BufferBase<H, D>::device : deviceMapped;
		r.Length = (unsigned int)BufferBase<H, D>::m_uLength;
		r.UsedCount = (unsigned int)BufferBase<H, D>::m_uPos;
		return r;
	}
};

template<typename T> class Stream : public BufferBase<T, T>
{
protected:
	virtual void updateElement(size_t)
	{

	}
	virtual void copyRange(size_t i, size_t l)
	{
		ThrowCudaErrors(cudaMemcpy(this->device + i, this->host + i, l * sizeof(T), cudaMemcpyHostToDevice));
	}
	virtual T* getDeviceMappedData()
	{
		return this->host;
	}
public:
	explicit Stream(size_t a_NumElements)
		: BufferBase<T, T>(a_NumElements, sizeof(T), false)
	{

	}
	virtual KernelBuffer<T> getKernelData(bool devicePointer = true) const
	{
		KernelBuffer<T> r;
		r.Data = devicePointer ? BufferBase<T, T>::device : BufferBase<T, T>::host;
		r.Length = (unsigned int)BufferBase<T, T>::m_uLength;
		r.UsedCount = (unsigned int)BufferBase<T, T>::m_uPos;
		return r;
	}
};

template<typename H, typename D> class CachedBuffer : public Buffer<H, D>
{
private:
	struct entry
	{
		size_t count;
		BufferReference<H, D> ref;
		entry()
		{
			count = 0;
		}
	};
	std::map<std::string, entry> m_sEntries;
public:
	std::vector<BufferReference<H, D>> m_UnusedEntries;
	explicit CachedBuffer(size_t a_Count, size_t a_ElementSize = MINUS_ONE)
		: Buffer<H, D>(a_Count, a_ElementSize)
	{

	}
	BufferReference<H, D> LoadCached(const std::string& file, bool& load)
	{
		typename std::map<std::string, entry>::iterator it = m_sEntries.find(file);
		if (it != m_sEntries.end())
		{
			it->second.count++;
			load = false;
			return it->second.ref;
		}
		load = true;
		entry e;
		e.count = 1;
		e.ref = Buffer<H, D>::malloc(1);
		m_sEntries[file] = e;
		return e.ref;
	}
	void Release(const std::string& file)
	{
		typename std::map<std::string, entry>::iterator it = m_sEntries.find(file);
		if (it == m_sEntries.end())
			throw std::runtime_error(std::string(__FUNCTION__) + " : Entry not found!");
		it->second.count--;
		if (it->second.count == 0)
		{
			m_UnusedEntries.push_back(it->second.ref);
			m_sEntries.erase(it);
		}
	}
};

}
