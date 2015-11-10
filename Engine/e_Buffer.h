//WARNING
//This header can not be included in any *.cu file due to a bug in nvcc (CUDA 7.5)
//WARNING

#pragma once
#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <type_traits>
#include <boost/icl/interval_set.hpp>

#include "e_Buffer_device.h"
#include "../Base/Platform.h"
#include "../CudaMemoryManager.h"
#include "../VirtualFuncType.h"

template<typename H, typename D> class e_BufferIterator;
template<typename H, typename D> class e_BufferRange
{
public:
	virtual e_BufferIterator<H, D> begin() = 0;

	virtual e_BufferIterator<H, D> end() = 0;

	virtual e_BufferReference<H, D> operator()(size_t i, size_t l = 1) = 0;

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

	virtual e_BufferReference<H, D> translate(e_Variable<D> var) = 0;
};
template<typename T> using e_StreamRange = e_BufferRange<T, T>;

template<typename H, typename D> class e_BufferBase : public e_BufferRange<H, D>
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
	virtual void realloc(){}
	virtual D* getDeviceMappedData() = 0;

	template<typename H2, typename D2> friend class e_BufferIterator;
	template<typename H2, typename D2> friend class e_BufferReference;

public:

	typedef e_BufferIterator<H, D> iterator;

	e_BufferBase(size_t a_NumElements, size_t a_ElementSize, bool callUpdateElement)
		: m_uPos(0), m_uLength(a_NumElements), m_uBlockSize(a_ElementSize != MINUS_ONE ? a_ElementSize : sizeof(H)), m_bUpdateElement(callUpdateElement)
	{
		host = (H*)::malloc(m_uBlockSize * m_uLength);
		Platform::SetMemory(host, m_uBlockSize * a_NumElements);
		CUDA_MALLOC(&device, sizeof(D) * a_NumElements);
		cudaMemset(device, 0, sizeof(D) * a_NumElements);
		m_uInvalidated = new range_set_t();
		m_uDeallocated = new range_set_t();
	}

	virtual ~e_BufferBase()
	{
		Free();
	}

	virtual void Free()
	{
		for (auto it : *this)
			(*it).~H();
		free(host);
		CUDA_FREE(device);
		delete m_uInvalidated;
		delete m_uDeallocated;
	}

	size_t getBufferLength() const
	{
		return m_uLength;
	}

	e_BufferReference<H, D> malloc(size_t a_Length)
	{
		e_BufferReference<H, D> res;
		for (range_set_t::iterator it = m_uDeallocated->begin(); it != m_uDeallocated->end(); it++)
			if (it->upper() - it->lower() >= a_Length)
			{
				ival i(it->lower(), it->lower() + a_Length);
				res = e_BufferReference<H, D>(this, it->lower(), a_Length);
				m_uDeallocated->erase(i);
				break;
			}
		if (a_Length <= m_uLength - m_uPos)
		{
			m_uPos += a_Length;
			res = e_BufferReference<H, D>(this, m_uPos - a_Length, a_Length);
		}
		else
		{
			//BAD_EXCEPTION("Cuda data stream malloc failure, %d elements requested, %d available.", a_Count, m_uLength - m_uPos)
			size_t newLength = m_uPos + a_Length;
			CUDA_FREE(device);
			H* newHost = (H*)::malloc(m_uBlockSize * newLength);
			::memcpy(newHost, host, m_uPos * m_uBlockSize);
			free(host);
			host = newHost;
			m_uLength = newLength;
			CUDA_MALLOC(&device, sizeof(D) * newLength);
			cudaMemset(device, 0, sizeof(D) * newLength);
			realloc();
			Invalidate(0, m_uPos);
			return malloc(a_Length);

		}
		Invalidate(res);
		return res;
	}

	e_BufferReference<H, D> malloc(e_BufferReference<H, D> r, bool copyToNew = true)
	{
		e_BufferReference<H, D> r2 = malloc(r.l);
		if (copyToNew)
			memcpy(r2, r);
		return r2;
	}

	void memcpy(e_BufferReference<H, D> dest, e_BufferReference<H, D> source)
	{
		if (source.l > dest.l || source.buf != this || dest.buf != this)
			throw std::runtime_error(__FUNCTION__);
		::memcpy(dest(), source(), m_uBlockSize * dest.l);
	}

	void dealloc(e_BufferReference<H, D> ref)
	{
		dealloc(ref.p, ref.l);
	}

	void dealloc(size_t p, size_t l)
	{
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

	void Invalidate(e_BufferReference<H, D> ref)
	{
		Invalidate(ref.p, ref.l);
	}

	void Invalidate(size_t idx, size_t l)
	{
		m_uInvalidated->insert(ival(idx, idx + l));
	}

	void UpdateInvalidated()
	{
		struct f{ void operator()(e_BufferReference<H, D> r)
		{ 
		} };
		f _f;
		UpdateInvalidated(_f);
	}

	template<typename CLB> void UpdateInvalidated(CLB& f)
	{
		for (range_set_t::iterator it = m_uInvalidated->begin(); it != m_uInvalidated->end(); it++)
		{
			for (size_t i = it->lower(); i < it->upper(); i++)
			{
				f(operator()(i, 1));
				if (m_bUpdateElement)
					updateElement(i);
			}
			copyRange(it->lower(), it->upper() - it->lower());
		}
		m_uInvalidated->clear();
	}

	void CopyFromDevice(e_BufferReference<H, D> ref)
	{
		static_assert(std::is_same<H, D>::value, "H != T");
		ThrowCudaErrors(cudaMemcpy(this->host + ref.p, device + ref.p, sizeof(H) * ref.l, cudaMemcpyDeviceToHost));
	}

	virtual e_BufferReference<H, D> operator()(size_t i, size_t l = 1)
	{
		if (i >= m_uPos)
			throw std::runtime_error("Invalid idx!");
		return e_BufferReference<H, D>(this, i, l);
	}

	virtual e_BufferIterator<H, D> begin()
	{
		return e_BufferIterator<H, D>(*this, 0);
	}

	virtual e_BufferIterator<H, D> end()
	{
		return e_BufferIterator<H, D>(*this, m_uPos);
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

	virtual e_KernelBuffer<D> getKernelData(bool devicePointer = true) const = 0;

	virtual e_BufferReference<H, D> translate(e_Variable<D> var)
	{
		size_t idx = var.device - device;
		return e_BufferReference<H, D>(this, idx, 1);
	}

	template<typename T> e_BufferReference<H, D> translate(e_Variable<T> var)
	{
		size_t idx = (D*)var.device - device;
		return e_BufferReference<H, D>(this, idx, unsigned int(sizeof(T) / sizeof(D)));
	}
};

template<typename H, typename D> class e_BufferIterator
{
	e_BufferBase<H, D>& buf;
	size_t idx;
	typename e_BufferBase<H, D>::range_set_t::iterator next_interval;
public:
	e_BufferIterator(e_BufferBase<H, D>& b, size_t i)
		: buf(b), idx(i)
	{
		next_interval = buf.m_uDeallocated->lower_bound(typename e_BufferBase<H, D>::ival(idx, idx + 1));
	}

	e_BufferReference<H, D> operator*() const
	{
		return buf(idx, 1);
	}

	H* operator->() const
	{
		return buf.host + idx;
	}

	e_BufferIterator<H, D>& operator++()
	{
		/*idx++;
		auto it = buf.m_uDeallocated->find(idx);
		if(it != buf.m_uDeallocated->end())
		{
		idx = it->upper();
		}*/
		if (next_interval != buf.m_uDeallocated->end() && idx + 1 == next_interval->lower())
		{
			idx = next_interval->upper();
			next_interval = buf.m_uDeallocated->lower_bound(typename e_BufferBase<H, D>::ival(idx, idx + 1));
		}
		else idx++;
		if (idx > buf.m_uLength)
			throw std::runtime_error("Out of bounds!");
		return *this;
	}

	bool operator==(const e_BufferIterator<H, D>& rhs) const
	{
		return &buf == &rhs.buf && idx == rhs.idx;
	}

	bool operator!=(const e_BufferIterator<H, D>& rhs) const
	{
		return !(*this == rhs);
	}
};

template<typename H, typename D> size_t num_in_iterator(e_BufferRange<H, D>& range, size_t i)
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

template<typename H, typename D> class e_Buffer : public e_BufferBase<H, D>
{
	D* deviceMapped;
protected:
	virtual void updateElement(size_t i)
	{
		deviceMapped[i] = operator()(i)->getKernelData();
	}
	virtual void copyRange(size_t i, size_t l)
	{
		cudaMemcpy(this->device + i, deviceMapped + i, l * sizeof(D), cudaMemcpyHostToDevice);
	}
	virtual void realloc()
	{
		free(deviceMapped);
		deviceMapped = (D*)::malloc(m_uLength * sizeof(D));
		memset(deviceMapped, 0, sizeof(D) * m_uLength);
	}
	virtual D* getDeviceMappedData()
	{
		return deviceMapped;
	}
public:
	e_Buffer(size_t a_NumElements, size_t a_ElementSize = MINUS_ONE)
		: e_BufferBase<H, D>(a_NumElements, a_ElementSize, true)
	{
		deviceMapped = (D*)::malloc(a_NumElements * sizeof(D));
		memset(deviceMapped, 0, sizeof(D) * a_NumElements);
	}
	virtual void Free()
	{
		free(deviceMapped);
		e_BufferBase<H, D>::Free();
	}
	virtual e_KernelBuffer<D> getKernelData(bool devicePointer = true) const
	{
		e_KernelBuffer<D> r;
		r.Data = devicePointer ? device : deviceMapped;
		r.Length = (unsigned int)m_uLength;
		r.UsedCount = (unsigned int)m_uPos;
		return r;
	}
};

template<typename T> class e_Stream : public e_BufferBase<T, T>
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
	e_Stream(size_t a_NumElements)
		: e_BufferBase<T, T>(a_NumElements, sizeof(T), false)
	{

	}
	virtual e_KernelBuffer<T> getKernelData(bool devicePointer = true) const
	{
		e_KernelBuffer<T> r;
		r.Data = devicePointer ? device : host;
		r.Length = (unsigned int)m_uLength;
		r.UsedCount = (unsigned int)m_uPos;
		return r;
	}
};

template<typename H, typename D> class e_CachedBuffer : public e_Buffer<H, D>
{
private:
	struct entry
	{
		size_t count;
		e_BufferReference<H, D> ref;
		entry()
		{
			count = 0;
		}
	};
	std::map<std::string, entry> m_sEntries;
public:
	std::vector<e_BufferReference<H, D>> m_UnusedEntries;
	e_CachedBuffer(size_t a_Count, size_t a_ElementSize = MINUS_ONE)
		: e_Buffer<H, D>(a_Count, a_ElementSize)
	{

	}
	e_BufferReference<H, D> LoadCached(const std::string& file, bool& load)
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
		e.ref = malloc(1);
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

