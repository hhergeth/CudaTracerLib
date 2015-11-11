#pragma once

#include <Defines.h>

namespace CudaTracerLib {

#define MINUS_ONE std::numeric_limits<size_t>::max()

template<typename T> struct KernelBuffer
{
	T* Data;
	unsigned int UsedCount;
	unsigned int Length;
	CUDA_FUNC_IN T& operator[](unsigned int i) const
	{
		CT_ASSERT(i < UsedCount);
		return Data[i];
	}
};

template<typename H, typename D> class BufferBase;
class IInStream;
class FileOutputStream;
template<typename H, typename D> class BufferReference
{
	template<typename H2, typename D2> friend class BufferBase;
	BufferBase<H, D>* buf;
	size_t p, l;
public:
	BufferReference()
		: buf(0), p(MINUS_ONE), l(MINUS_ONE)
	{

	}

	BufferReference(BufferBase<H, D>* buf, size_t p, size_t l)
		: buf(buf), p(p), l(l)
	{
	}

	bool operator==(const BufferReference<H, D>& rhs) const
	{
		return buf == rhs.buf && p == rhs.p && l == rhs.l;
	}

	bool operator!=(const BufferReference<H, D>& rhs) const
	{
		return !(*this == rhs);
	}

	H* operator->()
	{
		return atH(p);
	}

	H& operator*()
	{
		return *atH(p);
	}

	BufferReference<H, D> operator()(size_t i = 0)
	{
		return BufferReference<H, D>(buf, p + i, 1);
	}

	operator H*() const
	{
		return atH(p);
	}

	bool operator<(const BufferReference<H, D>& rhs) const
	{
		return p < rhs.p;
	}

	BufferReference<H, D>& operator= (const H &r)
	{
		*atH(p) = r;
		return *this;
	}

	/*BufferReference<H, D>& operator= (const BufferReference<H, D>& src)
	{
	if (*this != src)
	buf->memcpy_(*this, src);
	return *this;
	}*/

	unsigned int getIndex() const
	{
		return (unsigned int)p;
	}

	unsigned int getLength() const
	{
		return (unsigned int)l;
	}

	D* getDevice()
	{
		return atD(p);
	}

	void Invalidate()
	{
		buf->Invalidate(*this);
	}

	void CopyFromDevice()
	{
		buf->CopyFromDevice(*this);
	}

	e_Variable<D> AsVar()
	{
		return AsVar<D>();
	}

	template<typename T> e_Variable<T> AsVar()
	{
		return e_Variable<T>((T*)(buf->getDeviceMappedData() + p), (T*)atD(p));
	}

	size_t getHostSize() const
	{
		return l * buf->m_uBlockSize;
	}

	size_t getDeviceSize() const
	{
		return l * sizeof(D);
	}
private:
	H* atH(size_t i) const
	{
		return (H*)((char*)buf->host + buf->m_uBlockSize * p);
	}
	D* atD(size_t i) const
	{
		return buf->device + i;
	}
};

template<typename T> using StreamReference = BufferReference<T, T>;

}