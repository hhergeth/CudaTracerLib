#pragma once

#include <iostream>
#include <fstream>
#include <MathTypes.h>
#include "FixedString.h"
#include <Engine/Buffer_device.h>

namespace CudaTracerLib {

class IInStream
{
protected:
	unsigned long long m_uFileSize;
public:
	virtual ~IInStream()
	{

	}
	virtual void Read(void* a_Out, size_t a_Size) = 0;
	virtual size_t getPos() = 0;
	size_t getFileSize()
	{
		return m_uFileSize;
	}
	bool eof(){ return getPos() == getFileSize(); }
	virtual void Move(int off) = 0;
	virtual void Close() = 0;
	template<typename T> bool get(T& c)
	{
		if (getPos() + sizeof(T) <= getFileSize())
		{
			Read(&c, sizeof(T));
			return true;
		}
		else return false;
	}
	bool ReadTo(std::string& str, char end);
	bool getline(std::string& str)
	{
		return ReadTo(str, '\n');
	}
	std::string getline()
	{
		std::string s;
		getline(s);
		return s;
	}
	unsigned char* ReadToEnd();
	template<typename T> T* ReadToEnd()
	{
		return (T*)ReadToEnd();
	}
	template<typename T> void Read(T* a_Data, size_t a_Size)
	{
		Read((void*)a_Data, a_Size);
	}
	template<typename T> void Read(T& a)
	{
		Read((char*)&a, sizeof(T));
	}
	virtual const std::string& getFilePath() const = 0;
	template<int N> void Read(FixedString<N>& str)
	{
		Read((char*)&str, sizeof(FixedString<N>));
	}
	template<typename T, int N, bool b, unsigned char c> void Read(FixedSizeArray<T, N, b, c>& arr)
	{
		size_t l;
		*this >> l;
		arr.resize(l);
		for (unsigned int i = 0; i < l; i++)
			*this >> arr(i);
	}
public:
#define DCL_IN(TYPE) \
	IInStream& operator>>(TYPE& rhs) \
	{ \
		Read(rhs); \
		return *this; \
	}
	DCL_IN(signed char)
	DCL_IN(short int)
	DCL_IN(int)
	DCL_IN(long int)
	DCL_IN(long long)
	DCL_IN(unsigned char)
	DCL_IN(unsigned short)
	DCL_IN(unsigned int)
	DCL_IN(unsigned long int)
	DCL_IN(unsigned long long)
	DCL_IN(float)
	DCL_IN(double)
	DCL_IN(Vec2i)
	DCL_IN(Vec3i)
	DCL_IN(Vec4i)
	DCL_IN(Vec2f)
	DCL_IN(Vec3f)
	DCL_IN(Vec4f)
	DCL_IN(Spectrum)
	DCL_IN(AABB)
	DCL_IN(Ray)
	DCL_IN(float4x4)
#undef DCL_IN

	template<typename H, typename D> IInStream& operator>>(BufferReference<H, D> rhs)
	{
		Read(rhs(0), rhs.getHostSize());
		rhs.Invalidate();
		return *this;
	}
};

class FileInputStream : public IInStream
{
private:
	size_t numBytesRead;
	void* H;
	std::string path;
public:
	FileInputStream(const std::string& a_Name);
	~FileInputStream()
	{
		Close();
	}
	virtual void Close();
	virtual size_t getPos()
	{
		return numBytesRead;
	}
	virtual void Read(void* a_Data, size_t a_Size);
	void Move(int off);
	virtual const std::string& getFilePath() const
	{
		return path;
	}
};

class MemInputStream : public IInStream
{
private:
	size_t numBytesRead;
	const unsigned char* buf;
	std::string path;
public:
	MemInputStream(const unsigned char* buf, size_t length, bool canKeep = false);
	MemInputStream(FileInputStream& in);
	MemInputStream(const std::string& a_Name);
	~MemInputStream()
	{
		Close();
	}
	virtual void Close()
	{
		if (buf)
		{
			free((void*)buf);
			buf = 0;
		}
		buf = 0;
	}
	virtual size_t getPos()
	{
		return numBytesRead;
	}
	virtual void Read(void* a_Data, size_t a_Size);
	void Move(int off)
	{
		numBytesRead += off;
	}
	virtual const std::string& getFilePath() const
	{
		return path;
	}
};

IInStream* OpenFile(const std::string& filename);

class FileOutputStream
{
private:
	size_t numBytesWrote;
	void* H;
	void _Write(const void* data, size_t size);
public:
	FileOutputStream(const std::string& a_Name);
	virtual ~FileOutputStream()
	{
		Close();
	}
	void Close();
	size_t GetNumBytesWritten()
	{
		return numBytesWrote;
	}
	template<typename T> void Write(T* a_Data, size_t a_Size)
	{
		_Write(a_Data, a_Size);
	}
	template<typename T> void Write(const T& a_Data)
	{
		_Write(&a_Data, sizeof(T));
	}
	template<int N> void Write(const FixedString<N>& str)
	{
		Write((char*)&str, sizeof(FixedString<N>));
	}
	template<typename T, int N, bool b, unsigned char c> void Write(const FixedSizeArray<T, N, b, c>& arr)
	{
		*this << arr.size();
		for (size_t i = 0; i < arr.size(); i++)
			*this << arr(i);
	}
#define DCL_OUT(TYPE) \
	FileOutputStream& operator<<(TYPE rhs) \
		{ \
		Write(rhs); \
		return *this; \
		}
	DCL_OUT(signed char)
	DCL_OUT(short int)
	DCL_OUT(int)
	DCL_OUT(long int)
	DCL_OUT(long long)
	DCL_OUT(unsigned char)
	DCL_OUT(unsigned short)
	DCL_OUT(unsigned int)
	DCL_OUT(unsigned long int)
	DCL_OUT(unsigned long long)
	DCL_OUT(float)
	DCL_OUT(double)
	DCL_OUT(Vec2i)
	DCL_OUT(Vec3i)
	DCL_OUT(Vec4i)
	DCL_OUT(Vec2f)
	DCL_OUT(Vec3f)
	DCL_OUT(Vec4f)
	DCL_OUT(Spectrum)
	DCL_OUT(AABB)
	DCL_OUT(Ray)
	DCL_OUT(float4x4)
#undef DCL_OUT

	template<typename H, typename D> FileOutputStream& operator<<(BufferReference<H, D> rhs)
	{
		_Write(rhs(0), rhs.getHostSize());
		return *this;
	}
};

}
