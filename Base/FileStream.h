#pragma once

#include <iostream>
#include <fstream>
#include "..\Math\vector.h"
#include "..\Math\AABB.h"

class IInStream
{
private:
	template<typename T> void Read(T& a)
	{
		Read((char*)&a, sizeof(T));
	}
public:
	virtual void Read(char* a_Out, unsigned int a_Size) = 0;
	virtual unsigned long long getPos() = 0;
	virtual unsigned long long getFileSize() = 0;
public:
	IInStream& operator>>(char& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(short& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(int& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(unsigned char& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(unsigned short& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(unsigned int& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(unsigned long& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(unsigned long long& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(long long& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(float& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(double& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(float2& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(float3& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(float4& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(AABB& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(Ray& rhs)
	{
		Read(rhs);
		return *this;
	}
	IInStream& operator>>(float4x4& rhs)
	{
		Read(rhs);
		return *this;
	}
};

class InputStream : public IInStream
{
public:
	unsigned int numBytesRead;
	HANDLE H;
public:
	InputStream(const char* a_Name)
	{
		numBytesRead = 0;
		H = CreateFile(a_Name, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	}
	void Close()
	{
		CloseHandle(H);
	}
	virtual unsigned long long getFileSize()
	{
		DWORD h;
		DWORD l = GetFileSize(H, &h);
		return ((size_t)h << 32) | (size_t)l;
	}
	template<typename T> T* ReadToEnd()
	{
		unsigned long long l = getFileSize(), r = l - numBytesRead;
		void* v = malloc(r);
		Read((char*)v, r);
		return (T*)v;
	}
	template<typename T> void Read(T* a_Data, unsigned int a_Size)
	{
		Read((char*)a_Data, a_Size);
	}
	template<typename T> void Read(const T& a_Data)
	{
		Read(&a_Data, sizeof(T));
	}
	virtual unsigned long long getPos()
	{
		return numBytesRead;
	}
	virtual void Read(char* a_Data, unsigned int a_Size)
	{
		DWORD a;
		BOOL b = ReadFile(H, a_Data, a_Size, &a, 0);
		if(!b || a != a_Size)
			throw 1;
		numBytesRead += a_Size;
	}
	void Move(unsigned int off)
	{
		DWORD r = SetFilePointer(H, off, 0, 1);
		if(r == INVALID_SET_FILE_POINTER)
			throw 1;
	}
	template<typename T> void Move(unsigned int N)
	{
		Move(sizeof(T) * N);
	}
};

class OutputStream
{
public:
	unsigned int numBytesWrote;
	HANDLE H;
public:
	OutputStream(const char* a_Name)
	{
		numBytesWrote = 0;
		H = CreateFile(a_Name, GENERIC_WRITE, FILE_SHARE_WRITE, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
	}
	void Close()
	{
		CloseHandle(H);
	}
	template<typename T> void Write(T* a_Data, unsigned int a_Size)
	{
		DWORD a;
		BOOL b = WriteFile(H, a_Data, a_Size, &a, 0);
		if(!b || a != a_Size)
			throw 1;
		numBytesWrote += a_Size;
	}
	template<typename T> void Write(const T& a_Data)
	{
		DWORD a;
		BOOL b = WriteFile(H, (void*)&a_Data, sizeof(T), &a, 0);
		if(!b || a != sizeof(T))
			throw 1;
		numBytesWrote += sizeof(T);
	}
	OutputStream& operator<<(char rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(short rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(int rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(long long rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(unsigned char rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(unsigned short rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(unsigned int rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(unsigned long long rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(float rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(double rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(float2 rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(float3 rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(float4 rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(AABB rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(Ray rhs)
	{
		Write(rhs);
		return *this;
	}
	OutputStream& operator<<(float4x4 rhs)
	{
		Write(rhs);
		return *this;
	}
};