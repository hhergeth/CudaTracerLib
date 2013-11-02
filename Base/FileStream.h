#pragma once

#include <iostream>
#include <fstream>
#include <MathTypes.h>

class IInStream
{
protected:
	unsigned long long m_uFileSize;
public:
	virtual void Read(void* a_Out, unsigned int a_Size) = 0;
	virtual unsigned long long getPos() = 0;
	unsigned long long getFileSize()
	{
		return m_uFileSize;
	}
	bool eof(){return getPos() == getFileSize();}
	virtual void Move(int off) = 0;
	template<typename T> void Move(int num)
	{
		Move(num * sizeof(T));
	}
	template<typename T> bool get(T& c)
	{
		if(getPos() + sizeof(T) <= getFileSize())
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
	template<typename T> void Read(T* a_Data, unsigned int a_Size)
	{
		Read((void*)a_Data, a_Size);
	}
	template<typename T> void Read(T& a)
	{
		Read((char*)&a, sizeof(T));
	}
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
	IInStream& operator>>(Spectrum& rhs)
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
private:
	unsigned int numBytesRead;
	void* H;
public:
	InputStream(const char* a_Name);
	void Close();
	virtual unsigned long long getPos()
	{
		return numBytesRead;
	}
	virtual void Read(void* a_Data, unsigned int a_Size);
	void Move(int off);
};

class MemInputStream : public IInStream
{
private:
	unsigned int numBytesRead;
	const unsigned char* buf;
public:
	MemInputStream(const unsigned char* buf, unsigned int length, bool canKeep = false);
	MemInputStream(InputStream& in);
	MemInputStream(const char* a_Name);
	void Close()
	{
		delete [] buf;
	}
	virtual unsigned long long getPos()
	{
		return numBytesRead;
	}
	virtual void Read(void* a_Data, unsigned int a_Size);
	void Move(int off)
	{
		numBytesRead += off;
	}
};

unsigned long long GetFileSize(const char* filename);

IInStream* OpenFile(const char* filename);

void CreateDirectoryRecursively(const std::string &directory);

unsigned long long GetTimeStamp(const char* filename);

void SetTimeStamp(const char* filename, unsigned long long);

class OutputStream
{
private:
	unsigned int numBytesWrote;
	void* H;
	void _Write(const void* data, unsigned int size);
public:
	OutputStream(const char* a_Name);
	void Close();
	unsigned int GetNumBytesWritten()
	{
		return numBytesWrote;
	}
	template<typename T> void Write(T* a_Data, unsigned int a_Size)
	{
		_Write(a_Data, a_Size);
	}
	template<typename T> void Write(const T& a_Data)
	{
		_Write(&a_Data, sizeof(T));
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
	OutputStream& operator<<(Spectrum rhs)
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