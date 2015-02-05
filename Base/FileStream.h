#pragma once

#include <iostream>
#include <fstream>
#include <MathTypes.h>

#define DCL_IN(TYPE) \
	IInStream& operator>>(TYPE& rhs) \
	{ \
		Read(rhs); \
		return *this; \
	}

#define DCL_OUT(TYPE) \
	OutputStream& operator<<(TYPE rhs) \
	{ \
		Write(rhs); \
		return *this; \
	}

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
	virtual void Close() = 0;
	void ToBegin()
	{
		Move((int)getPos());
	}
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
	virtual const char* getFilePath() const = 0;
public:
	DCL_IN(char)
	DCL_IN(short)
	DCL_IN(int)
	DCL_IN(long long)
	DCL_IN(unsigned char)
	DCL_IN(unsigned short)
	DCL_IN(unsigned int)
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
};

class InputStream : public IInStream
{
private:
	unsigned int numBytesRead;
	void* H;
	std::string path;
public:
	InputStream(const char* a_Name);
	virtual void Close();
	virtual unsigned long long getPos()
	{
		return numBytesRead;
	}
	virtual void Read(void* a_Data, unsigned int a_Size);
	void Move(int off);
	virtual const char* getFilePath() const
	{
		return path.c_str();
	}
};

class MemInputStream : public IInStream
{
private:
	unsigned int numBytesRead;
	const unsigned char* buf;
	std::string path;
public:
	MemInputStream(const unsigned char* buf, unsigned int length, bool canKeep = false);
	MemInputStream(InputStream& in);
	MemInputStream(const char* a_Name);
	virtual void Close()
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
	virtual const char* getFilePath() const
	{
		return path.c_str();
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
	DCL_OUT(char)
	DCL_OUT(short)
	DCL_OUT(int)
	DCL_OUT(long long)
	DCL_OUT(unsigned char)
	DCL_OUT(unsigned short)
	DCL_OUT(unsigned int)
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
};