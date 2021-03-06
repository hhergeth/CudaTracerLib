#include <StdAfx.h>
#include "FileStream.h"
#include <filesystem.h>

namespace CudaTracerLib {

bool IInStream::ReadTo(std::string& str, char end)
{
	char ch;
	str.clear();
	str.reserve(128);
	while (get(ch) && ch != end)
		str.push_back(ch);
	return ch == end && (!eof() || str.length());
}

unsigned char* IInStream::ReadToEnd()
{
	unsigned long long l = getFileSize(), r = l - getPos();
	void* v = malloc(r);
	Read((char*)v, r);
	return (unsigned char*)v;
}

FileInputStream::FileInputStream(const std::string& a_Name)
	: path(a_Name)
{
	numBytesRead = 0;
	m_ptr = fopen(a_Name.c_str(), "rb");
	m_uFileSize = std::filesystem::file_size(a_Name);
}

void FileInputStream::Close()
{
	if (m_ptr)
	{
		fclose(m_ptr);
		m_ptr = 0;
	}
}

void FileInputStream::Read(void* a_Data, size_t a_Size)
{
	if (numBytesRead + a_Size <= m_uFileSize)
	{
		auto elements_read = fread(a_Data, a_Size, 1, m_ptr);
		if(elements_read != 1)// read one element of size a_Size
            throw std::runtime_error("Error reading from file!");
		numBytesRead += a_Size;
	}
	else throw std::runtime_error("Passed end of file!");
}

void FileInputStream::Move(int off)
{
	if (size_t(numBytesRead + off) > m_uFileSize)
		throw std::runtime_error("Passed end of file or tried moving before beginning!");
	numBytesRead += off;
}

MemInputStream::MemInputStream(const unsigned char* _buf, size_t length, bool canKeep)
	: numBytesRead(0), path("")
{
	if (canKeep)
		this->buf = _buf;
	else
	{
		void* v = malloc(length);
		buf = (unsigned char*)v;
		memcpy(v, _buf, length);
	}
	m_uFileSize = length;
}

MemInputStream::MemInputStream(FileInputStream& in)
	: numBytesRead(0), path(in.getFilePath())
{
	m_uFileSize = in.getFileSize() - in.getPos();
	buf = in.ReadToEnd();
}

MemInputStream::MemInputStream(const std::string& a_Name)
	: numBytesRead(0), path(a_Name)
{
	FileInputStream in(a_Name);
	m_uFileSize = in.getFileSize();
	buf = in.ReadToEnd();
}

void MemInputStream::Read(void* a_Data, size_t a_Size)
{
	if (a_Size + numBytesRead > m_uFileSize)
		throw std::runtime_error("Stream not long enough!");
	memcpy(a_Data, buf + numBytesRead, a_Size);
	numBytesRead += a_Size;
}

IInStream* OpenFile(const std::string& filename)
{
	if (std::filesystem::file_size(filename) < 1024 * 1024 * 512)
		return new MemInputStream(filename);
	else return new FileInputStream(filename);
	return 0;
}

FileOutputStream::FileOutputStream(const std::string& a_Name)
{
	numBytesWrote = 0;
	H = fopen(a_Name.c_str(), "wb");
	if (!H)
		throw std::runtime_error("Could not open file!");
}

void FileOutputStream::_Write(const void* data, size_t size)
{
	size_t i = fwrite(data, 1, size, (FILE*)H);
	if (i != size)
		throw std::runtime_error("Could not write to file!");
	numBytesWrote += size;
}

void FileOutputStream::Close()
{
	if (H)
		if (fclose((FILE*)H))
			throw std::runtime_error("Could not close file!");
	H = 0;
}

}
