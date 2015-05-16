#include <StdAfx.h>
#include "FileStream.h"
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp> 

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

InputStream::InputStream(const std::string& a_Name)
	: path(a_Name)
{
	numBytesRead = 0;
	H = new boost::iostreams::mapped_file(a_Name, boost::iostreams::mapped_file::readonly);
	boost::iostreams::mapped_file& mmap = *(boost::iostreams::mapped_file*)H;
	m_uFileSize = mmap.size();
}

void InputStream::Close()
{
	if (H)
	{
		boost::iostreams::mapped_file* mmap = (boost::iostreams::mapped_file*)H;
		mmap->close();
		delete mmap;
		H = 0;
	}
}

void InputStream::Read(void* a_Data, unsigned int a_Size)
{
	boost::iostreams::mapped_file& mmap = *(boost::iostreams::mapped_file*)H;
	if (numBytesRead + a_Size <= m_uFileSize)
	{
		memcpy(a_Data, mmap.const_data() + numBytesRead, a_Size);
		numBytesRead += a_Size;
	}
	else throw std::runtime_error("Passed end of file!");
}

void InputStream::Move(int off)
{
	if (numBytesRead + off > m_uFileSize)
		throw std::runtime_error("Passed end of file!");
	numBytesRead += off;
}

MemInputStream::MemInputStream(const unsigned char* _buf, unsigned int length, bool canKeep)
{
	path = "";
	if(canKeep)
		this->buf = _buf;
	else
	{
		void* v = malloc(length);
		buf = (unsigned char*)v;
		memcpy(v, _buf, length);
	}
	m_uFileSize = length;
	this->numBytesRead = 0;
}

MemInputStream::MemInputStream(InputStream& in)
{
	m_uFileSize = in.getFileSize() - in.getPos();
	buf = in.ReadToEnd();
	path = in.getFilePath();
}

MemInputStream::MemInputStream(const std::string& a_Name)
{
	InputStream in(a_Name);
	numBytesRead = 0;
	m_uFileSize = in.getFileSize();
	buf = in.ReadToEnd();
	path = a_Name;
}

void MemInputStream::Read(void* a_Data, unsigned int a_Size)
{
	if(a_Size + numBytesRead > m_uFileSize)
		throw std::runtime_error("Stream not long enough!");
	memcpy(a_Data, buf + numBytesRead, a_Size);
	numBytesRead += a_Size;
}

IInStream* OpenFile(const std::string& filename)
{
	if(boost::filesystem::file_size(filename) < 1024 * 1024 * 512)
		return new MemInputStream(filename);
	else return new InputStream(filename);
	return 0;
}

OutputStream::OutputStream(const std::string& a_Name)
{
	numBytesWrote = 0;
	H = fopen(a_Name.c_str(), "wb");
	if (!H)
		throw std::runtime_error("Could not open file!");
}

void OutputStream::_Write(const void* data, unsigned int size)
{
	size_t i = fwrite(data, 1, size, (FILE*)H);
	if (i != size)
		throw std::runtime_error("Could not write to file!");
	numBytesWrote += size;
}

void OutputStream::Close()
{
	if (H)
		if (fclose((FILE*)H))
			throw std::runtime_error("Could not close file!");
	H = 0;
}