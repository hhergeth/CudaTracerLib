#include <StdAfx.h>
#include "FileStream.h"
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp> 

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
	H = new boost::iostreams::mapped_file(a_Name, boost::iostreams::mapped_file::readonly);
	boost::iostreams::mapped_file& mmap = *(boost::iostreams::mapped_file*)H;
	m_uFileSize = mmap.size();
}

void FileInputStream::Close()
{
	if (H)
	{
		boost::iostreams::mapped_file* mmap = (boost::iostreams::mapped_file*)H;
		mmap->close();
		delete mmap;
		H = 0;
	}
}

void FileInputStream::Read(void* a_Data, size_t a_Size)
{
	boost::iostreams::mapped_file& mmap = *(boost::iostreams::mapped_file*)H;
	if (numBytesRead + a_Size <= m_uFileSize)
	{
		memcpy(a_Data, mmap.const_data() + numBytesRead, a_Size);
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
	if (boost::filesystem::file_size(filename) < 1024 * 1024 * 512)
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