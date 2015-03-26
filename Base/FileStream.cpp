#include <StdAfx.h>
#include "FileStream.h"
#include <Windows.h>

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

InputStream::InputStream(const char* a_Name)
{
	path = std::string(a_Name);
	numBytesRead = 0;
	H = CreateFile(a_Name, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if(H == INVALID_HANDLE_VALUE)
	{
		std::cout << a_Name << "\n";
		LPVOID lpMsgBuf;
		DWORD dw = GetLastError(); 

		FormatMessage(
			FORMAT_MESSAGE_ALLOCATE_BUFFER | 
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dw,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPTSTR) &lpMsgBuf,
			0, NULL );
		std::cout << (LPTSTR)lpMsgBuf << "\n";
		throw std::runtime_error((LPTSTR)lpMsgBuf);
	}
	DWORD h;
	DWORD l = GetFileSize(H, &h);
	m_uFileSize = ((size_t)h << 32) | (size_t)l;
}

void InputStream::Close()
{
	CloseHandle(H);
}

void InputStream::Read(void* a_Data, unsigned int a_Size)
{
	DWORD a;
	BOOL b = ReadFile(H, a_Data, a_Size, &a, 0);
	if(!b || a != a_Size)
		throw std::runtime_error("Impossible to read from file!");
	numBytesRead += a_Size;
}

void InputStream::Move(int off)
{
	DWORD r = SetFilePointer(H, off, 0, FILE_CURRENT);
	if(r == INVALID_SET_FILE_POINTER)
		throw std::runtime_error("Impossible to skip in file!");
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

MemInputStream::MemInputStream(const char* a_Name)
{
	InputStream in(a_Name);
	numBytesRead = 0;
	m_uFileSize = in.getFileSize();
	buf = in.ReadToEnd();
	in.Close();
	path = a_Name;
}

void MemInputStream::Read(void* a_Data, unsigned int a_Size)
{
	if(a_Size + numBytesRead > m_uFileSize)
		throw std::runtime_error("Stream not long enough!");
	memcpy(a_Data, buf + numBytesRead, a_Size);
	numBytesRead += a_Size;
}

unsigned long long GetFileSize(const char* filename)
{
	HANDLE hFile = CreateFile(filename, GENERIC_READ, 
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile==INVALID_HANDLE_VALUE)
        return -1; // error condition, could call GetLastError to find out more

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        return -1; // error condition, could call GetLastError to find out more
    }

    CloseHandle(hFile);
    return size.QuadPart;
}

IInStream* OpenFile(const char* filename)
{
	if(GetFileSize(filename) < 1024 * 1024 * 1024)
		return new MemInputStream(filename);
	else return new InputStream(filename);
	return 0;
}

void CreateDirectoryRecursively(const std::string &directory)
{
  static const std::string separators("\\/");
 
  // If the specified directory name doesn't exist, do our thing
  DWORD fileAttributes = ::GetFileAttributes(directory.c_str());
  if(fileAttributes == INVALID_FILE_ATTRIBUTES) {
 
    // Recursively do it all again for the parent directory, if any
    std::size_t slashIndex = directory.find_last_of(separators);
    if(slashIndex != std::wstring::npos) {
      CreateDirectoryRecursively(directory.substr(0, slashIndex));
    }
 
    // Create the last directory on the path (the recursive calls will have taken
    // care of the parent directories by now)
    BOOL result = ::CreateDirectory(directory.c_str(), nullptr);
    if(result == FALSE) {
      throw std::runtime_error("Could not create directory");
    }
 
  } else { // Specified directory name already exists as a file or directory
 
    bool isDirectoryOrJunction =
      ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) ||
      ((fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);
 
    if(!isDirectoryOrJunction) {
      throw std::runtime_error(
        "Could not create directory because a file with the same name exists"
      );
    }
 
  }
}

unsigned long long GetTimeStamp(const char* filename)
{
	WIN32_FILE_ATTRIBUTE_DATA r;
	GetFileAttributesEx(filename, GetFileExInfoStandard, &r);
ULARGE_INTEGER    lv_Large ;

lv_Large.LowPart  = r.ftLastWriteTime.dwLowDateTime   ;
  lv_Large.HighPart = r.ftLastWriteTime.dwHighDateTime  ;

  return lv_Large.QuadPart ;
}

void SetTimeStamp(const char* filename, unsigned long long val)
{
	HANDLE Handle = CreateFile(filename, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
	if(Handle == INVALID_HANDLE_VALUE)
		throw std::runtime_error("Invalid handle!");
	FILETIME f;
	f.dwHighDateTime = val >> 32;
	f.dwLowDateTime = val & 0xffffffff;
	if(!SetFileTime(Handle, &f, &f, &f))
		throw std::runtime_error("Setting time stamp failed!");
	CloseHandle(Handle);
}

OutputStream::OutputStream(const char* a_Name)
{
	numBytesWrote = 0;
	H = CreateFile(a_Name, GENERIC_WRITE, FILE_SHARE_WRITE, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
}

void OutputStream::_Write(const void* data, unsigned int size)
{
	DWORD a;
	BOOL b = WriteFile(H, data, size, &a, 0);
	if(!b || a != size)
		throw std::runtime_error("Writing to file failed!");
	numBytesWrote += size;
}

void OutputStream::Close()
{
	CloseHandle(H);
}

/*



*/