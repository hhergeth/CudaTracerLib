#pragma once
#define STOPINCL
#define FW_DO_NOT_OVERRIDE_NEW_DELETE
#include <base/Array.hpp>
#include <io/Stream.hpp>
#include <gpu/Buffer.hpp>
#include <3d/Mesh.hpp>
#include <gpu/CudaModule.hpp>
#include "FileStream.h"

class TmpInStream : public FW::InputStream
{
public:
	TmpInStream       (::InputStream* A)                      { F = A; }
	virtual ~TmpInStream      (void)
	{

	}
	virtual int             read                    (void* ptr, int size)
	{
		F->Read(ptr, size);
		return size;
	}

private:
	::InputStream* F;
};

class TmpOutStream : public FW::OutputStream
{
public:
	TmpOutStream(::OutputStream* A)
	{
		F = A;
	}
	virtual                 ~TmpOutStream     (void)
	{

	}

	virtual void            write                   (const void* ptr, int size)
	{
		F->Write(ptr, size);
	}
	virtual void            flush                   (void)
	{

	}
private:
	::OutputStream* F;
};