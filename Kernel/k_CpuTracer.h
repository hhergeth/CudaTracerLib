#pragma once

#include "k_Tracer.h"
#include <windows.h>
#include <process.h>    /* _beginthread, _endthread */
#include <stddef.h>
#include <stdlib.h>
#include <conio.h>

#define TCOUNT 6

class k_CpuTracer : public k_ProgressiveTracer
{
	static void threadStart(void* arg);
public:
	struct threadData
	{
		k_CpuTracer* tracer;
		HANDLE sem;
		int i;
	};
	bool m_bDirect;
	k_CpuTracer();
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
private:
	threadData data[TCOUNT];
	HANDLE m_sem;
	e_Image* IMG;
};