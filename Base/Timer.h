#include <Windows.h>

#pragma once

class cTimer
{
private:
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
    LARGE_INTEGER stop;
public:
	cTimer()
	{
		QueryPerformanceFrequency( &frequency ) ;
	}
	void StartTimer()
	{
		QueryPerformanceCounter(&start);
	}
	double EndTimer()
	{
		QueryPerformanceCounter(&stop);
		return getElapsedTime();
	}
	double getElapsedTime()
	{
		LARGE_INTEGER time;
		time.QuadPart = stop.QuadPart - start.QuadPart;
		return (double)time.QuadPart / (double)frequency.QuadPart;
	}
};