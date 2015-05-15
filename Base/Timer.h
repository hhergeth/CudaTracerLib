#pragma once

#ifdef ISWINDOWS
#include <Windows.h>
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
#else
#include <time.h>
class cTimer
{
private:
	int frequency;
	int start;
    int stop;
public:
	cTimer()
	{
		frequency = clock_getres( CLOCK_MONOTONIC ) ;
	}
	void StartTimer()
	{
		start = clock_gettime(CLOCK_MONOTONIC);
	}
	double EndTimer()
	{
		stop = clock_gettime(CLOCK_MONOTONIC);
		return getElapsedTime();
	}
	double getElapsedTime()
	{
		return double(stop - start) / double(frequency);
	}
};
#endif