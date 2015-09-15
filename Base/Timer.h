#pragma once

#include <chrono>

//High resolution timer, measuring in seconds(with fractions)
class InstructionTimer
{
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point stop;
public:
	void StartTimer()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	double EndTimer()
	{
		stop = std::chrono::high_resolution_clock::now();
		return getElapsedTime();
	}
	double getElapsedTime()
	{
		return (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
	}
};