#pragma once

#include <chrono>
#include <vector>
#include <map>
#include <sstream>
#include "Platform.h"
#include <cmath>

namespace CudaTracerLib {

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
	//Returns the elapsed time in seconds
	double EndTimer()
	{
		stop = std::chrono::high_resolution_clock::now();
		return getElapsedTime();
	}
	//Returns the elapsed time in seconds
	double getElapsedTime()
	{
		return (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
	}
};

//Abstraction class which helps measuring performance
class PerformanceTimer
{
public:
	struct BlockMeasure
	{
	private:
		PerformanceTimer& ref;
	public:
		std::string id;
		InstructionTimer timer;
		BlockMeasure(PerformanceTimer& r, const std::string& id)
			: ref(r), id(id)
		{
			timer.StartTimer();
		}
		~BlockMeasure()
		{
			auto sec = timer.EndTimer();
			ref.addTiming(id, sec);
		}
	};
	PerformanceTimer()
	{
	}
	BlockMeasure StartBlock(const std::string& id)
	{
		return BlockMeasure(*this, id);
	}
	const std::vector<double>& getTimings(const std::string& id, double& avg, double& stdDev) const
	{
		if (mTimings.count(id) == 0)
			throw std::runtime_error("Id not found in timings!");
		const auto& timings = mTimings.find(id)->second;
		avg = 0; stdDev = 0;
		for (auto v : timings)
		{
			avg += v;
			stdDev += v * v;
		}
		avg /= timings.size(); stdDev /= timings.size();
		stdDev = std::sqrt(stdDev - avg * avg);
		return timings;
	}
	std::string ToString(bool print_data = false) const
	{
		double avg, stdDev;
		double avg_sum = 0;
		for (auto& o : mTimings)
		{
			getTimings(o.first, avg, stdDev);
			avg_sum += avg;
		}

		std::ostringstream oss;
		oss << "[" << std::endl;
		for (auto& o : mTimings)
		{
			const auto& ti = getTimings(o.first, avg, stdDev);
			oss << format("%10s : %2.f%% {Avg = %3.2f[s], o = %3.2f[s]}", o.first.c_str(), avg / avg_sum * 100, avg, stdDev);
			if (print_data)
			{
				oss << " {";
				for (size_t i = 0; i < ti.size(); i++)
					oss << (i ? ", " : "") << format("%3.2f", ti[i]);
				oss << "}";
			}
			oss << std::endl;
		}
		oss << "]" << std::endl;
		return oss.str();
	}
	static PerformanceTimer& getInstance(const std::string& id)
	{
		return gStaticInstances[id];
	}
private:
	void addTiming(const std::string& id, double sec)
	{
		if (mTimings.count(id) == 0)
			mTimings[id] = std::vector<double>();
		auto& timings = mTimings[id];
		timings.push_back(sec);
	}

	std::map<std::string, std::vector<double>> mTimings;

	static std::map<std::string, PerformanceTimer> gStaticInstances;
};

#define GET_PER_BLOCKS() PerformanceTimer::getInstance(typeid(*this).name())

#define START_PERF_BLOCK(ID) GET_PER_BLOCKS().StartBlock(ID)

}
