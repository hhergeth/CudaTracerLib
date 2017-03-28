#pragma once

#include <vector>
#include <Math/Kernel.h>

namespace CudaTracerLib {

template<int DIM> struct DensityEstimator
{
	std::vector<size_t> data;
	int n_bins_per_dim;

	DensityEstimator(int n_bins_per_dim)
		: n_bins_per_dim(n_bins_per_dim), data((size_t)pow_int_compile<DIM>::pow(n_bins_per_dim), 0)
	{
	}

	void add(const Vector<float, DIM>& p)
	{
		size_t idx = 0;
		size_t q = 1;
		for (int i = 0; i < DIM; i++)
		{
			idx += q * (size_t)(p[i] * n_bins_per_dim);
			q *= n_bins_per_dim;
		}
		data[idx]++;
	}

	void printResult()
	{
		size_t N = 0;
		for (auto v : data)
			N += v;

		auto avg_vals_per_cell = N / data.size();
		for (size_t i = 0; i < data.size(); i++)
		{
			auto v = math::clamp((float)data[i] / avg_vals_per_cell, 0.0f, 2.0f) * 50.0f;
			std::cout << math::Floor2Int(v) << ", ";
			if (DIM == 2 && (i + 1) % n_bins_per_dim == 0)
				std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};

}