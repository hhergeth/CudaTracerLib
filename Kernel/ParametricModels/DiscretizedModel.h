#pragma once

#include <Math/Vector.h>
#include <Base/CudaRandom.h>
#include <qMatrixHelper.h>

namespace CudaTracerLib {

//computes A^B
template<int A, int B> struct pow_int_compile_type
{
	enum { VAL = A * pow_int_compile_type<A, B - 1>::VAL };
};

template<int A> struct pow_int_compile_type<A, 0>
{
	enum { VAL = 1 };
};

template<int D, int N_BINS_PER_DIM> struct DiscretizedModel
{
	typedef qMatrix<float, D, 1> vec;
	struct bin
	{
		float sum_weight_patch;
		bin()
			: sum_weight_patch(0.0f)
		{

		}
	};
	enum { NUM_CELLS = pow_int_compile_type<N_BINS_PER_DIM, D>::VAL };
	bin bins[NUM_CELLS];
	float sum_weight;

	DiscretizedModel()
		: sum_weight(0.0f)
	{

	}

	DiscretizedModel(CudaRNG& rng, const vec& range_min, const vec& range_max)
		: sum_weight(0.0f)
	{
	}

	struct trainingHelper
	{
		DiscretizedModel<D, N_BINS_PER_DIM>&  model;

		CUDA_FUNC_IN trainingHelper(DiscretizedModel<D, N_BINS_PER_DIM>& m)
			: model(m)
		{

		}

		CUDA_FUNC_IN void addSample(const vec& sample, float weight, const vec& range_min, const vec& range_max)
		{
			model.TrainSample(sample, weight, range_min, range_max);
		}

		CUDA_FUNC_IN void finish(const vec& range_min, const vec& range_max)
		{

		}
	};

	CUDA_FUNC_IN trainingHelper getTrainingHelper()
	{
		return trainingHelper(*this);
	}

	CUDA_FUNC_IN void TrainSample(const vec& sample, float weight, const vec& range_min, const vec& range_max)
	{
		auto idx = getIndex(sample, range_min, range_max);
		if (idx == 0xffffffff || idx >= NUM_CELLS)
			return;
		bins[idx].sum_weight_patch += weight;
		sum_weight += weight;
	}

	CUDA_FUNC_IN vec sample(CudaRNG& rng, const vec& range_min, const vec& range_max, float& pdf) const
	{
		int t1 = (int)(rng.randomFloat() * sum_weight);

		float sum = 0;
		for (int i = 0; i < NUM_CELLS; i++)
		{
			sum += bins[i].sum_weight_patch;
			if (sum > t1)
			{
				auto coords = getCoords(i);
				vec bin_length = (range_max - range_min) / (float)N_BINS_PER_DIM;
				auto cell_min = range_min + diagmat(coords) * bin_length;
				vec x = cell_min;
				for (int j = 0; j < D; j++)
					x(j) += rng.randomFloat() * bin_length(j);
				pdf = this->pdf(x, range_min, range_max);
				return x;
			}
		}
		printf("Illegal state, t1 = %d, N = %d\n", t1, num_samples);
		return vec();
	}

	CUDA_FUNC_IN float pdf(const vec& v, const vec& range_min, const vec& range_max) const
	{
		auto idx = getIndex(v, range_min, range_max);
		if (idx == 0xffffffff || idx >= NUM_CELLS)
			return 0.0f;
		float pdf_bin = float(bins[idx].sum_weight_patch) / sum_weight;
		auto dim_bin = (range_max - range_min);
		float length_bin = 1;
		for (int i = 0; i < D; i++)
			length_bin *= dim_bin(i) / float(N_BINS_PER_DIM);

		return pdf_bin * 1.0f / length_bin;
	}
private:
	CUDA_FUNC_IN static unsigned int getIndex(const vec& v, const vec& range_min, const vec& range_max)
	{
		vec cell_idx = (v - range_min).DivElement(range_max - range_min) * (float)N_BINS_PER_DIM;

		if (cell_idx.max() >= N_BINS_PER_DIM)
			return 0xffffffff;

		unsigned int flattened_idx = 0;
		unsigned int hyper_area = 1;
		for (int i = 0; i < D; i++)
		{
			flattened_idx += (unsigned int)cell_idx(i) * hyper_area;
			hyper_area *= N_BINS_PER_DIM;
		}
		return flattened_idx;
	}
	CUDA_FUNC_IN static vec getCoords(unsigned int flattened_idx)
	{
		vec coords = vec::Zero();
		for (int i = 0; i < D; i++)
		{
			coords(i) = float(flattened_idx % N_BINS_PER_DIM);
			flattened_idx = flattened_idx / N_BINS_PER_DIM;
		}
		return coords;
	}
};

}