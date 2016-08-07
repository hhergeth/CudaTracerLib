#pragma once
#include <Base/ValuePack.h>
#include "MathFunc.h"

namespace CudaTracerLib {
	
template<int N_BINS, typename PACK_TYPE> class MergingModel
{
private:
	//typedef ValuePack<ARGS...> PACK_TYPE;

	static CUDA_FUNC_IN float round(float val)
	{
		int s = val < 0 ? -1 : (val > 0 ? 1 : 0);
		return s * floorf(fabsf(val) + 0.5f);
	}

	struct bin
	{
		PACK_TYPE s_x, s_x2;
		int N;
		float t_p;

		CUDA_FUNC_IN bin()
		{

		}

		CUDA_FUNC_IN bin(float t, int n, const PACK_TYPE& s_x, const PACK_TYPE& s_x2)
			: s_x(s_x), s_x2(s_x2), N(n), t_p(t)
		{
		}
		CUDA_FUNC_IN bin add(const PACK_TYPE& v) const
		{
			return bin(t_p, N + 1, s_x + v, s_x2 + v * v);
		}
		CUDA_FUNC_IN static bin Merge(const bin& b1, const bin& b2)
		{
			return bin(fmax(b1.t_p, b2.t_p), b1.N + b2.N, b1.s_x + b2.s_x, b1.s_x2 + b2.s_x2);
		}
		CUDA_FUNC_IN PACK_TYPE E_X() const
		{
			return N == 0 ? PACK_TYPE::Zero() : s_x * (1.0f / float(N));
		}
		CUDA_FUNC_IN PACK_TYPE Var_X() const
		{
			return N < 2 ? PACK_TYPE::Zero() : s_x2 * (1.0f / float(N)) - (E_X() * E_X());
		}
		CUDA_FUNC_IN void Split(float t_s, float t, const PACK_TYPE& v, bin& left, bin& right) const
		{
			float old_bin_width = t_p - t_s;
			int nl = (int)max(1.0f, round(((t - t_s) / old_bin_width) * N)), nr = (int)max(1.0f, round(((t_p - t) / old_bin_width) * N));
			PACK_TYPE E_X = this->E_X(), E_X2 = N == 0 ? PACK_TYPE::Zero() : s_x2 * (1.0f / float(N));
			left = bin(t, nl + 1, E_X * (float)nl + v, E_X2 * (float)nl + v * v);//new bin
			right = bin(t_p, nr, E_X * (float)nr, E_X2 * (float)nr);//old bin
		}

		CUDA_FUNC_IN PACK_TYPE eval(float t) const
		{
			return E_X();
		}
	};

	bin bins[N_BINS];
public:
	
	typedef PACK_TYPE _PACK_TYPE;
	CUDA_FUNC_IN MergingModel()
	{
		auto z = PACK_TYPE::Zero();
		for (int i = 0; i < N_BINS; i++)
			bins[i] = bin((float)(i + 1) / (float)(N_BINS), 0, z, z);
	}

	template<typename F> CUDA_FUNC_IN MergingModel(const F& clb)
	{
		for (int i = 0; i < N_BINS; i++)
		{
			auto z = clb(i);
			bins[i] = bin((float)(i + 1) / (float)(N_BINS), 1, z, z * z);
		}
	}

	CUDA_FUNC_IN void Train(float t, const PACK_TYPE& v)
	{
		t = min(t, 1.0f);
		int idx = findIdx(t);
		if (idx == -1)
			return;

		bin left, right;
		bins[idx].Split(idx == 0 ? 0 : bins[idx - 1].t_p, t, v, left, right);
		auto best_merge = FindBestMerge(idx, v, left, right);
		if (best_merge == -1)
		{
			++bins[idx].N;
			bins[idx].s_x = bins[idx].s_x + v;
			bins[idx].s_x2 = bins[idx].s_x2 + v * v;
		}
		else
		{
			if (idx < best_merge)
			{
				bins[best_merge + 1] = bin::Merge(bins[best_merge], bins[best_merge + 1]);
				for (int i = best_merge - 1; i > idx; i--)
					bins[i + 1] = bins[i];
				for (int i = 0; i < N_BINS; i++)
					if (i != idx + 1 && bins[i].t_p >= t)
					{
						auto tmp_bin = bins[i];
						tmp_bin.Split(i > 0 ? bins[i - 1].t_p : 0, t, v, bins[i], bins[i + 1]);
						break;
					}
			}
			else if (best_merge + 1 < idx)
			{
				bins[best_merge] = bin::Merge(bins[best_merge], bins[best_merge + 1]);
				for (int i = best_merge + 1; i < idx - 1; i++)
					bins[i] = bins[i + 1];
				for (int i = 0; i < N_BINS; i++)
					if (i != idx - 1 && bins[i].t_p >= t)
					{
						auto tmp_bin = bins[i];
						tmp_bin.Split(i > 0 ? bins[i - 1].t_p : 0, t, v, bins[i - 1], bins[i]);
						break;
					}
			}
			else
			{
				auto tmp_bin = bin::Merge(bins[best_merge], bins[best_merge + 1]);
				tmp_bin.Split(best_merge > 0 ? bins[best_merge - 1].t_p : 0, t, v, bins[best_merge], bins[best_merge + 1]);
			}
		}
	}

	CUDA_FUNC_IN PACK_TYPE Eval(float t) const
	{
		for (int i = 0; i < N_BINS; i++)
			if (bins[i].t_p >= t)
				return bins[i].eval(t);
		return PACK_TYPE::Zero();
	}

	CUDA_FUNC_IN PACK_TYPE Average(float t_min, float t_max) const
	{
		float sum_w = 0;
		PACK_TYPE sum_val = PACK_TYPE::Zero();
		int start_i = findIdx(t_min), end_i = findIdx(t_max);
		if (start_i == -1) return PACK_TYPE::Zero();
		end_i = end_i == -1 ? N_BINS - 1 : end_i;
		float t = t_min;
		for (int i = start_i; i <= end_i; i++)
		{
			float t_i = bins[i].t_p;
			float w = t_i < t_max ? t_i - t : t_max - t;
			t = t_i;
			sum_w += w;
			sum_val = sum_val + bins[i].E_X() * w;
		}
		return sum_w != 0 ? sum_val * (1.0f / sum_w) : PACK_TYPE::Zero();
	}

	CUDA_FUNC_IN PACK_TYPE Average() const
	{
		PACK_TYPE sum = PACK_TYPE::Zero();
		for (int i = 0; i < N_BINS; i++)
			sum = sum + bins[i].E_X() * (bins[i].t_p - (i > 0 ? bins[i - 1].t_p : 0));
		return sum;
	}

	template<typename F> CUDA_FUNC_IN void Iterate(const F& clb) const
	{
		for (int i = 0; i < N_BINS; i++)
			clb(bins[i].E_X());
	}

	CUDA_FUNC_IN void Scale(float f)
	{
		for (int i = 0; i < N_BINS; i++)
		{
			bins[i].s_x = bins[i].s_x * f;
			bins[i].s_x2 = bins[i].s_x2 * (f * f);
		}
	}

	template<typename T, typename F> CUDA_FUNC_IN T Extremize(float t_min, float t_max, const T& init, const F& max_clb) const
	{
		int start_i = findIdx(t_min), end_i = findIdx(t_max);
		if (start_i == -1) return init;
		end_i = end_i == -1 ? N_BINS - 1 : end_i;
		T max_el = init;
		for (int i = start_i; i <= end_i; i++)
			max_el = max_clb(max_el, bins[i].E_X());
		return max_el;
	}
private:
	CUDA_FUNC_IN int findIdx(float t) const
	{
		int idx = 0;
		while (idx < N_BINS && t > bins[idx].t_p)
			idx++;
		if (idx == N_BINS || bins[idx].t_p < t)
			return -1;
		else return idx;
	}
	CUDA_FUNC_IN int FindBestMerge(int idx, const PACK_TYPE& v, const bin& left, const bin& right) const
	{
		int best_idx = -1;
		float best_merge = FLT_MAX;
		float left_right_var = PACK_TYPE(left.Var_X() * (float)left.N + right.Var_X() * (float)right.N).Sum(),
			idx_var = PACK_TYPE(bins[idx].add(v).Var_X() * (float)(bins[idx].N + 1)).Sum();
		for (int i = 0; i < N_BINS - 1; i++)
		{
			auto mergedBin = bin::Merge(bins[i], bins[i + 1]);
			float var_merge = PACK_TYPE(mergedBin.Var_X() * (float)mergedBin.N).Sum() + left_right_var;
			float var_keep = PACK_TYPE(bins[i].Var_X() * (float)bins[i].N + bins[i + 1].Var_X() * (float)bins[i + 1].N).Sum() + idx_var;
			if (var_merge < var_keep && var_merge < best_merge)
			{
				best_merge = var_merge;
				best_idx = i;
			}
		}
		return best_idx;

	}
};

}