#pragma once

#include <math.h>
#include <cmath>
#include <Defines.h>
#include <Math/Vector.h>
#include <qMatrixAlgorithms.h>
#include <qMatrixHelper.h>
#include <qLinearSolver.h>
#include <Base/CudaRandom.h>
#include <vector>

namespace CudaTracerLib {

CUDA_FUNC_IN float randomNormal(CudaRNG& rng)
{
	float U1 = rng.randomFloat(), U2 = rng.randomFloat();
	return math::sqrt(-2.0f * math::log(U1)) * cosf(2.0f * PI * U2);
}
template<int D, int K> struct GaussianMixtureModel
{
	typedef qMatrix<float, D, 1> vec;
	typedef qMatrix<float, D, D> mat;

	struct Component
	{
		vec mean;
		mat invCovariance;

		//cached
		float pdf_coeff;
		mat sample_mat;

		Component()
		{
			*this = Component(vec::Zero(), mat::Id());
		}

		CUDA_FUNC_IN Component(const vec& mean, const mat& covariance)
			: mean(mean)
		{
			invCovariance = inv(covariance);
			pdf_coeff = math::pow(2.0f * PI, -0.5f * D) * 1.0f / math::sqrt(norm(covariance));
			choleskyDecomposition(covariance, sample_mat);
		}

		CUDA_FUNC_IN float pdf(const vec& x) const
		{
			qMatrix<float, D, 1> d = x - mean;
			float b = -0.5f * (d.transpose() * invCovariance * d);
			return pdf_coeff * math::exp(b);
		}

		CUDA_FUNC_IN vec sample(CudaRNG& rng) const
		{
			vec x;
			for (int i = 0; i < D; i++)
				x(i) = randomNormal(rng);
			return mean + sample_mat * x;
		}

		static Component Random(CudaRNG& rng, const vec& min, const vec& max)
		{
			vec mean;
			mat coVar;
			for (int i = 0; i < D; i++)
				mean(i) = rng.randomFloat();
			float d = 0.05f;
			vec l = min + (max - min) * d, h = min + (max - min) * (1.0f - d);
			mean = l + mean.MulElement(h - l);
			for (int i = 0; i < D; i++)
				for (int j = 0; j < D; j++)
					coVar(i, j) = rng.randomFloat();
			coVar = coVar.transpose() * coVar;
			//coVar = coVar + coVar;
			//coVar = coVar + float(D) * mat::Id();
			return Component(mean, coVar);
		}
	};

	Component components[K];
	float weights[K];

	GaussianMixtureModel()
	{
		for (int i = 0; i < K; i++)
			weights[i] = 0.0f;
	}

	CUDA_FUNC_IN float pdf(const vec& x, const vec& range_min, const vec& range_max) const
	{
		float p = 0;
		for (int i = 0; i < K; i++)
			p += components[i].pdf(x) * weights[i];
		return p;
	}

	CUDA_FUNC_IN vec sample(CudaRNG& rng, const vec& range_min, const vec& range_max, float& pdf) const
	{
		float U = rng.randomFloat();
		float s = 0;
		for (int i = 0; i < K; i++)
		{
			if (s <= U && U < s + weights[i])
			{
				auto x = components[i].sample(rng);
				pdf = this->pdf(x, range_min, range_max);
				return x;
			}
			s += weights[i];
		}
		return vec();
	}

	static GaussianMixtureModel<D, K> Random(CudaRNG& rng, const vec& mi, const vec& ma)
	{
		GaussianMixtureModel<D, K> res;
		for (int i = 0; i < K; i++)
		{
			res.weights[i] = 1.0f / float(K);
			res.components[i] = Component::Random(rng, mi, ma);
		}
		return res;
	}

	static GaussianMixtureModel<D, K> BatchEM(const vec* samples, int N)
	{
		//compute bounds of samples
		vec mi = vec::Ones() * FLT_MAX, ma = vec::Ones() * -FLT_MAX;
		for (int i = 0; i < N; i++)
		{
			mi = minimize(mi, samples[i]);
			ma = maximize(ma, samples[i]);
		}

		//init initial mixture model based on computed bounds
		CudaRNG rng(174713);
		GaussianMixtureModel<D, K> gmm = Random(rng, mi, ma);

		struct gamma_t
		{
			//for one data point for all komponents
			struct responsibilities
			{
				float gamma[K];
			};
			std::vector<responsibilities> gamma;
			gamma_t(int N)
			{
				gamma = decltype(gamma)(N);
			}
			//data index, komponent index
			float& operator()(int i, int k)
			{
				return gamma[i].gamma[k];
			}
			const float& operator()(int i, int k) const
			{
				return gamma[i].gamma[k];
			}
		};

		gmm.offlineEM<gamma_t>(samples, N);
		return gmm;
	}
protected:
	template<typename gamma_t> CUDA_FUNC_IN gamma_t computeGamma(const vec* samples, int N)
	{
		gamma_t gamma(N);
		//compute gammas, iterate over all data points
		for (int i = 0; i < N; i++)
		{
			//iterate over all komponents
			//sum up all responsibilities
			float sum = 0;
			for (int k = 0; k < K; k++)
			{
				gamma(i, k) = weights[k] * components[k].pdf(samples[i]);
				sum += gamma(i, k);
			}
			//make the computed responsibilities relative by using the sum
			for (int k = 0; k < K; k++)
				gamma(i, k) /= sum;
		}
		return gamma;
	}
	template<typename gamma_t> CUDA_FUNC_IN void updateTheta(const gamma_t& gamma, const vec* samples, int N)
	{
		for (int k = 0; k < K; k++)
		{
			float new_weight = 0;
			vec new_mean = vec::Zero();
			mat new_covariance = mat::Zero();
			for (int i = 0; i < N; i++)
			{
				auto g = gamma(i, k);
				new_weight += g;
				new_mean = new_mean + g * samples[i];
				auto q = samples[i] - components[k].mean;
				new_covariance = new_covariance + g * q * q.transpose();
			}
			if (new_weight != 0)
			{
				auto comp = Component(new_mean / new_weight, new_covariance / new_weight);
				if (!comp.invCovariance.is_finite())
					continue;

				weights[k] = new_weight / N;
				components[k] = comp;
			}
		}
	}
	CUDA_FUNC_IN float computeL(const vec* samples, int N)
	{
		float s = 0;
		for (int i = 0; i < N; i++)
			s += log(pdf(samples[i], vec(), vec()));//at this point we know that this model doesn't need these
		return s;
	}
	template<typename gamma_t> CUDA_FUNC_IN void offlineEM(const vec* samples, int N, int* n_iterations_used = 0)
	{
		int iter = 0;
		const float eps = 0.0001f;
		float L_old, L_new = computeL(samples, N);
		do
		{
			auto gamma = computeGamma<gamma_t>(samples, N);
			updateTheta(gamma, samples, N);
			L_old = L_new;
			L_new = computeL(samples, N);
		} while (math::abs(L_old - L_new) > eps * math::abs(L_new) && iter++ < 1000);
		if(n_iterations_used)
			*n_iterations_used = iter;
	}
};

template <int D, int K> struct OnlineEMGaussianMixtureModel : public GaussianMixtureModel<D, K>
{
	struct SufStat
	{
		float u;
		vec s;
		mat ss;

		CUDA_FUNC_IN SufStat()
		{
			u = 0;
			s.zero();
			ss.zero();
		}

		CUDA_FUNC_IN SufStat(const vec& s_q)
		{
			u = 1.0f;
			s = s_q;
			ss = s_q * s_q.transpose();
		}

		CUDA_FUNC_IN SufStat operator*(float f) const
		{
			SufStat r;
			r.u = u * f;
			r.s = s * f;
			r.ss = ss * f;
			return r;
		}

		CUDA_FUNC_IN SufStat operator+(const SufStat& rhs) const
		{
			SufStat r;
			r.u = u + rhs.u;
			r.s = s + rhs.s;
			r.ss = ss + rhs.ss;
			return r;
		}
	};

	typedef GaussianMixtureModel<D, K> GMM;
	typedef typename GMM::vec vec;
	typedef typename GMM::mat mat;

	SufStat stats[K];
	int num_samples;
	int n_iteration;
	float w_avg;

	enum {N_TRAINING_DATA = 10};

	OnlineEMGaussianMixtureModel()
		: num_samples(0), n_iteration(0), w_avg(1.0f)
	{
	}

	OnlineEMGaussianMixtureModel(CudaRNG& rng, const vec& range_min, const vec& range_max)
		: num_samples(0), n_iteration(0), w_avg(1.0f)
	{
		GaussianMixtureModel<D, K>* gm = this;
		*gm = GaussianMixtureModel<D, K>::Random(rng, range_min, range_max);
		w_avg = 1.0f;
	}

	struct trainingHelper
	{
		OnlineEMGaussianMixtureModel<D, K>&  model;
		vec samples[N_TRAINING_DATA];
		float weights[N_TRAINING_DATA];
		int num_added;

		CUDA_FUNC_IN trainingHelper(OnlineEMGaussianMixtureModel<D, K>& m)
			: model(m), num_added(0)
		{

		}

		CUDA_FUNC_IN void addSample(const vec& sample, float weight, const vec& range_min, const vec& range_max)
		{
			if (weight < 1e-3f)
				return;
			samples[num_added] = sample;
			weights[num_added] = weight;
			num_added++;
			if (num_added == N_TRAINING_DATA)
				finish(range_min, range_max);
		}

		CUDA_FUNC_IN void finish(const vec& range_min, const vec& range_max)
		{
			if(num_added > 0)
				model.Train(samples, weights, num_added);
			num_added = 0;
		}
	};

	CUDA_FUNC_IN trainingHelper getTrainingHelper()
	{
		return trainingHelper(*this);
	}

	CUDA_FUNC_IN void Train(const vec* samples, const float* sample_weights, int N)
	{
		if (N < 3)
			return;
		if (num_samples == 0)
		{
			offlineEM<gamma_t>(samples, N, &n_iteration);
			num_samples = N * n_iteration;
		}
		else TrainBatch(samples, sample_weights, N);
	}
protected:
	struct gamma_t
	{
		//for one data point for all komponents
		struct responsibilities
		{
			float gamma[K];
		};
		responsibilities gamma[N_TRAINING_DATA];
		CUDA_FUNC_IN gamma_t(int N)
		{
		}
		//data index, komponent index
		CUDA_FUNC_IN float& operator()(int i, int k)
		{
			return gamma[i].gamma[k];
		}
		CUDA_FUNC_IN const float& operator()(int i, int k) const
		{
			return gamma[i].gamma[k];
		}
	};
	//OnlineEM for N_TRAINING_DATA samples, N <= N_TRAINING_DATA
	CUDA_FUNC_IN void TrainBatch(const vec* samples, const float* sample_weights, int N)
	{
		if (N == 0)
			return;
		CTL_ASSERT(N <= N_TRAINING_DATA);
		num_samples += N;
		n_iteration++;
		float n = (float)n_iteration;
		const float alpha = 0.7f;
		float ny = math::pow(n, -alpha);

		auto gamma = this->computeGamma<gamma_t>(samples, N);

		for (int j = 0; j < K; j++)
		{
			SufStat u_j_N;
			for (int q = 0; q < N; q++)
				u_j_N = u_j_N + SufStat(samples[q]) * gamma(q, j) * sample_weights[q];
			u_j_N = u_j_N * (1.0f / float(N));
			stats[j] = stats[j] * (1.0f - ny) + u_j_N * ny;
		}
		//update averaged total particle weight
		//guess(!) not from the paper clear what this is supposed to do (see equation (8))
		float w_q = 0;
		for (int i = 0; i < N; i++)
			w_q += sample_weights[i];
		w_avg = (1 - ny) * w_avg + ny * w_q / N;

		const float a = 2.01f, b = 5.0f * 1e-4f, v = 1.01f;
		mat b_nI = b / float(n) * mat::Id();
		for (int j = 0; j < K; j++)
		{
			weights[j] = (stats[j].u / w_avg + (v - 1) / float(n)) / (1 + K * (v - 1) / float(n));
			vec mu = stats[j].s / stats[j].u;
			mat A = stats[j].s * mu.transpose() + mu * stats[j].s.transpose();
			mat B = mu * mu.transpose();
			mat sigma = (b_nI + (stats[j].ss - A + stats[j].u * B) / w_avg) / ((a - 2) / float(n) + stats[j].u / w_avg);
			if (!sigma.is_symmetric())
				sigma.id();
			components[j] = Component(mu, sigma);
		}
	}
};

}