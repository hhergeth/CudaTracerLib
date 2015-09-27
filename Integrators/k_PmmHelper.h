#pragma once

#include <math.h>
#include <cmath>
#include "../Defines.h"
#include "../MathTypes.h"
#include <qMatrixAlgorithms.h>
#include <qMatrixHelper.h>
#include "../Base/CudaRandom.h"

CUDA_FUNC_IN float randomNormal(CudaRNG& rng)
{
	return math::sqrt(-2.0f * logf(rng.randomFloat())) * cosf(2.0f * PI * rng.randomFloat());
}

CUDA_FUNC_IN float ny(int i)
{
	const float alpha = 0.75f;
	return math::pow(float(i), -alpha);
}

CUDA_FUNC_IN qMatrix<float, 2, 1> HemishphereToSquare(const Vec3f& d)
{
	Vec3f q = normalize(d);
	Vec2f a = Warp::uniformDiskToSquareConcentric(Vec2f(d.x, d.y));
	return VEC<float, 2>() % a.x % a.y;
}

CUDA_FUNC_IN Vec3f SquareToHemisphere(const qMatrix<float, 2, 1>& s)
{
	Vec2f a = Warp::squareToUniformDiskConcentric(Vec2f(s(0, 0), s(1, 0)));
	return Vec3f(a.x, a.y, 1.0f - a.x * a.x - a.y * a.y);
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
		float a;
		mat Q;

		CUDA_FUNC_IN Component(){}

		CUDA_FUNC_IN Component(const vec& mean, const mat& covariance)
			: mean(mean)
		{
			invCovariance = inv(covariance);
			a = 1.0f / (math::pow(2.0f * PI, float(D) / 2.0f) * math::sqrt(det(covariance)));
			mat eigVectors;
			vec eigValues;
			qrAlgorithmSymmetric(covariance, eigValues, eigVectors);
			Q = diagmat<vec>(eigValues.sqrt()) * eigVectors;
		}

		CUDA_FUNC_IN float p(const vec& x) const
		{
			qMatrix<float, D, 1> d = x - mean;
			float b = -0.5f * (d.transpose() * invCovariance * d)(0, 0);
			return a * math::exp(b);
		}

		CUDA_FUNC_IN vec sample(CudaRNG& rng) const
		{
			vec x;
			for(int i = 0; i < D; i++)
				x(i, 0) = randomNormal(rng);
			return Q * x + mean;
		}

		CUDA_FUNC_IN static Component Random(CudaRNG& rng, const vec& min, const vec& max)
		{
			vec mean;
			mat coVar;
			for(int i = 0; i < D; i++)
				mean(i, 0) = rng.randomFloat();
			float d = 0.2f;
			vec l = min + (max - min) * d, h = min + (max - min) * (1.0f - d);
			mean = l + mean.MulElement(h - l);
			for(int i = 0; i < D; i++)
				for(int j = 0; j < D; j++)
					coVar(i, j) = rng.randomFloat();
			coVar = coVar + coVar;
			coVar = coVar + float(D) * mat::Id();
			return Component(mean, coVar);
		}
	};

	Component components[K];
	float weights[K];

	CUDA_FUNC_IN GaussianMixtureModel()
	{

	}

	CUDA_FUNC_IN float p(const vec& x) const
	{
		float p = 0;
		for(int i = 0; i < K; i++)
			p += components[i].p(x) * weights[i];
		return p;
	}

	CUDA_FUNC_IN vec sample(CudaRNG& rng) const
	{
		float U = rng.randomFloat();
		float s = 0;
		for(int i = 0; i < K; i++)
		{
			if(s <= U && U < s + weights[i])
				return components[i].sample(rng);
			s += weights[i];
		}
		return vec();
	}

	CUDA_FUNC_IN float L(const qMatrix<float, D, 1>* samples, int N) const
	{
		float s = 0;
		for(int i = 0; i < N; i++)
			s += log(p(samples[i]));
		return s;
	}

	CUDA_FUNC_IN static GaussianMixtureModel<D, K> Random(CudaRNG& rng, const vec& mi, const vec& ma)
	{
		GaussianMixtureModel<D, K> res;
		for(int i = 0; i < K; i++)
		{
			res.weights[i] = 1.0f / float(K);
			res.components[i] = Component::Random(rng, mi, ma);
		}
		return res;
	}

	struct SufStat
	{
		float u;
		vec s;
		mat ss;
		CUDA_FUNC_IN SufStat()
		{
#ifndef ISCUDA
			zero();
#endif
		}
		CUDA_FUNC_IN void zero()
		{
			u = 0;
			s.zero();
			ss.zero();
		}
		CUDA_FUNC_IN SufStat(const vec& s_q, float gamma)
		{
			u = gamma;
			s = s_q * gamma;
			ss = s_q * s_q.transpose() * gamma;
		}
		CUDA_FUNC_IN SufStat operator*(float f) const
		{
			SufStat s;
			s.u = u * f;
			s.s = this->s * f;
			s.ss = ss * f;
			return s;
		}
		CUDA_FUNC_IN SufStat operator+(const SufStat& rhs) const
		{
			SufStat s;
			s.u = u + rhs.u;
			s.s = this->s + rhs.s;
			s.ss = ss + rhs.ss;
			return s;
		}
		CUDA_FUNC_IN void mul(float f)
		{
			u = u * f;
			s = s * f;
			ss = ss * f;
		}
		CUDA_FUNC_IN void add(const SufStat& stat)
		{
			u = u + stat.u;
			s = s + stat.s;
			ss = ss + stat.ss;
		}
	};
	
	CUDA_FUNC_IN void RecomputeFromStats(const SufStat* stats, int n, float w)
	{
		const float a = 2.01f, b = 5.0f * math::pow(10, -4), v = 1.01f;
		mat b_nI = b / float(n) * mat::Id();
		for(int j = 0; j < K; j++)
		{
			weights[j] = (stats[j].u / w + (v - 1) / float(n)) / (1 + K * (v - 1) / float(n));
			vec mu = stats[j].s / stats[j].u;
			mat A = stats[j].s * mu.transpose() + mu * stats[j].s.transpose();
			mat B = mu * mu.transpose();
			mat sigma = (b_nI + (stats[j].ss - A + stats[j].u * B) / w) / ((a - 2) / float(n) + stats[j].u / w);
			components[j] = Component(mu, sigma);
		}
	}

	template<int max_SAMPLES> CUDA_FUNC_IN void OnlineEM(SufStat* stats, const vec* samples, int N, float ny, int n_all, float w_all)
	{
		float gamma[max_SAMPLES][K];
		for(int q = 0; q < N; q++)
		{
			float sumRes = 0;
			for(int h = 0; h < K; h++)
				sumRes += weights[h] * components[h].p(samples[q]);

			for(int j = 0; j < K; j++)
				gamma[q][j] = weights[j] * components[j].p(samples[q]) / sumRes;
		}
		for(int j = 0; j < K; j++)
		{
			SufStat s;
			s.zero();
			for(int q = 0; q < N; q++)
				s.add(SufStat(samples[q], gamma[q][j]));
			s.mul(1.0f / float(N));
			stats[j] = stats[j] * (1.0f - ny) + s * ny;
		}
		RecomputeFromStats(stats, n_all, w_all);
	}

	static GaussianMixtureModel<D, K> BatchEM(const vec* samples, int N)
	{
		vec mi = vec::Ones() * FLT_MAX, ma = vec::Ones() * -FLT_MAX;
		for(int i = 0; i < N; i++)
		{
			mi = minimize(mi, samples[i]);
			ma = maximize(ma, samples[i]);
		}

		const float eps = 0.01f;
		CudaRNG rng;
		rng.Initialize(0, 1234, 0);
		GaussianMixtureModel<D, K> res = Random(rng);
		std::vector<std::vector<float>> gamma;
		for(int i = 0; i < N; i++)
		{
			gamma.push_back(std::vector<float>());
			gamma[i].resize(K);
		}
		std::vector<SufStat> stats;
		stats.resize(K);
		float L_old, L_new = res.L(samples, N);
		do
		{
			L_old = L_new;
			for(int q = 0; q < N; q++)
			{
				float sumRes = 0;
				for(int h = 0; h < K; h++)
					sumRes += res.weights[h] * res.components[h].p(samples[q]);

				for(int j = 0; j < K; j++)
					gamma[q][j] = res.weights[j] * res.components[j].p(samples[q]) / sumRes;
			}
			for(int j = 0; j < K; j++)
			{
				SufStat s;
				for(int q = 0; q < N; q++)
					s.add(SufStat(samples[q], gamma[q][j]));
				s.mul(1.0f / float(N));
				stats[j] = s;
			}
			res.RecomputeFromStats(&stats[0], N, 1);
			L_new = res.L(samples, N);
		}
		//while(I++ < 100);
		while (math::abs(L_old - L_new) > eps * math::abs(L_new));
		return res;
	}
};

struct SpatialEntry
{
	Vec3f wi;
	CUDA_FUNC_IN SpatialEntry(const Vec3f& wi)
		: wi(wi)
	{
	}
};

struct DirectionModel
{
	static const int K = 4;

	GaussianMixtureModel<2, K> gmm;
	GaussianMixtureModel<2, K>::SufStat stats[K];
	int numSamples;

	DirectionModel()
		: numSamples(0)
	{
	}

	CUDA_FUNC_IN void Initialze(CudaRNG& rng)
	{
		gmm = GaussianMixtureModel<2, K>::Random(rng, VEC<float, 2>() % 0 % 0, VEC<float, 2>() % 1 % 1);
	}

	template<int max_SAMPLES> CUDA_FUNC_IN void Update(const e_SpatialLinkedMap<SpatialEntry>& sMap, const Vec3f& mi, const Vec3f& ma, float ny)
	{
		qMatrix<float, 2, 1> samples[max_SAMPLES];
		int N = 0;
		for(e_SpatialLinkedMap<SpatialEntry>::iterator it = sMap.begin(mi, ma); it != sMap.end(mi, ma); ++it)
		{
			samples[N++] = HemishphereToSquare(it->wi);
			if(N == max_SAMPLES)
				break;
		}
		numSamples += N;
		if(N)
			gmm.OnlineEM<max_SAMPLES>(stats, samples, N, ny, numSamples, 1);
	}

	CUDA_FUNC_IN Vec3f Sample(CudaRNG& rng)
	{
		qMatrix<float, 2, 1> s = gmm.sample(rng);
		return SquareToHemisphere(s);
	}
};

void plotModel(const DirectionModel& model);