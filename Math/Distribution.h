#pragma once

#include "../Base/STL.h"

CUDA_DEVICE CUDA_HOST unsigned int sampleReuse(float *cdf, unsigned int size, float &sample, float& pdf);

template<int N> struct Distribution1D
{
	CUDA_FUNC_IN Distribution1D()
	{

	}
	Distribution1D(const float *f, unsigned int n)
	{
		if (n > N)
			throw 1;
		m_cdf[0] = 0.0f;
		count = 1;
#define append(pdfValue) {m_cdf[count] = m_cdf[count - 1] + pdfValue; count++;}
		for (unsigned int i = 0; i < n; i++)
			append(f[i])
#undef append
			Normalize();//questionable, removing it would break code
	}
	CUDA_FUNC_IN float operator[](unsigned int entry) const
	{
		return m_cdf[entry + 1] - m_cdf[entry];
	}
	CUDA_FUNC_IN float Normalize()
	{
		m_sum = m_cdf[count - 1];
		if (m_sum > 0.0f)
		{
			m_normalization = 1.0f / m_sum;
			for (unsigned int i = 0; i < count; i++)
				m_cdf[i] *= m_normalization;
			m_cdf[count - 1] = 1.0f;
		}
		else m_normalization = 0.0f;
		return m_sum;
	}
	CUDA_FUNC_IN unsigned int SampleDiscrete(float u, float *pdf = 0) const
	{
		if (count == 1)
		{
			*pdf = 0;
			return 0xffffffff;
		}
		const float *ptr = STL_lower_bound(m_cdf, m_cdf + count, u);
		unsigned int index = min(unsigned int(count - 2U), max(0U, unsigned int(ptr - m_cdf - 1)));
		while (operator[](index) == 0 && index < count - 1)
			++index;
		if (pdf)
			*pdf = operator[](index);
		return index;
	}
	CUDA_FUNC_IN unsigned int SampleReuse(float &sampleValue, float &pdf) const
	{
		unsigned int index = SampleDiscrete(sampleValue, &pdf);
		if (index == 0xffffffff)
			return index;
		sampleValue = (sampleValue - m_cdf[index]) / (m_cdf[index + 1] - m_cdf[index]);
		return index;
	}
public:
	float m_cdf[N];
	float m_sum, m_normalization;
	unsigned int count;
};

template<int NU, int NV> struct Distribution2D
{
	CUDA_FUNC_IN Distribution2D()
	{

	}

	Distribution2D(const float *data, unsigned int nu, unsigned int nv)
	{
		Initialize(data, nu, nv);
	}

	void Initialize(const float *data, unsigned int nu, unsigned int nv)
	{
		this->nu = nu;
		this->nv = nv;
		float marginalFunc[NV];
		for (unsigned int v = 0; v < nv; ++v)
		{
			pConditionalV[v] = Distribution1D<NU>(&data[v*nu], nu);
			marginalFunc[v] = pConditionalV[v].Normalize();
		}
		pMarginal = Distribution1D<NV>(&marginalFunc[0], nv);
		pMarginal.Normalize();
	}

	CUDA_FUNC_IN float Pdf(float u, float v) const
	{
		int iu = math::clamp(Float2Int(u * pConditionalV[0].count), 0, pConditionalV[0].count - 1);
		int iv = math::clamp(Float2Int(v * pMarginal.count), 0, pMarginal.count - 1);
		if (pConditionalV[iv].m_sum * pMarginal.m_sum == 0.f)
			return 0.f;
		return (pConditionalV[iv][iu] * pMarginal[iv]) / (pConditionalV[iv].m_sum * pMarginal.m_sum);
	}
private:
	Distribution1D<NU> pConditionalV[NV];
	Distribution1D<NV> pMarginal;
	unsigned int nu, nv;
};