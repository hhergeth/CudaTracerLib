#include "e_RoughTransmittance.h"
#include "..\Math\Spline.h"

e_RoughTransmittance::e_RoughTransmittance(const char* name)
{
	char path[255];
	sprintf(path, "data/microfacet/%s.dat", name);
	InputStream I(path);
	const char header[] = "MTS_TRANSMITTANCE";
	char *fileHeader = (char *) alloca(strlen(header));
	I.Read(fileHeader, (unsigned int)strlen(header));
	if (memcmp(fileHeader, header, strlen(header)) != 0)
		throw 1;
	I >> m_etaSamples;
	I >> m_alphaSamples;
	I >> m_thetaSamples;
	m_transSize = 2 * m_etaSamples * m_alphaSamples * m_thetaSamples;
	m_diffTransSize = 2 * m_etaSamples * m_alphaSamples;
	m_transHost = new float[m_transSize];
	m_diffTransHost = new float[m_diffTransSize];
	I >> m_etaMin;
	I >> m_etaMax;
	I >> m_alphaMin;
	I >> m_alphaMax;
	float *temp = new float[m_transSize + m_diffTransSize];
	I.Read(temp, (unsigned int)(m_transSize + m_diffTransSize) * sizeof(float));
	float *ptr = temp;
	size_t fdrEntry = 0, dataEntry = 0;
	for (size_t i=0; i<2*m_etaSamples; ++i) {
		for (size_t j=0; j<m_alphaSamples; ++j) {
			for (size_t k=0; k<m_thetaSamples; ++k)
				m_transHost[dataEntry++] = *ptr++;
			m_diffTransHost[fdrEntry++] = *ptr++;
		}
	}
	delete[] temp;
	CUDA_MALLOC(&m_transDevice, sizeof(float) * m_transSize);
	CUDA_MALLOC(&m_diffTransDevice, sizeof(float) * m_diffTransSize);
	cudaMemcpy(m_transDevice, m_transHost, sizeof(float) * m_transSize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_diffTransDevice, m_diffTransHost, sizeof(float) * m_diffTransSize, cudaMemcpyHostToDevice);
	if(I.getPos() != I.getFileSize())
		throw 1;
}

float e_RoughTransmittance::Evaluate(float cosTheta, float alpha, float eta) const
{
	float warpedCosTheta = powf(abs(cosTheta), (float) 0.25f),
			  result;

	if (cosTheta < 0)
	{
		cosTheta = -cosTheta;
		eta = 1.0f / eta;
	}
#ifdef ISCUDA
	float *data = m_transDevice;
#else
	float *data = m_transHost;
#endif
	if (eta < 1)
	{
		/* Entering a less dense medium -- skip ahead to the
			second data block */
		data += m_etaSamples * m_alphaSamples * m_thetaSamples;
		eta = 1.0f / eta;
	}

	if (eta < m_etaMin)
		eta = m_etaMin;

	/* Transform the roughness and IOR values into the warped parameter space */
	float warpedAlpha = powf((alpha - m_alphaMin)
			/ (m_alphaMax-m_alphaMin), (float) 0.25f);
	float warpedEta = powf((eta - m_etaMin)
			/ (m_etaMax-m_etaMin), (float) 0.25f);

	result = Spline::evalCubicInterp3D(make_float3(warpedCosTheta, warpedAlpha, warpedEta),
		data, make_uint3((unsigned int)m_thetaSamples, (unsigned int)m_alphaSamples, (unsigned int)m_etaSamples),
		make_float3(0.0f), make_float3(1.0f));

	return MIN((float) 1.0f, MAX((float) 0.0f, result));
}

float e_RoughTransmittance::EvaluateDiffuse(float alpha, float eta) const
{
	float result;
#ifdef ISCUDA
	float *data = m_diffTransDevice;
#else
	float *data = m_diffTransHost;
#endif
	if (eta < 1) {
		/* Entering a less dense medium -- skip ahead to the
			second data block */
		data += m_etaSamples * m_alphaSamples;
		eta = 1.0f / eta;
	}

	if (eta < m_etaMin)
		eta = m_etaMin;

	/* Transform the roughness and IOR values into the warped parameter space */
	float warpedAlpha = powf((alpha - m_alphaMin)
			/ (m_alphaMax-m_alphaMin), (float) 0.25f);
	float warpedEta = powf((eta - m_etaMin)
			/ (m_etaMax-m_etaMin), (float) 0.25f);

	result = Spline::evalCubicInterp2D(make_float2(warpedAlpha, warpedEta), data,
		make_uint2((unsigned int)m_alphaSamples, (unsigned int)m_etaSamples), make_float2(0.0f), make_float2(1.0f));
	return MIN((float) 1.0f, MAX((float) 0.0f,  result));
}

static e_RoughTransmittance m_sObjectsHost[3];
CUDA_CONST e_RoughTransmittance m_sObjectsDevice[3];

void e_RoughTransmittanceManager::StaticInitialize()
{
	m_sObjectsHost[0] = e_RoughTransmittance("beckmann");
	m_sObjectsHost[1] = e_RoughTransmittance("phong");
	m_sObjectsHost[2] = e_RoughTransmittance("ggx");
	cudaMemcpyToSymbol(m_sObjectsDevice, m_sObjectsHost, sizeof(e_RoughTransmittance) * 3);
}

void e_RoughTransmittanceManager::StaticDeinitialize()
{
	for(int i = 0; i < 3; i++)
		m_sObjectsHost[i].Free();
}

float e_RoughTransmittanceManager::Evaluate(MicrofacetDistribution::EType type, float cosTheta, float alpha, float eta)
{
#ifdef ISCUDA
	e_RoughTransmittance* o = m_sObjectsDevice;
#else
	e_RoughTransmittance* o = m_sObjectsHost;
#endif
	return o[(unsigned int)type].Evaluate(cosTheta, alpha, eta);
}

float e_RoughTransmittanceManager::EvaluateDiffuse(MicrofacetDistribution::EType type, float alpha, float eta)
{
#ifdef ISCUDA
	e_RoughTransmittance* o = m_sObjectsDevice;
#else
	e_RoughTransmittance* o = m_sObjectsHost;
#endif
	return o[(unsigned int)type].EvaluateDiffuse(alpha, eta);
}