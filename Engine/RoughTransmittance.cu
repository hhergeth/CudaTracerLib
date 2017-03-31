#include "RoughTransmittance.h"
#include <Math/Spline.h>
#include <Base/FileStream.h>
#include <Base/CudaMemoryManager.h>

namespace CudaTracerLib {

RoughTransmittance::RoughTransmittance(const std::string& name)
{
	FileInputStream I(name.c_str());
	const char header[] = "MTS_TRANSMITTANCE";
	char *fileHeader = (char *)alloca(strlen(header));
	I.Read(fileHeader, (unsigned int)strlen(header));
	if (memcmp(fileHeader, header, strlen(header)) != 0)
		throw std::runtime_error("Invalid filetype for rough transmittance!");
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
	for (size_t i = 0; i < 2 * m_etaSamples; ++i) {
		for (size_t j = 0; j < m_alphaSamples; ++j) {
			for (size_t k = 0; k < m_thetaSamples; ++k)
				m_transHost[dataEntry++] = *ptr++;
			m_diffTransHost[fdrEntry++] = *ptr++;
		}
	}
	delete[] temp;
	CUDA_MALLOC(&m_transDevice, sizeof(float) * m_transSize);
	CUDA_MALLOC(&m_diffTransDevice, sizeof(float) * m_diffTransSize);
	ThrowCudaErrors(cudaMemcpy(m_transDevice, m_transHost, sizeof(float) * m_transSize, cudaMemcpyHostToDevice));
	ThrowCudaErrors(cudaMemcpy(m_diffTransDevice, m_diffTransHost, sizeof(float) * m_diffTransSize, cudaMemcpyHostToDevice));
	if (I.getPos() != I.getFileSize())
		throw std::runtime_error(__FUNCTION__);
}

void RoughTransmittance::Free()
{
	delete[] m_transHost;
	delete[] m_diffTransHost;
	CUDA_FREE(m_transDevice);
	CUDA_FREE(m_diffTransDevice);
}

float RoughTransmittance::Evaluate(float cosTheta, float alpha, float eta) const
{
	float warpedCosTheta = math::pow(math::abs(cosTheta), (float) 0.25f),
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
	float warpedAlpha = math::pow((alpha - m_alphaMin)
		/ (m_alphaMax - m_alphaMin), (float) 0.25f);
	float warpedEta = math::pow((eta - m_etaMin)
		/ (m_etaMax - m_etaMin), (float) 0.25f);

	result = Spline::evalCubicInterp3D(Vec3f(warpedCosTheta, warpedAlpha, warpedEta),
		data, make_uint3((unsigned int)m_thetaSamples, (unsigned int)m_alphaSamples, (unsigned int)m_etaSamples),
		Vec3f(0.0f), Vec3f(1.0f));

	return min((float) 1.0f, max((float) 0.0f, result));
}

float RoughTransmittance::EvaluateDiffuse(float alpha, float eta) const
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
	float warpedAlpha = math::pow((alpha - m_alphaMin)
		/ (m_alphaMax - m_alphaMin), (float) 0.25f);
	float warpedEta = math::pow((eta - m_etaMin)
		/ (m_etaMax - m_etaMin), (float) 0.25f);

	result = Spline::evalCubicInterp2D(Vec2f(warpedAlpha, warpedEta), data,
		make_uint2((unsigned int)m_alphaSamples, (unsigned int)m_etaSamples), Vec2f(0.0f), Vec2f(1.0f));
	return min((float) 1.0f, max((float) 0.0f, result));
}

static RoughTransmittance m_sObjectsHost[3];
CUDA_CONST RoughTransmittance m_sObjectsDevice[3];

void RoughTransmittanceManager::StaticInitialize(const std::string& a_Path)
{
	m_sObjectsHost[0] = RoughTransmittance(a_Path + "/microfacet/beckmann.dat");
	m_sObjectsHost[1] = RoughTransmittance(a_Path + "/microfacet/phong.dat");
	m_sObjectsHost[2] = RoughTransmittance(a_Path + "/microfacet/ggx.dat");
	ThrowCudaErrors(cudaMemcpyToSymbol(m_sObjectsDevice, m_sObjectsHost, sizeof(RoughTransmittance) * 3));
}

void RoughTransmittanceManager::StaticDeinitialize()
{
	for (int i = 0; i < 3; i++)
		m_sObjectsHost[i].Free();
}

float RoughTransmittanceManager::Evaluate(MicrofacetDistribution::EType type, float cosTheta, float alpha, float eta)
{
#ifdef ISCUDA
	RoughTransmittance* o = m_sObjectsDevice;
#else
	RoughTransmittance* o = m_sObjectsHost;
#endif
	return o[(unsigned int)type].Evaluate(cosTheta, alpha, eta);
}

float RoughTransmittanceManager::EvaluateDiffuse(MicrofacetDistribution::EType type, float alpha, float eta)
{
#ifdef ISCUDA
	RoughTransmittance* o = m_sObjectsDevice;
#else
	RoughTransmittance* o = m_sObjectsHost;
#endif
	return o[(unsigned int)type].EvaluateDiffuse(alpha, eta);
}

}