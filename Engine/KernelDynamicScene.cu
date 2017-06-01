#include "KernelDynamicScene.h"
#include <Kernel/TraceHelper.h>
#include <SceneTypes/Light.h>
#include <Base/STL.h>

namespace CudaTracerLib {

const Light* KernelDynamicScene::getLight(unsigned int idx) const
{
	return m_sLightBuf.Data + m_pLightIndices[idx];
}

const Light* KernelDynamicScene::getLight(const TraceResult& tr) const
{
	return m_sLightBuf.Data + tr.LightIndex();
}

const Light* KernelDynamicScene::getEnvironmentMap() const
{
	if (m_uEnvMapIndex == UINT_MAX)
		return 0;
	else return m_sLightBuf.Data + m_uEnvMapIndex;
}

const Light* KernelDynamicScene::sampleEmitter(float& emPdf, Vec2f& sample) const
{
	if (m_numLights == 0)
		return 0;
	unsigned int idx = (unsigned int)(STL_upper_bound(m_pLightCDF, m_pLightCDF + m_numLights, sample.x) - m_pLightCDF);
	//unsigned int idx = (unsigned int)(m_sLightData.UsedCount * sample.x);
	CTL_ASSERT(idx < m_numLights);
	if (idx >= m_numLights)
		idx = m_numLights - 1;
	//sample.x = sample.x - idx / float(m_sLightData.UsedCount);
	float fU = m_pLightCDF[idx], fL = idx > 0 ? m_pLightCDF[idx - 1] : 0.0f;
	sample.x = (sample.x - fL) / (fU - fL);
	//emPdf = 1.0f / float(m_sLightData.UsedCount);
	emPdf = fU - fL;
	return getLight(idx);
}

float KernelDynamicScene::pdfEmitter(const Light* L) const
{
	unsigned int idx = (unsigned int)(L - m_sLightBuf.Data);
	return m_pLightCDF[idx] - (idx == 0 ? 0.0f : m_pLightCDF[idx - 1]);
}

float KernelDynamicScene::pdfEmitterDiscrete(const Light *emitter) const
{
	unsigned int idx = (unsigned int)(emitter - m_sLightBuf.Data);
	return m_pLightPDF[idx];
}

Spectrum KernelDynamicScene::EvalEnvironment(const Ray& r) const
{
	const Light* l = getEnvironmentMap();
	if (l)
		return l->As<InfiniteLight>()->evalEnvironment(r);
	else return 0.0f;
}

Spectrum KernelDynamicScene::EvalEnvironment(const Ray& r, const Ray& rX, const Ray& rY) const
{
	const Light* l = getEnvironmentMap();
	if (l)
		return l->As<InfiniteLight>()->evalEnvironment(r, rX, rY);
	else return 0.0f;
}

bool KernelDynamicScene::Occluded(const Ray& r, float tmin, float tmax, TraceResult* res) const
{
	//remember this is an occluded test, so we shrink the interval!
	TraceResult r2 = traceRay(r);
	if (r2.hasHit() && res)
		*res = r2;
	bool end = r2.m_fDist < tmax - MIN_RAYTRACE_DISTANCE;
	if (isinf(tmax) && !r2.hasHit())
		end = false;
	return r2.m_fDist > tmin + MIN_RAYTRACE_DISTANCE && end;
}

Spectrum KernelDynamicScene::evalTransmittance(const Vec3f& p1, const Vec3f& p2) const
{
	Vec3f d = p2 - p1;
	float l = d.length();
	d /= l;
	return (-m_sVolume.tau(Ray(p1, d), 0, l)).exp();
}

Spectrum KernelDynamicScene::sampleEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &_sample) const
{
	dRec.pdf = 0;
	dRec.object = 0;

	Vec2f sample = _sample;
	float emPdf;
	const Light *emitter = sampleEmitter(emPdf, sample);
	if (emitter == 0)
		return 0.0f;
	Spectrum value = emitter->sampleDirect(dRec, sample);
	if (dRec.pdf != 0)
	{
		dRec.pdf *= emPdf;
		value /= emPdf;
		dRec.object = emitter;
		return value;
	}
	else return Spectrum(0.0f);
}

Spectrum KernelDynamicScene::sampleAttenuatedEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &_sample) const
{
	Spectrum value = sampleEmitterDirect(dRec, _sample);
	return value * evalTransmittance(dRec.ref, dRec.p);
}

Spectrum KernelDynamicScene::sampleSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Spectrum value = m_Camera.sampleDirect(dRec, sample);
	if (dRec.pdf != 0)
	{
		dRec.object = &g_SceneData;
		return value;
	}
	else
	{
		return Spectrum(0.0f);
	}
}

Spectrum KernelDynamicScene::sampleAttenuatedSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Spectrum value = sampleSensorDirect(dRec, sample);
	return value * evalTransmittance(dRec.ref, dRec.p);
}

float KernelDynamicScene::pdfEmitterDirect(const DirectSamplingRecord &dRec) const
{
	const Light *emitter = (Light*)dRec.object;
	return emitter->pdfDirect(dRec) * pdfEmitterDiscrete(emitter);
}

float KernelDynamicScene::pdfSensorDirect(const DirectSamplingRecord &dRec) const
{
	return m_Camera.pdfDirect(dRec);
}

Spectrum KernelDynamicScene::sampleEmitterPosition(PositionSamplingRecord &pRec, const Vec2f &_sample) const
{
	Vec2f sample = _sample;
	float emPdf;
	const Light *emitter = sampleEmitter(emPdf, sample);

	Spectrum value = emitter->samplePosition(pRec, sample);

	pRec.object = emitter;
	pRec.pdf *= emPdf;

	return value / emPdf;
}

Spectrum KernelDynamicScene::sampleSensorPosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	pRec.object = &m_Camera;
	return m_Camera.samplePosition(pRec, sample, extra);
}

float KernelDynamicScene::pdfEmitterPosition(const PositionSamplingRecord &pRec) const
{
	const Light *emitter = (const Light*)pRec.object;
	return emitter->pdfPosition(pRec) * pdfEmitterDiscrete(emitter);
}

float KernelDynamicScene::pdfSensorPosition(const PositionSamplingRecord &pRec) const
{
	const Sensor *sensor = (const Sensor*)pRec.object;
	return sensor->pdfPosition(pRec);
}

Spectrum KernelDynamicScene::sampleEmitterRay(NormalizedT<Ray>& ray, const Light*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	Vec2f sample = spatialSample;
	float emPdf;
	emitter = sampleEmitter(emPdf, sample);

	return emitter->sampleRay(ray, sample, directionalSample) / emPdf;
}

Spectrum KernelDynamicScene::sampleSensorRay(NormalizedT<Ray>& ray, const Sensor*& sensor, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	sensor = &m_Camera;
	return sensor->sampleRay(ray, spatialSample, directionalSample);
}

}
