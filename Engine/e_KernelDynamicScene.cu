#include "e_KernelDynamicScene.h"
#include "..\Kernel\k_TraceHelper.h"
#include "e_Light.h"

const e_KernelLight* e_KernelDynamicScene::sampleLight(float& emPdf, Vec2f& sample) const
{
	unsigned int index = m_emitterPDF.SampleReuse(sample.x, emPdf);
	if (index == 0xffffffff)
		return 0;
	return m_sLightData.Data + g_SceneData.m_uEmitterIndices[index];
}

float e_KernelDynamicScene::pdfLight(const e_KernelLight* light)
{
	unsigned int index = light - this->m_sLightData.Data;
	return m_emitterPDF[index];
}

bool e_KernelDynamicScene::Occluded(const Ray& r, float tmin, float tmax, TraceResult* res) const
{
	const float eps = 0.01f;//remember this is an occluded test, so we shrink the interval!
	TraceResult r2 = k_TraceRay(r);
	if (r2.hasHit() && res)
		*res = r2;
	return r2.m_fDist > tmin * (1.0f + eps) && r2.m_fDist < tmax * (1.0f - eps);
	//return tmin < r2.m_fDist && r2.m_fDist < tmax;
}

Spectrum e_KernelDynamicScene::EvalEnvironment(const Ray& r) const
{
	if(m_uEnvMapIndex != 0xffffffff)
		return m_sLightData[m_uEnvMapIndex].As<e_InfiniteLight>()->evalEnvironment(r);
	else return Spectrum(0.0f);
}

Spectrum e_KernelDynamicScene::EvalEnvironment(const Ray& r, const Ray& rX, const Ray& rY) const
{
	if (m_uEnvMapIndex != 0xffffffff)
		return m_sLightData[m_uEnvMapIndex].As<e_InfiniteLight>()->evalEnvironment(r, rX, rY);
	else return Spectrum(0.0f);
}

float e_KernelDynamicScene::pdfEmitterDiscrete(const e_KernelLight *emitter) const
{
	unsigned int index = emitter - m_sLightData.Data;
	return m_emitterPDF[index];
}

Spectrum e_KernelDynamicScene::sampleEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &_sample) const
{
	Vec2f sample = _sample;
	float emPdf;
	const e_KernelLight *emitter = sampleLight(emPdf, sample);
	if (emitter == 0)
	{
		dRec.pdf = 0;
		dRec.object = 0;
		return 0.0f;
	}
	Spectrum value = emitter->sampleDirect(dRec, sample);
	if (dRec.pdf != 0)
	{
		/*if (testVisibility && Occluded(Ray(dRec.ref, dRec.n), 0, dRec.dist))
		{
			dRec.object = 0;
			return Spectrum(0.0f);
		}*/
		dRec.pdf *= emPdf;
		value /= emPdf;
		dRec.object = emitter;
		return value;
	}
	else
	{
		return Spectrum(0.0f);
	}
}

Spectrum e_KernelDynamicScene::sampleSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Spectrum value = m_Camera.sampleDirect(dRec, sample);
	if (dRec.pdf != 0)
	{
		/*if (testVisibility && Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
		{
			dRec.object = 0;
			return Spectrum(0.0f);
		}*/
		dRec.object = &g_SceneData;
		return value;
	}
	else
	{
		return Spectrum(0.0f);
	}
}

float e_KernelDynamicScene::pdfEmitterDirect(const DirectSamplingRecord &dRec) const
{
	const e_KernelLight *emitter = (e_KernelLight*)dRec.object;
	return emitter->pdfDirect(dRec) * pdfEmitterDiscrete(emitter);
}

float e_KernelDynamicScene::pdfSensorDirect(const DirectSamplingRecord &dRec) const
{
	return m_Camera.pdfDirect(dRec);
}

Spectrum e_KernelDynamicScene::sampleEmitterPosition(PositionSamplingRecord &pRec, const Vec2f &_sample) const
{
	Vec2f sample = _sample;
	float emPdf;
	const e_KernelLight *emitter = sampleLight(emPdf, sample);

	Spectrum value = emitter->samplePosition(pRec, sample);

	pRec.object = emitter;
	pRec.pdf *= emPdf;

	return value / emPdf;
}

Spectrum e_KernelDynamicScene::sampleSensorPosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	pRec.object = &m_Camera;
	return m_Camera.samplePosition(pRec, sample, extra);
}

float e_KernelDynamicScene::pdfEmitterPosition(const PositionSamplingRecord &pRec) const
{
	const e_KernelLight *emitter = (const e_KernelLight*)pRec.object;
	return emitter->pdfPosition(pRec) * pdfEmitterDiscrete(emitter);
}

float e_KernelDynamicScene::pdfSensorPosition(const PositionSamplingRecord &pRec) const
{
	const e_Sensor *sensor = (const e_Sensor*)pRec.object;
	return sensor->pdfPosition(pRec);
}

Spectrum e_KernelDynamicScene::sampleEmitterRay(Ray& ray, const e_KernelLight*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	Vec2f sample = spatialSample;
	float emPdf;
	emitter = sampleLight(emPdf, sample);

	return emitter->sampleRay(ray, sample, directionalSample) / emPdf;
}

Spectrum e_KernelDynamicScene::sampleSensorRay(Ray& ray, const e_Sensor*& sensor, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	sensor = &m_Camera;
	return sensor->sampleRay(ray, spatialSample, directionalSample);
}