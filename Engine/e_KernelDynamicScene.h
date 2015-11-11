#pragma once

#include <MathTypes.h>
#include "e_Buffer_device.h"
#include "e_SceneBVH_device.h"
#include "e_Volumes.h"
#include "e_Sensor.h"

namespace CudaTracerLib {

struct e_KernelLight;
class e_Node;
struct e_KernelMesh;
struct e_TriangleData;
struct e_TriIntersectorData;
struct e_TriIntersectorData2;
class e_DynamicScene;
struct e_Sensor;
struct e_BVHNodeData;
struct e_KernelMaterial;
struct e_KernelMIPMap;
struct TraceResult;

struct e_KernelDynamicScene
{
	e_KernelBuffer<e_TriangleData> m_sTriData;
	e_KernelBuffer<e_TriIntersectorData> m_sBVHIntData;
	e_KernelBuffer<e_BVHNodeData> m_sBVHNodeData;
	e_KernelBuffer<e_TriIntersectorData2> m_sBVHIndexData;
	e_KernelBuffer<e_KernelMaterial> m_sMatData;
	e_KernelBuffer<e_KernelMIPMap> m_sTexData;
	e_KernelBuffer<e_KernelMesh> m_sMeshData;
	e_KernelBuffer<e_Node> m_sNodeData;
	e_KernelBuffer<e_KernelLight> m_sLightData;
	e_KernelBuffer<char> m_sAnimData;
	e_KernelSceneBVH m_sSceneBVH;
	e_KernelAggregateVolume m_sVolume;
	unsigned int m_uEnvMapIndex;
	AABB m_sBox;
	e_Sensor m_Camera;

	CUDA_HOST CUDA_DEVICE bool Occluded(const Ray& r, float tmin, float tmax, TraceResult* res = 0) const;
	CUDA_DEVICE CUDA_HOST Spectrum EvalEnvironment(const Ray& r) const;
	CUDA_DEVICE CUDA_HOST Spectrum EvalEnvironment(const Ray& r, const Ray& rX, const Ray& rY) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDiscrete(const e_KernelLight *emitter) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDirect(const DirectSamplingRecord &dRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterPosition(PositionSamplingRecord &pRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorPosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = NULL) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterPosition(const PositionSamplingRecord &pRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorPosition(const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterRay(Ray& ray, const e_KernelLight*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorRay(Ray& ray, const e_Sensor*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum sampleEmitterRay(Ray& ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		const e_KernelLight* emitter;
		return sampleEmitterRay(ray, emitter, spatialSample, directionalSample);
	}
	CUDA_FUNC_IN Spectrum sampleSensorRay(Ray& ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		const e_Sensor* emitter;
		return sampleSensorRay(ray, emitter, spatialSample, directionalSample);
	}
	CUDA_FUNC_IN Spectrum sampleSensorRay(Ray& ray, Ray& rX, Ray& rY, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		return m_Camera.sampleRayDifferential(ray, rX, rY, spatialSample, directionalSample);
	}

	CUDA_FUNC_IN Ray GenerateSensorRay(int x, int y)
	{
		Ray r;
		sampleSensorRay(r, Vec2f(x, y), Vec2f(0));
		return r;
	}

	CUDA_DEVICE CUDA_HOST const e_KernelLight* sampleLight(float& emPdf, Vec2f& sample) const;
};

}