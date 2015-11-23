#pragma once

#include <MathTypes.h>
#include "Buffer_device.h"
#include "SceneBVH_device.h"
#include "Volumes.h"
#include "Sensor.h"

namespace CudaTracerLib {

struct KernelLight;
class Node;
struct KernelMesh;
struct TriangleData;
struct TriIntersectorData;
struct TriIntersectorData2;
class DynamicScene;
struct Sensor;
struct BVHNodeData;
struct Material;
struct KernelMIPMap;
struct TraceResult;

struct KernelDynamicScene
{
	KernelBuffer<TriangleData> m_sTriData;
	KernelBuffer<TriIntersectorData> m_sBVHIntData;
	KernelBuffer<BVHNodeData> m_sBVHNodeData;
	KernelBuffer<TriIntersectorData2> m_sBVHIndexData;
	KernelBuffer<Material> m_sMatData;
	KernelBuffer<KernelMIPMap> m_sTexData;
	KernelBuffer<KernelMesh> m_sMeshData;
	KernelBuffer<Node> m_sNodeData;
	KernelBuffer<KernelLight> m_sLightData;
	KernelBuffer<char> m_sAnimData;
	KernelSceneBVH m_sSceneBVH;
	KernelAggregateVolume m_sVolume;
	unsigned int m_uEnvMapIndex;
	AABB m_sBox;
	Sensor m_Camera;

	CUDA_DEVICE CUDA_HOST bool Occluded(const Ray& r, float tmin, float tmax, TraceResult* res = 0) const;
	CUDA_DEVICE CUDA_HOST Spectrum EvalEnvironment(const Ray& r) const;
	CUDA_DEVICE CUDA_HOST Spectrum EvalEnvironment(const Ray& r, const Ray& rX, const Ray& rY) const;
	CUDA_DEVICE CUDA_HOST const KernelLight* sampleEmitter(float& emPdf, Vec2f& sample) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDiscrete(const KernelLight *emitter) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleAttenuatedEmitterDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleAttenuatedSensorDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDirect(const DirectSamplingRecord &dRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterPosition(PositionSamplingRecord &pRec, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorPosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = NULL) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterPosition(const PositionSamplingRecord &pRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorPosition(const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterRay(Ray& ray, const KernelLight*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorRay(Ray& ray, const Sensor*& emitter, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalTransmittance(const Vec3f& p1, const Vec3f& p2) const;

	CUDA_FUNC_IN Spectrum sampleEmitterRay(Ray& ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		const KernelLight* emitter;
		return sampleEmitterRay(ray, emitter, spatialSample, directionalSample);
	}
	CUDA_FUNC_IN Spectrum sampleSensorRay(Ray& ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		const Sensor* emitter;
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

};

}