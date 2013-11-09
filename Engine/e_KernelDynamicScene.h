#pragma once

#include <MathTypes.h>
#include "e_Buffer.h"
#include "e_SceneBVH.h"
#include "e_TerrainHeader.h"
#include "e_Volumes.h"
#include "..\Math\Sampling.h"

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

#define MAX_LIGHT_COUNT 32
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
	e_KernelTerrainData m_sTerrain;
	e_KernelAggregateVolume m_sVolume;
	unsigned int m_uEnvMapIndex;
	Distribution1D<MAX_LIGHT_COUNT> m_emitterPDF;
	unsigned int m_uEmitterIndices[MAX_LIGHT_COUNT];
	unsigned int m_uEmitterCount;
	AABB m_sBox;

	CUDA_HOST CUDA_DEVICE bool Occluded(const Ray& r, float tmin, float tmax) const;
	CUDA_DEVICE CUDA_HOST Spectrum EvalEnvironment(const Ray& r) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDiscrete(const e_KernelLight *emitter) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterDirect(DirectSamplingRecord &dRec, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorDirect(DirectSamplingRecord &dRec, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterDirect(const DirectSamplingRecord &dRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterPosition(PositionSamplingRecord &pRec, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorPosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra = NULL) const;
	CUDA_DEVICE CUDA_HOST float pdfEmitterPosition(const PositionSamplingRecord &pRec) const;
	CUDA_DEVICE CUDA_HOST float pdfSensorPosition(const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleEmitterRay(Ray& ray, const e_KernelLight*& emitter, const float2 &spatialSample, const float2 &directionalSample) const;
	CUDA_DEVICE CUDA_HOST Spectrum sampleSensorRay(Ray& ray, const e_Sensor*& emitter, const float2 &spatialSample, const float2 &directionalSample) const;
private:
	CUDA_DEVICE CUDA_HOST const e_KernelLight* sampleLight(float& emPdf, float2& sample) const;
};