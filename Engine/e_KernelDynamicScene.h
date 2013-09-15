#pragma once

#include <MathTypes.h>
#include "e_Buffer.h"
#include "e_SceneBVH.h"
#include "e_TerrainHeader.h"
#include "e_Volumes.h"
#include "e_EnvironmentMap.h"

struct e_KernelLight;
class e_Node;
struct e_KernelMesh;
struct e_TriangleData;
struct e_TriIntersectorData;
class e_DynamicScene;
class e_Camera;
struct e_BVHNodeData;
struct e_KernelMaterial;
struct e_KernelMIPMap;

struct e_ImportantLightSelector
{
	unsigned int m_sIndices[32];
	unsigned int m_uCount;
	CUDA_FUNC_IN e_ImportantLightSelector(){}
	e_ImportantLightSelector(e_DynamicScene* S, e_Camera* C);
private:
	bool Use(AABB& box, float3& p, float4x4& proj);
	bool Use(float3& p, float4x4& vp);
};

struct e_KernelDynamicScene
{
	e_KernelBuffer<e_TriangleData> m_sTriData;
	e_KernelBuffer<e_TriIntersectorData> m_sBVHIntData;
	e_KernelBuffer<e_BVHNodeData> m_sBVHNodeData;
	e_KernelBuffer<int> m_sBVHIndexData;
	e_KernelBuffer<e_KernelMaterial> m_sMatData;
	e_KernelBuffer<e_KernelMIPMap> m_sTexData;
	e_KernelBuffer<e_KernelMesh> m_sMeshData;
	e_KernelBuffer<e_Node> m_sNodeData;
	e_KernelBuffer<e_KernelLight> m_sLightData;
	e_KernelBuffer<char> m_sAnimData;
	e_KernelSceneBVH m_sSceneBVH;
	e_KernelTerrainData m_sTerrain;
	e_KernelAggregateVolume m_sVolume;
	e_EnvironmentMap m_sEnvMap;
	e_ImportantLightSelector m_sLightSelector;
	AABB m_sBox;
};