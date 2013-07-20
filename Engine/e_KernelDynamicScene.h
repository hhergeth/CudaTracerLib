#pragma once

#include "..\Math\vector.h"
#include "e_Buffer.h"
#include "e_Node.h"
#include "e_SceneBVH.h"
#include "e_Light.h"
#include "e_TerrainHeader.h"
#include "Engine/e_KernelMaterial.h"
#include "e_Volumes.h"
#include "e_KernelDynamicScene.h"
#include "e_EnvironmentMap.h"

struct e_KernelLight;

struct e_KernelDynamicScene
{
	e_KernelBuffer<e_TriangleData> m_sTriData;
	e_KernelBuffer<e_TriIntersectorData> m_sBVHIntData;
	e_KernelBuffer<e_BVHNodeData> m_sBVHNodeData;
	e_KernelBuffer<int> m_sBVHIndexData;
	e_KernelBuffer<e_KernelMaterial> m_sMatData;
	e_KernelBuffer<e_KernelTexture> m_sTexData;
	e_KernelBuffer<e_KernelMesh> m_sMeshData;
	e_KernelBuffer<e_Node> m_sNodeData;
	e_KernelBuffer<e_KernelLight> m_sLightData;
	e_KernelBuffer<char> m_sAnimData;
	e_KernelSceneBVH m_sSceneBVH;
	e_KernelTerrainData m_sTerrain;
	e_KernelAggregateVolume m_sVolume;
	e_EnvironmentMap m_sEnvMap;
	AABB m_sBox;
};