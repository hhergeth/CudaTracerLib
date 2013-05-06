#pragma once

#include "..\Math\vector.h"
#include "e_DataStream.h"
#include "e_HostDeviceBuffer.h"
#include "e_Node.h"
#include "e_SceneBVH.h"
#include "e_Light.h"
#include "e_TerrainHeader.h"
#include "Engine/e_KernelMaterial.h"
#include "e_Volumes.h"
#include "e_KernelDynamicScene.h"

struct e_KernelLight;

struct e_KernelDynamicScene
{
	e_KernelDataStream<e_TriangleData> m_sTriData;
	e_KernelDataStream<e_TriIntersectorData> m_sBVHIntData;
	e_KernelDataStream<e_BVHNodeData> m_sBVHNodeData;
	e_KernelDataStream<int> m_sBVHIndexData;
	e_KernelDataStream<e_KernelMaterial> m_sMatData;
	e_KernelHostDeviceBuffer<e_KernelTexture> m_sTexData;
	e_KernelHostDeviceBuffer<e_KernelMesh> m_sMeshData;
	e_KernelDataStream<e_Node> m_sNodeData;
	e_KernelDataStream<e_KernelLight> m_sLightData;
	e_KernelDataStream<char> m_sAnimData;
	e_KernelSceneBVH m_sSceneBVH;
	e_KernelTerrainData m_sTerrain;
	e_KernelAggregateVolume m_sVolume;
	AABB m_sBox;
};