#pragma once
#include "../../Defines.h"
#include "../../MathTypes.h"
#include <Engine/e_SpatialGrid.h>

struct k_PointStorage
{
	struct volPhoton
	{

	};
	e_SpatialLinkedMap<volPhoton> m_sStorage;

	k_PointStorage(unsigned int gridDim, unsigned int numPhotons)
		: m_sStorage(gridDim, numPhotons)
	{

	}

	void Free()
	{
		m_sStorage.Free();
	}

	void StartNewPass()
	{
		m_sStorage.ResetBuffer();
	}

	void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sStorage.SetSceneDimensions(box, a_InitRadius);
	}


};