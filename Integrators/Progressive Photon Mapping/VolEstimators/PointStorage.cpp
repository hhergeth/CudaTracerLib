#include <StdAfx.h>
#include "PointStorage.h"

namespace CudaTracerLib {

	void PointStorage::StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
	{
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(3);
		m_sStorage.ResetBuffer();
	}

}