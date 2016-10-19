#include <StdAfx.h>
#include "PointStorage.h"

namespace CudaTracerLib {

	void PointStorage::StartNewPass(DynamicScene* scene)
	{
		m_sStorage.ResetBuffer();
	}

}