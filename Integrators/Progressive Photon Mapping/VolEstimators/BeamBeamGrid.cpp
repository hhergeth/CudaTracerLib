#include <StdAfx.h>
#include "BeamBeamGrid.h"
#include <set>
#include <Base/Timer.h>
#include <Math/Compression.h>
#include <Math/half.h>
#include <bitset>
#include <set>

namespace CudaTracerLib {

void BeamBeamGrid::StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
{
	m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
	m_uBeamIdx = 0;
	m_sStorage.ResetBuffer();
}

void BeamBeamGrid::PrepareForRendering()
{
	
}

}
