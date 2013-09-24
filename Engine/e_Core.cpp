#include <StdAfx.h>
#include "e_Core.h"

#include "..\Base\FrameworkInterop.h"
#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"

void InitializeCuda4Tracer()
{
	FW::CudaModule::staticInit();
	SpectrumHelper::StaticInitialize();
	e_RoughTransmittanceManager::StaticInitialize();
}