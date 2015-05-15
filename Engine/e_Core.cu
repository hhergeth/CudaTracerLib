#include "e_Core.h"

#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"

#define FREEIMAGE_LIB
#include <FreeImage.h>

#include "e_FileTexture.h"
#include <crtdbg.h>

void InitializeCuda4Tracer(const char* dataPath)
{
#ifndef NDEBUG
	_CrtSetDbgFlag(_CrtSetDbgFlag(0) | _CRTDBG_CHECK_ALWAYS_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
	cudaError er = CUDA_FREE(0);
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	e_RoughTransmittanceManager::StaticInitialize(std::string(dataPath));
}

void DeInitializeCuda4Tracer()
{
	FreeImage_DeInitialise();
	SpectrumHelper::StaticDeinitialize();
	e_RoughTransmittanceManager::StaticDeinitialize();
#ifndef NDEBUG
	//_CrtDumpMemoryLeaks();
#endif
}