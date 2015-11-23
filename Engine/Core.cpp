#include <StdAfx.h>
#include "Core.h"
#include <CudaMemoryManager.h>
#include <MathTypes.h>
#include "RoughTransmittance.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>
#include "MIPMap.h"
#include <crtdbg.h>
#include <Kernel/Tracer.h>
#include <ctime>

namespace CudaTracerLib {

void InitializeCuda4Tracer(const std::string& dataPath)
{
#ifndef NDEBUG
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetDbgFlag(_CrtSetDbgFlag(0) | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
	ThrowCudaErrors(cudaFree(0));
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	RoughTransmittanceManager::StaticInitialize(dataPath);
}

void DeInitializeCuda4Tracer()
{
	TracerBase::g_sRngs.Free();
	FreeImage_DeInitialise();
	SpectrumHelper::StaticDeinitialize();
	RoughTransmittanceManager::StaticDeinitialize();
#ifndef NDEBUG
	//_CrtDumpMemoryLeaks();
#endif
}

}