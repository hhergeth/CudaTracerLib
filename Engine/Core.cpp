#include <StdAfx.h>
#include "Core.h"
#include <Base/CudaMemoryManager.h>
#include "RoughTransmittance.h"
#include <Kernel/Tracer.h>
#include <ctime>
#include <cuda_runtime.h>
#define FREEIMAGE_LIB
#include <FreeImage/FreeImage.h>
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

void InitializeCuda4Tracer(const std::string& dataPath)
{
	ThrowCudaErrors(cudaFree(0));
	InitializeKernel();
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	RoughTransmittanceManager::StaticInitialize(dataPath);
}

void DeInitializeCuda4Tracer()
{
	DeinitializeKernel();
	FreeImage_DeInitialise();
	SpectrumHelper::StaticDeinitialize();
	RoughTransmittanceManager::StaticDeinitialize();
	CudaSetToZero_FreeBuffer();
}

}
