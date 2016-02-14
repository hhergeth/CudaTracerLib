#include <StdAfx.h>
#include "Core.h"
#include <CudaMemoryManager.h>
#include "RoughTransmittance.h"
#include <Kernel/Tracer.h>
#include <ctime>
#include <cuda_runtime.h>
#define FREEIMAGE_LIB
#include <FreeImage.h>

namespace CudaTracerLib {

void InitializeCuda4Tracer(const std::string& dataPath)
{
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
	CudaSetToZero_FreeBuffer();
}

}
