#include "e_Core.h"

#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"

#define FREEIMAGE_LIB
#include <FreeImage.h>

#include "e_FileTexture.h"

void InitializeCuda4Tracer(const char* dataPath)
{
	cudaError er = CUDA_FREE(0);
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	e_RoughTransmittanceManager::StaticInitialize(std::string(dataPath));
}