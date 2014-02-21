#include <StdAfx.h>
#include "e_Core.h"


#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"

#include <FreeImage.h>

void InitializeCuda4Tracer()
{
	cudaError er = cudaFree(0);
	SpectrumHelper::StaticInitialize();
	e_RoughTransmittanceManager::StaticInitialize();
	FreeImage_Initialise();
}

void ThrowCudaErrors()
{
	cudaError r = cudaGetLastError();
	if(r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}

void ThrowCudaErrors(cudaError r)
{
	if(r)
	{
		const char* msg = cudaGetErrorString(r);
		std::cout << msg;
		throw std::runtime_error(msg);
	}
}