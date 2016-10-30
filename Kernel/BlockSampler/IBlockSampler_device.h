#pragma once
#include <Engine/Image.h>

namespace CudaTracerLib {

#ifdef CUDA_RELEASE_BUILD
#define BLOCK_FACTOR 4
#else
#define BLOCK_FACTOR 2
#endif

//Please distinguish between the concept of cuda blocks and the actual blocks the sampler manages, the latter is formed of multiple cuda blocks 

//the number of threads in a cuda block in each dimension
#define BLOCK_SAMPLER_ThreadsPerBlock dim3(16, 8)
//the number of cuda blocks in each block
#define BLOCK_SAMPLER_NumBlocks dim3(2 * BLOCK_FACTOR, 4 * BLOCK_FACTOR)
//the number of pixels in each block
#define BLOCK_SAMPLER_BlockSize (32 * BLOCK_FACTOR)

//a launch configuration for a cuda kernel which implements the block sampler "interface"
#define BLOCK_SAMPLER_LAUNCH_CONFIG BLOCK_SAMPLER_NumBlocks,BLOCK_SAMPLER_ThreadsPerBlock

}