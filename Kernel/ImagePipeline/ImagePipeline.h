#pragma once

#include <vector>
#include <Engine/Image.h>
#include "Filter/Filter.h"
#include "PostProcess/PostProcess.h"
#include <Engine/Filter.h>
#include "Filter/CanonicalFilter.h"

namespace CudaTracerLib
{

struct ImagePipelineInfo
{
	unsigned int numPasses;
	float splatScale;
};

class TracerBase;
ImagePipelineInfo constructImagePipelineInfo(const TracerBase& tracer);

//generic pipeline
void applyImagePipeline(ImagePipelineInfo info, Image& img, ImageSamplesFilter* filter, const std::vector<PostProcess*>& postProcesses);

//pipeline which doesn't use postprocesses
inline void applyImagePipeline(ImagePipelineInfo info, Image& img, ImageSamplesFilter* filter)
{
	static std::vector<PostProcess*> emptyProcesses;
	applyImagePipeline(info, img, filter, emptyProcesses);
}

//pipeline which only uses one of the canonical filters
inline void applyImagePipeline(ImagePipelineInfo info, Image& img, const Filter& F)
{
	CanonicalFilter f(F);
	applyImagePipeline(info, img, &f);
}

}