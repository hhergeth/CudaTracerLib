#pragma once

#include <vector>
#include <Engine/Image.h>
#include "Filter/Filter.h"
#include "PostProcess/PostProcess.h"
#include <SceneTypes/Filter.h>
#include "Filter/CanonicalFilter.h"
#include <Base/SynchronizedBuffer.h>

namespace CudaTracerLib
{

class TracerBase;

//generic pipeline
void applyImagePipeline(const TracerBase& tracer, Image& img, ImageSamplesFilter* filter, PostProcess* postProcess);

//pipeline which doesn't use a PostProcess
inline void applyImagePipeline(TracerBase& tracer, Image& img, ImageSamplesFilter* filter)
{
	applyImagePipeline(tracer, img, filter, 0);
}

//pipeline which only uses one of the canonical filters
inline void applyImagePipeline(TracerBase& tracer, Image& img, const Filter& F)
{
	CanonicalFilter f(F);
	applyImagePipeline(tracer, img, &f);
}

}