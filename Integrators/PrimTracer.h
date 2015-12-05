#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

#define PTDM(X) X(linear_depth) X(D3D_depth) X(v_absdot_n_geo) X(v_dot_n_geo) X(v_dot_n_shade) X(n_geo_colored) X(n_shade_colored) X(uv) X(bary_coords) X(first_Le) X(first_f) X(first_f_direct) X(first_non_delta_Le) X(first_non_delta_f) X(first_non_delta_f_direct)
ENUMIZE(PathTrace_DrawMode, PTDM)
#undef PTDM

class PrimTracer : public Tracer<false, false>, public IDepthTracer
{
public:

	PARAMETER_KEY(PathTrace_DrawMode, DrawingMode)
	PARAMETER_KEY(int, MaxPathLength)

	PrimTracer();
	virtual void Debug(Image* I, const Vec2i& pixel);
protected:
	virtual void DoRender(Image* I);
};

}