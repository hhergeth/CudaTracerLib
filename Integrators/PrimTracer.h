#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PrimTracer : public Tracer<false, false>, public IDepthTracer
{
public:
	enum DrawMode
	{
		linear_depth,
		D3D_depth,

		v_absdot_n_geo,
		v_dot_n_geo,
		v_dot_n_shade,
		n_geo_colored,
		n_shade_colored,
		uv,
		bary_coords,

		first_Le,
		first_f,
		first_f_direct,

		first_non_delta_Le,
		first_non_delta_f,
		first_non_delta_f_direct,
	};

	PARAMETER_KEY(DrawMode, DrawingMode)
	PARAMETER_KEY(int, MaxPathLength)

	PrimTracer();
	virtual void Debug(Image* I, const Vec2i& pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
protected:
	virtual void DoRender(Image* I);
};

}