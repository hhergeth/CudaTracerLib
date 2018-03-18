#include "Texture.h"
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

Spectrum ImageTexture::Evaluate(const Vec2f& _uv) const
{
    if (tex_idx == 0xffffffff)
        return Spectrum(0.0f);

    Vec2f uv = mapping.TransformPoint(_uv);
    return getTexture().Sample(uv) * m_scale;
}

Spectrum ImageTexture::Evaluate(const DifferentialGeometry& its) const
{
    if (tex_idx == 0xffffffff)
        return Spectrum(0.0f);

    if (its.hasUVPartials)
    {
        Vec2f uv = mapping.Map(its);
        float dsdx, dsdy,
            dtdx, dtdy;
        mapping.differentiate(its, dsdx, dsdy, dtdx, dtdy);
        return getTexture().eval(uv, Vec2f(dsdx, dtdx), Vec2f(dsdy, dtdy)) * m_scale;
    }
    else return Evaluate(its.uv[mapping.setId]);
}

Spectrum ImageTexture::Average() const
{
    if (tex_idx == 0xffffffff)
        return Spectrum(0.0f);

    return getTexture().Sample(Vec2f(0), 1) * m_scale;
}

const KernelMIPMap& ImageTexture::getTexture() const
{
    return g_SceneData.m_sTexData[tex_idx];
}

}