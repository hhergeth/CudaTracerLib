#pragma once

#include "Utils.h"

#include <fstream>
#include <streambuf>

#include <Math/Vector.h>
#include <Math/float4x4.h>
#include <Math/Spectrum.h>

namespace CudaTracerLib {

Vec3f parseVector(const XMLNode& node, ParserState& S);

inline float4x4 parseMatrix_Id(ParserState& S)
{
	return S.id_matrix;
}
float4x4 parseMatrix(const XMLNode& node, ParserState& S, bool apply_id_rot = true);

Spectrum parseRGB(const XMLNode& node, ParserState& S, bool srgb);

Spectrum parseSpectrum(const XMLNode& node, ParserState& S);

}