#pragma once

#include <Math/Vector.h>

namespace CudaTracerLib {

CTL_EXPORT void ComputeTangentSpace(const Vec3f* V, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, NormalizedT<Vec3f>* a_Normals, bool flipOrder = false);

}