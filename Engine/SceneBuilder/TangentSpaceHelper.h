#pragma once

#include <Math/Vector.h>

namespace CudaTracerLib {

void ComputeTangentSpace(const Vec3f* V, const Vec2f* T, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, NormalizedT<Vec3f>* a_Normals, NormalizedT<Vec3f>* a_Tangents, NormalizedT<Vec3f>* a_BiTangents = 0, bool flipOrder = false);

}