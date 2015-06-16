#pragma once

#include "../../Math/Vector.h"

void ComputeTangentSpace(const Vec3f* V, const Vec2f* T, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, Vec3f* a_Normals, Vec3f* a_Tangents, Vec3f* a_BiTangents = 0);