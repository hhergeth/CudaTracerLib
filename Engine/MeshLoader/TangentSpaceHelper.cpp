#include <StdAfx.h>
#include "TangentSpaceHelper.h"
#include <Base/Platform.h>

namespace CudaTracerLib {

void ComputeTangentSpace(const Vec3f* V, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, NormalizedT<Vec3f>* a_Normals, bool flipOrder)
{
	Vec3f* NOR = (Vec3f*)a_Normals;
	for (unsigned int i = 0; i < vertexCount; i++)
		NOR[i] = Vec3f(0.0f);
	unsigned int o1 = 2 * !!flipOrder, o3 = 2 - o1;
	for (unsigned int f = 0; f < triCount; f++)
	{
		unsigned int i1 = I ? I[f * 3 + o1] : f * 3 + 0;
		unsigned int i2 = I ? I[f * 3 + 1] : f * 3 + 1;
		unsigned int i3 = I ? I[f * 3 + o3] : f * 3 + 2;
		const Vec3f v1 = V[i1], v2 = V[i2], v3 = V[i3];

		const Vec3f n1 = v1 - v2, n2 = v3 - v2;
		const Vec3f normal = cross(n1, n2);

		NOR[i1] += normal;
		NOR[i2] += normal;
		NOR[i3] += normal;
	}

	for (unsigned int a = 0; a < vertexCount; a++)
	{
		auto n = NOR[a].normalized();
		a_Normals[a] = n;
	}
}

}