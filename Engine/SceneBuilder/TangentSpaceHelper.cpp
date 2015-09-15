#include <StdAfx.h>
#include "TangentSpaceHelper.h"
#include "../../Base/Platform.h"

//http://www.terathon.com/code/tangent.html
void ComputeTangentSpace(const Vec3f* V, const Vec2f* T, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, Vec3f* a_Normals, Vec3f* a_Tangents, Vec3f* a_BiTangents, bool flipOrder)
{
	bool hasUV = false;
	if (T)
		for (unsigned int i = 0; i < min(12u, vertexCount); i++)
			if (length(T[i]) != 0)
			{
				hasUV = true;
				break;
			}

	Vec3f *tan1 = new Vec3f[vertexCount * 2];
	Vec3f *tan2 = tan1 + vertexCount;
	Platform::SetMemory(tan1, vertexCount * sizeof(float3) * 2);
	Platform::SetMemory(a_Normals, vertexCount * sizeof(float3));
	unsigned int o1 = 2 * !!flipOrder, o3 = 2 - o1;
	for (unsigned int f = 0; f < triCount; f++)
	{
		unsigned int i1 = I ? I[f * 3 + o1] : f * 3 + 0;
		unsigned int i2 = I ? I[f * 3 + 1] : f * 3 + 1;
		unsigned int i3 = I ? I[f * 3 + o3] : f * 3 + 2;
		const Vec3f v1 = V[i1], v2 = V[i2], v3 = V[i3];

		const Vec3f n1 = v1 - v2, n2 = v3 - v2;//, n = normalize(v1q.n + v2q.n + v3q.n);
		const Vec2f w1 = T ? T[i1] : Vec2f(0), w2 = T ? T[i2] : Vec2f(0), w3 = T ? T[i3] : Vec2f(0);
		const Vec3f normal = cross(n1, n2);
		//if(fsumf(n2 - n) > 1e-3)
		//	throw std::runtime_error(__FUNCTION__);
		a_Normals[i1] += normal;
		a_Normals[i2] += normal;
		a_Normals[i3] += normal;

		float x1 = v2.x - v1.x;
		float x2 = v3.x - v1.x;
		float y1 = v2.y - v1.y;
		float y2 = v3.y - v1.y;
		float z1 = v2.z - v1.z;
		float z2 = v3.z - v1.z;

		float s1 = w2.x - w1.x;
		float s2 = w3.x - w1.x;
		float t1 = w2.y - w1.y;
		float t2 = w3.y - w1.y;
		if (!hasUV)
		{
			s1 = x1 - x2 + 0.1f;
			s2 = x1 + x2 + 0.1f;
			t1 = y1 - y2 + 0.1f;
			t2 = z1 + z2 + 0.1f;
		}

		float r = 1.0F / (s1 * t2 - s2 * t1);
		Vec3f sdir = Vec3f((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
		Vec3f tdir = Vec3f((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;
	}

	for (unsigned int a = 0; a < vertexCount; a++)
	{
		const Vec3f n = a_Normals[a] = normalize(a_Normals[a]);
		const Vec3f t = tan1[a];
		Vec3f tangent = normalize(t - n * dot(n, t));
		a_Tangents[a] = tangent;

		float h = (dot(cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
		if (a_BiTangents)
			a_BiTangents[a] = normalize(cross(a_Tangents[a], n) * h);
	}

	delete[] tan1;
}