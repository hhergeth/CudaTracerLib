#include <StdAfx.h>
#include "TangentSpaceHelper.h"
#include "../../Base/Platform.h"
#include "MD5Parser.h"
#include "../e_AnimatedMesh.h"

void ComputeTangentSpace(const Vec3f* V, const Vec2f* T, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, Vec3f* a_Normals, Vec3f* a_Tangents, Vec3f* a_BiTangents)
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
	for (unsigned int f = 0; f < triCount; f++)
	{
		unsigned int i1 = I ? I[f * 3 + 0] : f * 3 + 0;
		unsigned int i2 = I ? I[f * 3 + 1] : f * 3 + 1;
		unsigned int i3 = I ? I[f * 3 + 2] : f * 3 + 2;
		const Vec3f v1 = V[i1], v2 = V[i2], v3 = V[i3];

		const Vec3f e0 = v3 - v1, e1 = v2 - v1;//, n = normalize(v1q.n + v2q.n + v3q.n);
		const Vec2f w1 = T ? T[i1] : Vec2f(0), w2 = T ? T[i2] : Vec2f(0), w3 = T ? T[i3] : Vec2f(0);
		const Vec3f n2 = normalize(cross(e1, e0));
		//if(fsumf(n2 - n) > 1e-3)
		//	throw 1;
		a_Normals[i1] += n2;
		a_Normals[i2] += n2;
		a_Normals[i3] += n2;

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
	/*
	if(triCount < (1 << 16))
	for (unsigned int a = 0; a < vertexCount; a++)
	for (unsigned int i = a + 1; i < vertexCount; i++)
	if(DistanceSquared(V[a], V[i]) < 0.01f)
	{
	tan1[a] += tan1[i];
	tan1[i] = tan1[a];
	tan2[a] += tan2[i];
	tan2[i] = tan2[a];
	}
	*/

	for (unsigned int i = 0; i < triCount; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			unsigned int a = I ? I[i * 3 + j] : i * 3 + j;
			const Vec3f n = a_Normals[a] = normalize(normalize(a_Normals[a]));
			const Vec3f t = tan1[a];

			// Gram-Schmidt orthogonalize
			a_Tangents[a] = normalize(t - n * dot(n, t));

			// Calculate handedness
			float h = (dot(cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
			if (a_BiTangents)
				a_BiTangents[a] = normalize(cross(a_Tangents[a], n) * h);
		}
	}
	delete[] tan1;
}

void ComputeTangentSpace(MD5Model* a_Mesh, e_AnimatedVertex** a_Vertices, Vec3f** a_Pos, unsigned int* a_NumV)
{
	unsigned int vertexCount = 0;
	for (int i = 0; i < a_Mesh->meshes.size(); i++)
		vertexCount += (unsigned int)a_Mesh->meshes[i]->verts.size();
	*a_NumV = vertexCount;
	*a_Pos = new Vec3f[vertexCount];
	*a_Vertices = new e_AnimatedVertex[vertexCount];
	Platform::SetMemory(a_Pos[0], sizeof(float3) * vertexCount);
	Platform::SetMemory(a_Vertices[0], sizeof(e_AnimatedVertex) * vertexCount);

	unsigned int off = 0;
	for (int i = 0; i < a_Mesh->meshes.size(); i++)
	{
		for (int v = 0; v < a_Mesh->meshes[i]->verts.size(); v++)
		{
			e_AnimatedVertex& av = a_Vertices[0][off + v];
			Vec3f pos = Vec3f(0);
			Vertex& V = a_Mesh->meshes[i]->verts[v];
			int a = min(g_uMaxWeights, V.weightCount);
			for (int k = 0; k < a; k++)
			{
				Weight &w = a_Mesh->meshes[i]->weights[V.weightIndex + k];
				Joint &joint = a_Mesh->joints[w.joint];
				Vec3f r = joint.quat.toMatrix().TransformPoint(w.pos);
				pos += (joint.pos + r) * w.w;
				av.m_cBoneIndices[k] = (unsigned char)w.joint;
				av.m_fBoneWeights[k] = (unsigned char)(w.w * 255.0f);
			}
			av.m_fVertexPos = a_Pos[0][off + v] = pos;
		}
		off += (unsigned int)a_Mesh->meshes[i]->verts.size();
	}

	off = 0;
	for (int submesh = 0; submesh < a_Mesh->meshes.size(); submesh++)
	{
		std::vector<Tri> indices = a_Mesh->meshes[submesh]->tris;
		std::vector<Vertex> vertices = a_Mesh->meshes[submesh]->verts;
		for (int i = 0; i < indices.size(); i++)
		{
			long i1 = indices[i].v[0];
			long i2 = indices[i].v[1];
			long i3 = indices[i].v[2];
			Vertex& v1 = vertices[i1], &v2 = vertices[i2], &v3 = vertices[i3];

			Vec3f p0 = a_Pos[0][i1 + off], p1 = a_Pos[0][i2 + off], p2 = a_Pos[0][i3 + off], e0 = p1 - p0, e1 = p2 - p0, n = normalize(cross(e0, e1));
			a_Vertices[0][i1].m_fNormal += n;
			a_Vertices[0][i2].m_fNormal += n;
			a_Vertices[0][i3].m_fNormal += n;

			float x1 = p1.x - p0.x;
			float x2 = p2.x - p0.x;
			float y1 = p1.y - p0.y;
			float y2 = p2.y - p0.y;
			float z1 = p1.z - p0.z;
			float z2 = p2.z - p0.z;

			float s1 = v2.tc[0] - v1.tc[0];
			float s2 = v3.tc[0] - v1.tc[0];
			float t1 = v2.tc[1] - v1.tc[1];
			float t2 = v3.tc[1] - v1.tc[1];

			float r = 1.0F / (s1 * t2 - s2 * t1);
			Vec3f sdir = Vec3f((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
			Vec3f tdir = Vec3f((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

			a_Vertices[0][i1 + off].m_fTangent += sdir;
			a_Vertices[0][i2 + off].m_fTangent += sdir;
			a_Vertices[0][i3 + off].m_fTangent += sdir;

			a_Vertices[0][i1 + off].m_fBitangent += tdir;
			a_Vertices[0][i2 + off].m_fBitangent += tdir;
			a_Vertices[0][i3 + off].m_fBitangent += tdir;
		}
		off += (unsigned int)vertices.size();
	}

	off = 0;
	for (int submesh = 0; submesh < a_Mesh->meshes.size(); submesh++)
	{
		std::vector<Tri> indices = a_Mesh->meshes[submesh]->tris;
		std::vector<Vertex> vertices = a_Mesh->meshes[submesh]->verts;
		for (int i = 0; i < indices.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				unsigned int a = indices[i].v[j] + off;
				const Vec3f n = a_Vertices[0][a].m_fNormal = normalize(a_Vertices[0][a].m_fNormal);
				const Vec3f t = a_Vertices[0][a].m_fTangent;

				// Gram-Schmidt orthogonalize
				a_Vertices[0][a].m_fTangent = normalize(t - n * dot(n, t));

				// Calculate handedness
				float h = (dot(cross(n, t), a_Vertices[0][a].m_fTangent) < 0.0F) ? -1.0F : 1.0F;
				a_Vertices[0][a].m_fBitangent = normalize(cross(a_Vertices[0][a].m_fTangent, n) * h);
			}
		}
		off += (unsigned int)vertices.size();
	}
}