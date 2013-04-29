#pragma once
/*
inline float3 getVec(const obj_vector* q)
{
	return make_float3((float)q->e[0], (float)q->e[1], (float)q->e[2]);
}

inline void ComputeTangentSpace(objLoader* L, float3* a_Normals, float3* a_Tangents, float3* a_BiTangents)
{
	unsigned int vertexCount = L->vertexCount;
	float3 *tan1 = new float3[vertexCount * 2];
    float3 *tan2 = tan1 + vertexCount;
    ZeroMemory(tan1, vertexCount * sizeof(float3) * 2);
	for (long a = 0; a < L->faceCount; a++)
    {
		long i1 = L->faceList[a]->vertex_index[0], tv1 = L->faceList[a]->texture_index[0];
        long i2 = L->faceList[a]->vertex_index[1], tv2 = L->faceList[a]->texture_index[1];
        long i3 = L->faceList[a]->vertex_index[2], tv3 = L->faceList[a]->texture_index[2];
        
		const obj_vector* v1 = L->vertexList[i1];
        const obj_vector* v2 = L->vertexList[i2];
        const obj_vector* v3 = L->vertexList[i3];
        
		const obj_vector* w1 = L->textureList[tv1];
        const obj_vector* w2 = L->textureList[tv2];
        const obj_vector* w3 = L->textureList[tv3];

		const float3 p0 = getVec(v1), p1 = getVec(v2), p2 = getVec(v3), e0 = p1 - p0, e1 = p2 - p0, n = normalize(cross(e0, e1));
		a_Normals[i1] += n;
		a_Normals[i2] += n;
		a_Normals[i3] += n;
        
		float x1 = (float)v2->e[0] - (float)v1->e[0];
        float x2 = (float)v3->e[0] - (float)v1->e[0];
        float y1 = (float)v2->e[1] - (float)v1->e[1];
        float y2 = (float)v3->e[1] - (float)v1->e[1];
        float z1 = (float)v2->e[2] - (float)v1->e[2];
        float z2 = (float)v3->e[2] - (float)v1->e[2];
        
        float s1 = (float)w2->e[0] - (float)w1->e[0];
        float s2 = (float)w3->e[0] - (float)w1->e[0];
        float t1 = (float)w2->e[1] - (float)w1->e[1];
        float t2 = (float)w3->e[1] - (float)w1->e[1];
        
        float r = 1.0F / (s1 * t2 - s2 * t1);
		float3 sdir = make_float3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
        float3 tdir = make_float3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);
        
        tan1[i1] += sdir;
        tan1[i2] += sdir;
        tan1[i3] += sdir;
        
        tan2[i1] += tdir;
        tan2[i2] += tdir;
        tan2[i3] += tdir;
    }
	for (long a = 0; a < vertexCount; a++)
    {
        const float3 n = a_Normals[a] = normalize(a_Normals[a]);
        const float3 t = tan1[a];
        
        // Gram-Schmidt orthogonalize
		a_Tangents[a] = normalize(t - n * dot(n, t));
        
        // Calculate handedness
        float h = (dot(cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
		a_BiTangents[a] = normalize(cross(a_Tangents[a], n) * h);
    }
    delete[] tan1;
}
*/

#ifdef TS_DEC_FRAMEWORK
inline float3 getVec(const FW::VertexPNT& v)
{
	return make_float3(v.p.x, v.p.y, v.p.z);
}
inline void ComputeTangentSpace(FW::Mesh<FW::VertexPNT>* a_Mesh, float3* a_Normals, float3* a_Tangents, float3* a_BiTangents)
{
	unsigned int vertexCount = a_Mesh->numVertices();
	float3 *tan1 = new float3[vertexCount * 2];
    float3 *tan2 = tan1 + vertexCount;
    ZeroMemory(tan1, vertexCount * sizeof(float3) * 2);
	for (int submesh = 0; submesh < a_Mesh->numSubmeshes(); submesh++)
	{
		const FW::Array<FW::Vec3i>& indices = a_Mesh->indices(submesh);
		for (int i = 0; i < indices.getSize(); i++)
		{
			long i1 = indices[i].x;
			long i2 = indices[i].y;
			long i3 = indices[i].z;
			const FW::VertexPNT& v1q = a_Mesh[0][i1], &v2q = a_Mesh[0][i2], &v3q = a_Mesh[0][i3];

			const float3 v1 = getVec(v1q), v2 = getVec(v2q), v3 = getVec(v3q), e0 = v3 - v1, e1 = v2 - v1, n = normalize(v1q.n + v2q.n + v3q.n);
			const float2 w1 = v1q.t, w2 = v2q.t, w3 = v3q.t;
			const float3 n2 = normalize(cross(e1, e0));
			//if(fsumf(n2 - n) > 1e-3)
			//	throw 1;
			a_Normals[i1] += n;
			a_Normals[i2] += n;
			a_Normals[i3] += n;
        
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
        
			float r = 1.0F / (s1 * t2 - s2 * t1);
			float3 sdir = make_float3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,	(t2 * z1 - t1 * z2) * r);
			float3 tdir = make_float3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,	(s1 * z2 - s2 * z1) * r);
        
			tan1[i1] += sdir;
			tan1[i2] += sdir;
			tan1[i3] += sdir;
        
			tan2[i1] += tdir;
			tan2[i2] += tdir;
			tan2[i3] += tdir;
		}
	}
	for (int submesh = 0; submesh < a_Mesh->numSubmeshes(); submesh++)
	{
		const FW::Array<FW::Vec3i>& indices = a_Mesh->indices(submesh);
		for (int i = 0; i < indices.getSize(); i++)
		{
			for(int j = 0; j < 3; j++)
			{
				unsigned int a = indices[i][j];
				const float3 n = a_Normals[a] = normalize(normalize(a_Normals[a]));
				const float3 t = tan1[a];
        
				// Gram-Schmidt orthogonalize
				a_Tangents[a] = normalize(t - n * dot(n, t));
        
				// Calculate handedness
				float h = (dot(cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
				a_BiTangents[a] = normalize(cross(a_Tangents[a], n) * h);
			}
		}
	}
    delete[] tan1;
}
#endif

#ifdef TS_DEC_MD5
inline void ComputeTangentSpace(MD5Model* a_Mesh, e_AnimatedVertex** a_Vertices, float3** a_Pos, unsigned int* a_NumV)
{
	unsigned int vertexCount = 0;
	for(int i = 0; i < a_Mesh->meshes.size(); i++)
		vertexCount += a_Mesh->meshes[i]->verts.size();
	*a_NumV = vertexCount;
	*a_Pos = new float3[vertexCount];
	*a_Vertices = new e_AnimatedVertex[vertexCount];
	ZeroMemory(a_Pos[0], sizeof(float3) * vertexCount);
	ZeroMemory(a_Vertices[0], sizeof(e_AnimatedVertex) * vertexCount);

	unsigned int off = 0;
	for(int i = 0; i < a_Mesh->meshes.size(); i++)
	{
		for(int v = 0; v < a_Mesh->meshes[i]->verts.size(); v++)
		{
			e_AnimatedVertex& av = a_Vertices[0][off + v];
			float3 pos = make_float3(0);
			Vertex& V = a_Mesh->meshes[i]->verts[v];
			int a = MIN(g_uMaxWeights, V.weightCount);
			for ( int k=0; k < a; k++ )
			{
				Weight &w = a_Mesh->meshes[i]->weights[V.weightIndex + k];
				Joint &joint = a_Mesh->joints[w.joint];
				float3 r = joint.quat.toMatrix() * *(float3*)&w.pos;
				pos += (*(float3*)&joint.pos + r) * w.w;
				av.m_cBoneIndices[k] = (unsigned char)w.joint;
				av.m_fBoneWeights[k] = (unsigned char)(w.w * 255.0f);
			}
			av.m_fVertexPos = a_Pos[0][off + v] = pos;
		}
		off += a_Mesh->meshes[i]->verts.size();
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

			float3 p0 = a_Pos[0][i1 + off], p1 = a_Pos[0][i2 + off], p2 = a_Pos[0][i3 + off], e0 = p1 - p0, e1 = p2 - p0, n = normalize(cross(e0, e1));
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
			float3 sdir = make_float3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
			float3 tdir = make_float3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);
        
			a_Vertices[0][i1 + off].m_fTangent += sdir;
			a_Vertices[0][i2 + off].m_fTangent += sdir;
			a_Vertices[0][i3 + off].m_fTangent += sdir;
        
			a_Vertices[0][i1 + off].m_fBitangent += tdir;
			a_Vertices[0][i2 + off].m_fBitangent += tdir;
			a_Vertices[0][i3 + off].m_fBitangent += tdir;
		}
		off += vertices.size();
	}

	off = 0;
	for (int submesh = 0; submesh < a_Mesh->meshes.size(); submesh++)
	{
		std::vector<Tri> indices = a_Mesh->meshes[submesh]->tris;
		std::vector<Vertex> vertices = a_Mesh->meshes[submesh]->verts;
		for (int i = 0; i < indices.size(); i++)
		{
			for(int j = 0; j < 3; j++)
			{
				unsigned int a = indices[i].v[j] + off;
				const float3 n = a_Vertices[0][a].m_fNormal = normalize(a_Vertices[0][a].m_fNormal);
				const float3 t = a_Vertices[0][a].m_fTangent;
        
				// Gram-Schmidt orthogonalize
				a_Vertices[0][a].m_fTangent = normalize(t - n * dot(n, t));
        
				// Calculate handedness
				float h = (dot(cross(n, t), a_Vertices[0][a].m_fTangent) < 0.0F) ? -1.0F : 1.0F;
				a_Vertices[0][a].m_fBitangent = normalize(cross(a_Vertices[0][a].m_fTangent, n) * h);
			}
		}
		off += vertices.size();
	}

	off = 0;
	for(int i = 0; i < a_Mesh->meshes.size(); i++)
	{
		for(int v = 0; v < a_Mesh->meshes[i]->verts.size(); v++)
		{
			e_AnimatedVertex& av = a_Vertices[0][off + v];
			Vertex& V = a_Mesh->meshes[i]->verts[v];
			float3 n = av.m_fNormal, t = av.m_fTangent, b = av.m_fBitangent;
			av.m_fNormal = av.m_fTangent = av.m_fBitangent = make_float3(0);
			int a = MIN(g_uMaxWeights, V.weightCount);
			for ( int k=0; k < a; k++ )
			{
				Weight &w = a_Mesh->meshes[i]->weights[V.weightIndex + k];
				Joint &joint = a_Mesh->joints[w.joint];
				av.m_fNormal += joint.quat.toMatrix() * n * w.w;
			}
		}
		off += a_Mesh->meshes[i]->verts.size();
	}
}
#endif