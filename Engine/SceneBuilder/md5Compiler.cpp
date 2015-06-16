#include <StdAfx.h>
#include "../e_AnimatedMesh.h"
#include "TangentSpaceHelper.h"
#include <Engine\SceneBuilder\Importer.h>
#include "MD5Parser.h"
#include "Importer.h"

void build_e_Animation(Anim* A, MD5Model* M, e_Animation& res, const std::string& name, const std::vector<float4x4>& inverseJoints)
{
	res.m_sName = name;
	res.m_uFrameRate = A->bFrameRate;
	for (int idxFrame = 0; idxFrame < A->numbFrames; idxFrame++)
	{
		e_Frame frame;
		const bFrame& F = A->bFrames[idxFrame];
		for (int j = 0; j < F.joints.size(); j++)
		{
			const Joint& joint = F.joints[j];
			frame.m_sHostConstructionData.push_back(float4x4::Translate(joint.pos) % joint.quat.toMatrix());
			if (joint.parent != -1)
				frame.m_sHostConstructionData[j] = frame.m_sHostConstructionData[joint.parent] % frame.m_sHostConstructionData[j];
		}
		for (int j = 0; j < F.joints.size(); j++)
			frame.m_sHostConstructionData[j] = frame.m_sHostConstructionData[j] % inverseJoints[j];
		res.m_pFrames.push_back(frame);
	}
}

void compilemd5(IInStream& in, std::vector<IInStream*>& animFiles, OutputStream& a_Out)
{
	MD5Model M;
	M.loadMesh(in.getFilePath().c_str());
	for (unsigned int i = 0; i < animFiles.size(); i++)
		M.loadAnim(animFiles[i]->getFilePath().c_str());
	AABB box;
	box = box.Identity();
	std::vector<uint3> triData2;
	std::vector<Vec3f> v_Pos;
	std::vector<Vec2f> tCoord;
	std::vector<e_AnimatedVertex> v_Data;
	std::vector<unsigned int> tData;
	unsigned int off = 0;
	for (int i = 0; i < M.meshes.size(); i++)
	{
		for (int v = 0; v < M.meshes[i]->verts.size(); v++)
		{
			e_AnimatedVertex av;
			Vec3f pos = Vec3f(0);
			Vertex& V = M.meshes[i]->verts[v];
			int a = min(g_uMaxWeights, V.weightCount);
			for (int k = 0; k < a; k++)
			{
				const Weight &w = M.meshes[i]->weights[V.weightIndex + k];
				const Joint &joint = M.joints[w.joint];
				const Vec3f r = joint.quat.toMatrix().TransformPoint(w.pos);
				pos += (joint.pos + r) * w.w;
				av.m_cBoneIndices[k] = (unsigned char)w.joint;
				av.m_fBoneWeights[k] = (unsigned char)(w.w * 255.0f);
			}
			av.m_fVertexPos = pos;
			v_Pos.push_back(pos);
			tCoord.push_back(V.tc);
			v_Data.push_back(av);
		}

		for (size_t t = 0; t < M.meshes[i]->tris.size(); t++)
			for (int j = 0; j < 3; j++)
				tData.push_back(M.meshes[i]->tris[t].v[j] + off);
		off += (unsigned int)M.meshes[i]->verts.size();
	}
	unsigned int m_numVertices = (unsigned int)v_Data.size();
	Vec3f* v_Normals = new Vec3f[m_numVertices], *v_Tangents = new Vec3f[m_numVertices], *v_BiTangents = new Vec3f[m_numVertices];
	Platform::SetMemory(v_Normals, sizeof(Vec3f) * m_numVertices);
	Platform::SetMemory(v_Tangents, sizeof(Vec3f) * m_numVertices);
	Platform::SetMemory(v_BiTangents, sizeof(Vec3f) * m_numVertices);
	ComputeTangentSpace(&v_Pos[0], &tCoord[0], &tData[0], (unsigned int)v_Pos.size(), (unsigned int)tData.size() / 3, v_Normals, v_Tangents, v_BiTangents);
	for (unsigned int v = 0; v < m_numVertices; v++)
	{
		v_Data[v].m_fNormal = v_Normals[v];
		v_Data[v].m_fTangent = v_Tangents[v];
		v_Data[v].m_fBitangent = v_BiTangents[v];
	}

	std::vector<e_TriangleData> triData;
	std::vector<e_KernelMaterial> matData;
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	unsigned int lc = 0;
	diffuse stdMaterial;
	stdMaterial.m_reflectance = CreateTexture(Spectrum(1, 0, 0));

	off = 0;
	for (int s = 0; s < M.meshes.size(); s++)
	{
		Mesh* sm = M.meshes[s];

		e_KernelMaterial mat(sm->texture.c_str());
		mat.NodeLightIndex = -1;
		mat.bsdf.SetData(stdMaterial);
		matData.push_back(mat);

		size_t st = triData2.size();

		for (int t = 0; t < sm->tris.size(); t++)
		{
			Vec3f P[3], N[3], Tan[3], BiTan[3];
			Vec2f T[3];
			for (int j = 0; j < 3; j++)
			{
				unsigned int v = sm->tris[t].v[j] + off;
				P[j] = v_Pos[v];
				T[j] = sm->verts[v - off].tc;
				box.Enlarge(P[j]);
			}
			e_TriangleData d(P, s, T, N, Tan, BiTan);
			triData.push_back(d);
			Vec3u q = *(Vec3u*)&sm->tris[t].v + Vec3u(off);
			triData2.push_back(q);
		}
		off += (unsigned int)sm->verts.size();
	}

	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lc;

	a_Out << (unsigned int)triData.size();
	a_Out.Write(&triData[0], sizeof(e_TriangleData) * (unsigned int)triData.size());
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	BVH_Construction_Result bvh;
	ConstructBVH(&v_Pos[0], (unsigned int*)&triData2[0], (int)v_Pos.size(), (int)triData2.size() * 3, a_Out, &bvh);

	e_KernelAnimatedMesh mesh;
	mesh.m_uAnimCount = (unsigned int)M.anims.size();
	mesh.m_uJointCount = (unsigned int)M.joints.size();
	mesh.m_uVertexCount = (unsigned int)v_Pos.size();
	a_Out.Write(mesh);
	std::vector<float4x4> inverseJoints;
	for (unsigned int i = 0; i < M.joints.size(); i++)
		inverseJoints.push_back((float4x4::Translate(M.joints[i].pos) % M.joints[i].quat.toMatrix()).inverse());
	for (int a = 0; a < M.anims.size(); a++)
	{
		e_Animation anim;
		build_e_Animation(M.anims[a], &M, anim, animFiles[a]->getFilePath(), inverseJoints);
		anim.serialize(a_Out);
	}
	a_Out.Write(&v_Data[0], (unsigned int)v_Pos.size() * sizeof(e_AnimatedVertex));
	a_Out.Write(&triData2[0], (unsigned int)triData2.size() * sizeof(uint3));
	///delete[] v_Data;
	//delete[] v_Pos;
}