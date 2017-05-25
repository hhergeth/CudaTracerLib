#include <StdAfx.h>
#include <Engine/AnimatedMesh.h>
#include "MD5Parser.h"
#include <Base/FileStream.h>
#include <Engine/TriangleData.h>
#include <Engine/Material.h>
#include <Engine/TriIntersectorData.h>

namespace CudaTracerLib {

void build_Animation(Anim* A, MD5Model* M, Animation& res, const std::string& name, const std::vector<NormalizedT<OrthogonalAffineMap>>& inverseJoints)
{
	res.m_sName = name;
	res.m_uFrameRate = A->bFrameRate;
	for (int idxFrame = 0; idxFrame < A->numbFrames; idxFrame++)
	{
		AnimationFrame frame;
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

void compilemd5(IInStream& in, std::vector<IInStream*>& animFiles, FileOutputStream& a_Out)
{
	MD5Model M;
	M.loadMesh(in.getFilePath().c_str());
	for (unsigned int i = 0; i < animFiles.size(); i++)
		M.loadAnim(animFiles[i]->getFilePath().c_str());

	std::vector<uint3> triData2;
	std::vector<Vec3f> v_Pos;
	std::vector<Vec2f> tCoord;
	std::vector<AnimatedVertex> v_Data;
	std::vector<unsigned int> tData;
	std::vector<Material> matData;
	std::vector<unsigned int> submeshData;
	unsigned int off = 0;
	for (int i = 0; i < M.meshes.size(); i++)
	{
		submeshData.push_back((unsigned int)M.meshes[i]->tris.size());

		Material mat(M.meshes[i]->texture.c_str());
		diffuse stdMaterial;
		stdMaterial.m_reflectance = CreateTexture(Spectrum(1, 0, 0));
		mat.bsdf.SetData(stdMaterial);
		matData.push_back(mat);

		for (int v = 0; v < M.meshes[i]->verts.size(); v++)
		{
			AnimatedVertex av;
			Vec3f pos = Vec3f(0);
			Vertex& V = M.meshes[i]->verts[v];
			int a = min(8, V.weightCount);
			for (int k = 0; k < a; k++)
			{
				const Weight &w = M.meshes[i]->weights[V.weightIndex + k];
				const Joint &joint = M.joints[w.joint];
				const Vec3f r = joint.quat.toMatrix().TransformPoint(w.pos);
				pos += (joint.pos + r) * w.w;
				((unsigned char*)&av.m_cBoneIndices)[k] = (unsigned char)w.joint;
				((unsigned char*)&av.m_fBoneWeights)[k] = (unsigned char)(w.w * 255.0f);
			}
			av.m_fVertexPos = pos;
			v_Pos.push_back(pos);
			tCoord.push_back(V.tc);
			v_Data.push_back(av);
		}

		for (size_t t = 0; t < M.meshes[i]->tris.size(); t++)
		{
			for (int j = 0; j < 3; j++)
				tData.push_back(M.meshes[i]->tris[t].v[j] + off);
			Vec3u q = *(Vec3u*)&M.meshes[i]->tris[t].v + Vec3u(off);
			triData2.push_back(q);
		}
		off += (unsigned int)M.meshes[i]->verts.size();
	}
	unsigned int m_numVertices = (unsigned int)v_Data.size();
	std::vector<NormalizedT<Vec3f>> normals(m_numVertices);
	Mesh::ComputeVertexNormals(&v_Pos[0], &tData[0], (unsigned int)v_Pos.size(), (unsigned int)tData.size() / 3, &normals[0], false);
	for (unsigned int v = 0; v < m_numVertices; v++)
		v_Data[v].m_fNormal = normals[v];

	const Vec2f* uv_sets[1] = {&tCoord[0]};
	Mesh::CompileMesh(&v_Pos[0], m_numVertices, &normals[0], uv_sets, 1, &tData[0], (unsigned int)tData.size(), &matData[0], 0, &submeshData[0], 0, a_Out);

	e_KernelAnimatedMesh mesh;
	mesh.m_uAnimCount = (unsigned int)M.anims.size();
	mesh.m_uJointCount = (unsigned int)M.joints.size();
	mesh.m_uVertexCount = (unsigned int)v_Pos.size();
	a_Out.Write(mesh);
	std::vector<NormalizedT<OrthogonalAffineMap>> inverseJoints;
	for (unsigned int i = 0; i < M.joints.size(); i++)
		inverseJoints.push_back((float4x4::Translate(M.joints[i].pos) % M.joints[i].quat.toMatrix()).inverse());
	for (int a = 0; a < M.anims.size(); a++)
	{
		Animation anim;
		build_Animation(M.anims[a], &M, anim, animFiles[a]->getFilePath(), inverseJoints);
		anim.serialize(a_Out);
	}
	a_Out.Write(&v_Data[0], (unsigned int)v_Pos.size() * sizeof(AnimatedVertex));
	a_Out.Write(&triData2[0], (unsigned int)triData2.size() * sizeof(uint3));
}

}