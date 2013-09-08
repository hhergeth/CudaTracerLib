#pragma once

struct e_SceneInitData
{
	unsigned int m_uNumTriangles;
	unsigned int m_uNumInt;
	unsigned int m_uNumBvhNodes;
	unsigned int m_uNumBvhIndices;
	unsigned int m_uNumMaterials;
	unsigned int m_uNumTextures;
	unsigned int m_uNumNodes;
	unsigned int m_uNumLights;
	unsigned int m_uSizeAnimStream;

	static e_SceneInitData CreateForSpecificMesh(unsigned int a_Triangles, unsigned int a_Int, unsigned int a_Nodes, unsigned int a_Indices, unsigned int a_Mats, unsigned int a_Lights, unsigned int a_SceneNodes = 16)
	{
		e_SceneInitData r;
		r.m_uNumTriangles = a_Triangles;
		r.m_uNumInt = a_Int;
		r.m_uNumBvhNodes = a_Nodes;
		r.m_uNumBvhIndices = a_Indices;
		r.m_uNumMaterials = r.m_uNumTextures = a_Mats;
		r.m_uNumNodes = a_SceneNodes;
		r.m_uNumLights = a_Lights;
		r.m_uSizeAnimStream = 16;
		return r;
	}

	static e_SceneInitData CreateForScene(unsigned int a_NumObjects, unsigned int a_NumAvgTriPerObj, unsigned int a_NumAvgMatPerObj = 5, unsigned int a_NumLights = 1 << 10, unsigned int a_AnimSize = 0)
	{
		e_SceneInitData r;
		r.m_uNumTriangles = a_NumObjects * a_NumAvgTriPerObj;
		r.m_uNumBvhNodes = a_NumObjects * a_NumAvgTriPerObj / 2;
		r.m_uNumBvhIndices = a_NumObjects * a_NumAvgTriPerObj * 4;
		r.m_uNumMaterials = r.m_uNumTextures = a_NumObjects * a_NumAvgMatPerObj;
		r.m_uNumNodes = a_NumObjects;
		r.m_uNumLights = a_NumLights;
		r.m_uSizeAnimStream = a_AnimSize;
		return r;
	}

	static e_SceneInitData CreateFor_S_SanMiguel(unsigned int a_SceneNodes = 16, unsigned int a_Lights = 16);

	e_SceneInitData e_SceneInitData::operator*=(unsigned int s)
	{
		e_SceneInitData r = *this;
		r.m_uNumTriangles *= s;
		r.m_uNumInt *= s;
		r.m_uNumBvhNodes *= s;
		r.m_uNumBvhIndices *= s;
		r.m_uNumTextures *= s;
		r.m_uNumNodes *= s;
		r.m_uNumLights *= s;
		r.m_uSizeAnimStream *= s;
		return r;
	}
};