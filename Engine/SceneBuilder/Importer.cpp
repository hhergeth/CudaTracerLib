#include <StdAfx.h>
#include "CudaBVH.hpp"
#include "..\..\Base\FileStream.h"
#include "..\..\Base\FrameworkInterop.h"

class TmpInStream : public FW::InputStream
{
public:
	TmpInStream       (::InputStream* A)                      { F = A; }
	virtual ~TmpInStream      (void)
	{

	}
	virtual int             read                    (void* ptr, int size)
	{
		F->Read(ptr, size);
		return size;
	}

private:
	::InputStream* F;
};

class TmpOutStream : public FW::OutputStream
{
public:
	TmpOutStream(::OutputStream* A)
	{
		F = A;
	}
	virtual                 ~TmpOutStream     (void)
	{

	}

	virtual void            write                   (const void* ptr, int size)
	{
		F->Write(ptr, size);
	}
	virtual void            flush                   (void)
	{

	}
private:
	::OutputStream* F;
};

void ConstructBVH2(FW::MeshBase* M, ::OutputStream& O)
{
	FW::BVH::BuildParams m_buildParams;
	FW::BVH::Stats stats;
	FW::Scene* m_scene = new FW::Scene(*M);
    FW::BVH bvh(m_scene, FW::Platform(), m_buildParams);
    stats.print();
	FW::CudaBVH* m_bvh = new FW::CudaBVH(bvh, BVHLayout_Compact2);
	TmpOutStream t(&O);
	m_bvh->serialize(t);
	delete m_bvh;
	delete m_scene;
}

#include "Importer.h"
void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O)
{/*
	FW::Mesh<FW::VertexP> M;
	M.addSubmesh();
	M.addVertices((FW::VertexP*)vertices, vCount);
	M.setIndices(0, (int*)indices, cCount);
	ConstructBVH2(&M, O);return;*/
	
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	O << 5u;
	O << (unsigned long long)c.nodeIndex * 64;
	if(c.nodeIndex)
		O.Write(c.nodes, (unsigned int)c.nodeIndex * sizeof(e_BVHNodeData));
	O << (unsigned long long)c.triIndex * 16;
	if(c.triIndex )
		O.Write(c.tris, (unsigned int)c.triIndex * 16);
	O << (unsigned long long)c.triIndex * 4;
	if(c.triIndex )
		O.Write(c.indices, (unsigned int)c.triIndex * sizeof(int));
	c.Free();
}