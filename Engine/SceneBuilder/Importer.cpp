#include "StdAfx.h"
#include "CudaBVH.hpp"

using namespace FW;

void ConstructBVH(Mesh<VertexP>& M, OutputStream& O, float4** a_Nodes)
{
	Scene S(M);
	BVH B(&S, Platform(), BVH::BuildParams());
	CudaBVH B2(B, BVHLayout_Compact2);
	*a_Nodes = new float4[B2.getNodeBuffer().getSize() / 16];
	memcpy(*a_Nodes, B2.getNodeBuffer().getPtr(),B2.getNodeBuffer().getSize()); 
	B2.serialize(O);
}

#include <io/File.hpp>
void exportBVH(char* Input, char* Output)
{
	FW::BVH::BuildParams m_buildParams;
	FW::BVH::Stats stats;
    m_buildParams.stats = &stats;

	FW::MeshBase* mesh = importMesh(Input);
	FW::Scene* m_scene = new Scene(*mesh);

	FW::printf("\nBuilding BVH...\nThis will take a while.\n");

    // Build BVH.

    FW::BVH bvh(m_scene, FW::Platform(), m_buildParams);
    stats.print();
	FW::CudaBVH* m_bvh = new FW::CudaBVH(bvh, BVHLayout_Compact2);

    // Display status.

    FW::printf("Done.\n\n");
}


void ConstructBVH2(FW::MeshBase* M, FW::OutputStream& O)
{
	FW::BVH::BuildParams m_buildParams;
	FW::BVH::Stats stats;
	FW::Scene* m_scene = new FW::Scene(*M);
    FW::BVH bvh(m_scene, FW::Platform(), m_buildParams);
    stats.print();
	FW::CudaBVH* m_bvh = new FW::CudaBVH(bvh, BVHLayout_Compact2);
	m_bvh->serialize(O);
}