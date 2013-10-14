#pragma once
/*
void ConstructBVH(FW::Mesh<FW::VertexP>& M, FW::OutputStream& O, float4** a_Nodes);

void exportBVH(char* Input, char* Output);

void ConstructBVH2(FW::MeshBase* M, FW::OutputStream& O);
*/
void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, std::vector<e_BVHNodeData>* nodes, std::vector<char>* tris, std::vector<int>* indicesA);
void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O);