#pragma once

void ConstructBVH(FW::Mesh<FW::VertexP>& M, FW::OutputStream& O, float4** a_Nodes);

void exportBVH(char* Input, char* Output);

void ConstructBVH2(FW::MeshBase* M, FW::OutputStream& O);