#include <StdAfx.h>
#include "e_SceneBVH.h"
#include "e_Node.h"
#include "e_Mesh.h"

#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#define NO_NODE 0x76543210

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
	boost::property<boost::vertex_name_t, std::string> > Graph;

void write(int idx, e_BVHNodeData* nodes, int parent, std::ofstream& f, int& leafC)
{
	if (parent != -1)
		f << parent << " -> " << idx << ";\n";
	Vec2i c = nodes[idx].getChildren();
	if (c.x > 0 && c.x != NO_NODE)
		write(c.x / 4, nodes, idx, f, leafC);
	else if (c.x < 0)
		f << idx << " -> " << c.x << ";\n";
	if (c.y > 0 && c.y != NO_NODE)
		write(c.y / 4, nodes, idx, f, leafC);
	else if (c.y < 0)
		f << idx << " -> " << c.y << ";\n";
}

void e_SceneBVH::printGraph(const std::string& path, e_BufferReference<e_Node, e_Node> a_Nodes)
{
	std::ofstream file;
	file.open(path);
	file << "digraph SceneBVH {\nnode [fontname=\"Arial\"];\n";
	int leafC = 0;
	write(startNode / 4, m_pNodes->operator()(0), -1, file, leafC);
	file << "}";
	file.close();
}