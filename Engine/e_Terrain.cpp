#include "StdAfx.h"
#include "e_Terrain.h"
#include <float.h>

e_TerrainData_Leaf* e_Terrain::getValAt(unsigned int x, unsigned int y)
{
	return 0;
	//return (e_TerrainData_Leaf*)m_pStream->operator()(calcIndex(x, y, m_uDepth - 1, m_uDepth));
}

bool upd(e_TerrainData_Inner* root, CACHE_LEVEL_TYPE* data, int d, int t, int x, int y, float2& ret, float2 sdxy)
{
	int i = calcIndex(x, y, d, t);
	ret = make_float2(FLT_MAX, -FLT_MAX);
	int of = calcBoxSize(d + 1, t);
	bool b = false;
	if(d < t - 1)
		for(int x2 = 0; x2 < 2; x2++)
			for(int y2 = 0; y2 < 2; y2++)
			{
				int xc = x + x2 * of, yc = y + y2 * of;
				float2 q;
				bool c = false;
				if(upd(root, data, d + 1, t, xc, yc, q, sdxy))
				{
					c = b = true;
					ret.x = MIN(q.x, ret.x);
					ret.y = MAX(q.y, ret.y);
				}
				else q = make_float2(0, 0);
				root[i].blocks[x2 + y2 * 2] = half2(make_float2(q.x, c ? MAX(q.y - q.x, 1e-2f) : 0));
			}
	else
	{
		e_TerrainData_Leaf* l = (e_TerrainData_Leaf*)(root + i);
		if(l->hasValidHeight())
		{
			float2 ra = l->getRange();
			ret.x = MIN(ra.x, ret.x);
			ret.y = MAX(ra.y, ret.y);
			return true;
		}
	}

	if(d == CACHE_LEVEL)
	{
		float mi = FLT_MAX, ma = -FLT_MAX;
		for(int j = 0; j < 4; j++)
		{
			float2 f = root[i].blocks[j].ToFloat2(); f.y += f.x;
			mi = MIN(mi, f.x);
			ma = MAX(ma, f.y);
		}
		unsigned int s = calcBoxSize(d, t);
		data[y / s * pow2(CACHE_LEVEL) + x / s] = make_float2(mi, ma);
	}

	return b;
}

void e_Terrain::updateFromTriangles()
{
	//upd(m_pStream->getHost(), m_pCacheStream->getHost(), 0, m_uDepth, 0, 0, m_sSpan, getKernelData().getsdxy());
	//m_pStream->Invalidate(DataStreamRefresh_Buffered);
	//m_pCacheStream->Invalidate(DataStreamRefresh_Buffered);
}

void prnt(e_TerrainData_Inner* root,unsigned int lvl, uchar3* data, unsigned int d, unsigned int t, int x, int y, float2 rng, float2 val)
{
	int i = calcIndex(x, y, d, t);
	if(lvl == d)
	{
		int w = pow2(t + 1);
		int of = calcBoxSize(d, t) * 2;
		for(int x2 = 0; x2 < of; x2++)
			for(int  y2 = 0; y2 < of; y2++)
			{
				float a = (val.x - rng.x) / (rng.y - rng.x);
				float b = (val.y - rng.x) / (rng.y - rng.x);
				data[(y + y2) * w + x + x2] = make_uchar3((unsigned char)(a * 255.0f), (unsigned char)(b * 255.0f), 0);
			}
	}
	else
	{
		int of = calcBoxSize(d + 1, t);
		if(d < t - 1)
		{
			e_TerrainData_Inner& n = root[i];
			for(int x2 = 0; x2 < 2; x2++)
				for(int  y2 = 0; y2 < 2; y2++)
					prnt(root, lvl, data, d + 1, t, x + x2, y + y2, rng, n.blocks[y2 * 2 + x2].ToFloat2());
		}
		else
		{

		}
	}
}

void e_Terrain::printLevelMap(unsigned int lvl, uchar3* data)
{
	//prnt(m_pStream->operator()(0u), lvl, data, 0, m_uDepth, 0, 0, m_sSpan, m_sSpan);
}