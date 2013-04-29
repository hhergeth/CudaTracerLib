#include "StdAfx.h"
#include "k_sPpmTracer.h"
#include <base\Sort.hpp>

struct sortData
{
	float3 mi, ma, dif;
	k_Kernel_sPpmBuffers Buf;
	sortData(k_sPpmBuffers* A, float3& _mi, float3& _ma)
	{
		mi = _mi;
		ma = _ma;
		dif = ma - mi;
		Buf = k_Kernel_sPpmBuffers(A, false);
	}
	sortData(){}
};

unsigned int hash(float3& p, sortData* D)
{
	float3 q = (p - D->mi) / D->dif;
	if(q.x <= 0 || q.x >= 1 || q.y <= 0 || q.y >= 1 || q.z <= 0 || q.z >= 1)
		return -1;
	q *= GRID_SUBS;
	uint3 a = make_uint3(q);
	return CALC_INDEX(a);
}

bool cmp(void* v, int i, int j)
{
	sortData* d = (sortData*)v;
	return hash(d->Buf.b1[i].Pos, d) < hash(d->Buf.b1[j].Pos, d);
}

void swapEntries(void* v, int i, int j)
{
	sortData* d = (sortData*)v;
	d->Buf.swapEntries(i, j);
}

void k_sPpmTracer::SortHostData()
{
	sortData d(m_pEyeHits, m_vLow, m_vHigh);
	FW::sort(&d, 0, m_uNumEyeHits, cmp, swapEntries, true);
	unsigned int i = 0;
	while(i < m_uNumEyeHits)
	{
		unsigned int c = 0;
		unsigned int h0 = hash(d.Buf.b1[i].Pos, &d);
		while(hash(d.Buf.b1[i + c].Pos, &d) == h0)
			c++;
#ifdef USE_BLOCK
		for(int j = 1; j < c; j++)
			d.Buf.b0[i + j].ref = i;
#endif
		m_pGrid->m_pHost[h0] = make_uint2(i, c);
		i += c;
	}
	m_uLastValidIndex = m_uNumEyeHits;
}