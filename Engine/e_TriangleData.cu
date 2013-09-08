#include "e_TriangleData.h"
#include "e_Node.h"

Frame e_TriangleData::lerpFrame(const float2& bCoords, const float4x4& localToWorld, float3* ng) const
{
	//float3 na = NOR[0], nb = NOR[1], nc = NOR[2];
	uint4 q = m_sDeviceData.Row0;
	float3 na = Uchar2ToNormalizedFloat3(q.x), nb = Uchar2ToNormalizedFloat3(q.x >> 16), nc = Uchar2ToNormalizedFloat3(q.y);
	float3 ta = Uchar2ToNormalizedFloat3(q.y >> 16), tb = Uchar2ToNormalizedFloat3(q.z), tc = Uchar2ToNormalizedFloat3(q.z >> 16);
	Frame sys;
	float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;
	sys.n = (u * na + v * nb + w * nc);
	sys.t = (u * ta + v * tb + w * tc);

	sys = sys * localToWorld;
	sys.n = normalize(sys.n);
	sys.s = normalize(cross(sys.t, sys.n));
	sys.t = normalize(cross(sys.s, sys.n));
	if(ng)
		*ng = normalize(localToWorld.TransformNormal((na + nb + nc) / 3.0f));//that aint the def of a face normal but to get the vertices we would have to invert the matrix
	return sys;
}

float2 e_TriangleData::lerpUV(const float2& bCoords) const
{
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
	#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x & 0xffff0000)).ToFloat())
#endif
	float2 a = tof2(m_sDeviceData.Row1.x), b = tof2(m_sDeviceData.Row1.y), c = tof2(m_sDeviceData.Row1.z);
	float u = bCoords.y, v = 1.0f - u - bCoords.x;
	return a + u * (b - a) + v * (c - a);
#undef tof2
}

bool TraceResult::hasHit() const
{
	return m_pTri != 0;
}

TraceResult::operator bool() const
{
	return hasHit();
}

void TraceResult::Init(bool first)
{
	m_fDist = FLT_MAX;
	if(first)
	{
		m_pNode = 0;
		__internal__earlyExit = -1;
	}
	m_pTri = 0;
}

Frame TraceResult::lerpFrame() const
{
	return m_pTri->lerpFrame(m_fUV, m_pNode->getWorldMatrix(), 0);
}

unsigned int TraceResult::getMatIndex() const
{
	return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
}

float2 TraceResult::lerpUV() const
{
	return m_pTri->lerpUV(m_fUV);
}

void TraceResult::getBsdfSample(const Ray& r, CudaRNG _rng, BSDFSamplingRecord* bRec) const
{
	bRec->Clear(_rng);
	bRec->map.P = r(m_fDist);
	bRec->map.sys = m_pTri->lerpFrame(m_fUV, m_pNode->getWorldMatrix(), &bRec->ng);
	/*if(dot(r.direction, bRec->ng) > 0.0f)
	{
		bRec->ng *= -1.0f;
		bRec->map.sys.n *= -1.0f;
	}*/
	bRec->map.uv = lerpUV();
	bRec->wi = normalize(bRec->map.sys.toLocal(-1.0f * r.direction));
	float3 nor;
	if(getMat().SampleNormalMap(bRec->map, &nor))
		bRec->map.sys.RecalculateFromNormal(normalize(bRec->map.sys.toWorld(nor)));
}

void TraceResult::getBsdfSample(const Ray& r, CudaRNG _rng, BSDFSamplingRecord* bRec, const float3& wo) const
{
	getBsdfSample(r, _rng, bRec);
	bRec->wo = bRec->map.sys.toLocal(wo);
}