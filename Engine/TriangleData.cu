#include "TriangleData.h"
#include "Node.h"
#include <Math/Compression.h>

namespace CudaTracerLib {

#ifdef EXT_TRI

TriangleData::TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const NormalizedT<Vec3f>* N, const Vec3f* Tan, const Vec3f* BiTan)
{
	m_sHostData.MatIndex = matIndex;
	setUvSetData(0, T[0], T[1], T[2]);
	setData(P[0], P[1], P[2], N[0], N[1], N[2]);
}

void TriangleData::setUvSetData(int setId, const Vec2f& a, const Vec2f& b, const Vec2f& c)
{
	half2 a1 = half2(a), b1 = half2(b), c1 = half2(c);
	m_sHostData.UV_Sets[setId].TexCoord[0] = *(ushort2*)&a1;
	m_sHostData.UV_Sets[setId].TexCoord[1] = *(ushort2*)&b1;
	m_sHostData.UV_Sets[setId].TexCoord[2] = *(ushort2*)&c1;
}

void TriangleData::getUVSetData(int setId, Vec2f& a, Vec2f& b, Vec2f& c) const
{
#define ToVec2f(V) Vec2f(half((unsigned short)(V >> 16)).ToFloat(), half((unsigned short)(V & 0xffff)).ToFloat())
	a = ToVec2f(m_sDeviceData.UVSets[setId].x);
	b = ToVec2f(m_sDeviceData.UVSets[setId].y);
	c = ToVec2f(m_sDeviceData.UVSets[setId].z);
#undef ToVec2f
}

void TriangleData::setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
						   const NormalizedT<Vec3f>& n0, const NormalizedT<Vec3f>& n1, const NormalizedT<Vec3f>& n2)
{
	uint3 uvset = m_sDeviceData.UVSets[0];
	Vec2f   t0 = Vec2f(half((unsigned short)uvset.x), half((unsigned short)(uvset.x >> 16))),
		t1 = Vec2f(half((unsigned short)uvset.y), half((unsigned short)(uvset.y >> 16))),
		t2 = Vec2f(half((unsigned short)uvset.z), half((unsigned short)(uvset.z >> 16)));

	Vec3f dP1 = v1 - v0, dP2 = v2 - v0;
	Vec2f dUV1 = t1 - t0, dUV2 = t2 - t0;
	NormalizedT<Vec3f> n = normalize(cross(dP1, dP2));
	float determinant = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
	Vec3f dpdu, dpdv;
	if (determinant == 0)
	{
		NormalizedT<Vec3f> a, b;
		coordinateSystem(n, a, b);
		dpdu = a; dpdv = b;
	}
	else
	{
		float invDet = 1.0f / determinant;
		dpdu = ((dUV2.y * dP1 - dUV1.y * dP2) * invDet);
		dpdv = ((-dUV2.x * dP1 + dUV1.x * dP2) * invDet);
	}

	half3 a = half3(dpdu), b = half3(dpdv);
	m_sDeviceData.NorMatExtra.x = NormalizedFloat3ToUchar2(n0) | (NormalizedFloat3ToUchar2(n1) << 16);
	m_sDeviceData.NorMatExtra.y = NormalizedFloat3ToUchar2(n2) | (m_sDeviceData.NorMatExtra.y & 0xffff0000);
	m_sDeviceData.DpduDpdv.x = a.x.bits() | (a.y.bits() << 16);
	m_sDeviceData.DpduDpdv.y = a.z.bits() | (b.x.bits() << 16);
	m_sDeviceData.DpduDpdv.z = b.y.bits() | (b.z.bits() << 16);
}

void TriangleData::getNormals(NormalizedT<Vec3f>& n0, NormalizedT<Vec3f>& n1, NormalizedT<Vec3f>& n2) const
{
	uint2 nme = m_sDeviceData.NorMatExtra;
	n0 = Uchar2ToNormalizedFloat3(nme.x);
	n1 = Uchar2ToNormalizedFloat3(nme.x >> 16);
	n2 = Uchar2ToNormalizedFloat3(nme.y);
}

void TriangleData::fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const
{
	uint2 nme = m_sDeviceData.NorMatExtra;
	Vec3f na = Uchar2ToNormalizedFloat3(nme.x), nb = Uchar2ToNormalizedFloat3(nme.x >> 16), nc = Uchar2ToNormalizedFloat3(nme.y);
	float w = 1.0f - dg.bary.x - dg.bary.y, u = dg.bary.x, v = dg.bary.y;
	Vec3f n = normalize(u * na + v * nb + w * nc);
	uint3 dpd = m_sDeviceData.DpduDpdv;
	Vec3f dpdu = Vec3f(half((unsigned short)dpd.x), half((unsigned short)(dpd.x >> 16)), half((unsigned short)dpd.y));
	Vec3f dpdv = Vec3f(half((unsigned short)(dpd.y >> 16)), half((unsigned short)dpd.z), half((unsigned short)(dpd.z >> 16)));
	Vec3f s = dpdu - n * dot(n, dpdu);
	n = localToWorld.TransformDirection(n); s = localToWorld.TransformDirection(s);
	Vec3f t = cross(s, n);
	dg.sys = Frame(s.normalized(), t.normalized(), n.normalized());
	dg.n = normalize(localToWorld.TransformDirection(na + nb + nc));
	dg.dpdu = (localToWorld.TransformDirection(dpdu));
	dg.dpdv = (localToWorld.TransformDirection(dpdv));
	for (int i = 0; i < NUM_UV_SETS; i++)
	{
		uint3 uvset = m_sDeviceData.UVSets[i];
		Vec2f   ta = Vec2f(half((unsigned short)uvset.x), half((unsigned short)(uvset.x >> 16))),
			tb = Vec2f(half((unsigned short)uvset.y), half((unsigned short)(uvset.y >> 16))),
			tc = Vec2f(half((unsigned short)uvset.z), half((unsigned short)(uvset.z >> 16)));
		dg.uv[i] = u * ta + v * tb + w * tc;
	}
	dg.extraData = nme.y >> 24;

	if (dot(dg.n, dg.sys.n) < 0.0f)
		dg.n = NormalizedT<Vec3f>(-dg.n);
}
#else
TriangleData::TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const NormalizedT<Vec3f>* N, const Vec3f* Tan, const Vec3f* BiTan)
{
	Vec3f p = P[0] - P[2];
	Vec3f q = P[1] - P[2];
	Vec3f d = normalize(cross(q, p));
	m_sHostData.Normal = NormalizedFloat3ToUchar3(d);
	m_sHostData.MatIndex = matIndex;
}

void TriangleData::fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const
{
	NormalizedT<Vec3f> n = normalize(Uchar3ToNormalizedFloat3(m_sHostData.Normal));
	dg.sys = Frame(n) * localToWorld;
	dg.n = normalize(localToWorld.TransformDirection(na + nb + nc));
	dg.dpdu = Vec3f(1,0,0);
	dg.dpdv = Vec3f(0,0,1);
	for (int i = 0; i < NUM_UV_SETS; i++)
		dg.uv[i] = Vec2f(0.0f);
	dg.extraData = 0;
}

void TriangleData::setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
	const NormalizedT<Vec3f>& n0, const NormalizedT<Vec3f>& n1, const NormalizedT<Vec3f>& n2)
{
	m_sHostData.Normal = NormalizedFloat3ToUchar3(n0 + n1 + n2);
}

void TriangleData::getNormals(NormalizedT<Vec3f>& n0, NormalizedT<Vec3f>& n1, NormalizedT<Vec3f>& n2) const
{
	n0 = n1 = n2 = Uchar3ToNormalizedFloat3(m_sHostData.Normal);
}

#endif

}
