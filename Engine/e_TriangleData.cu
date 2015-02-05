#include "e_TriangleData.h"
#include "e_Node.h"
#include "..\Math\Compression.h"

#ifdef EXT_TRI

e_TriangleData::e_TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const Vec3f* N, const Vec3f* Tan, const Vec3f* BiTan)
{
	m_sHostData.MatIndex = matIndex;
	setUvSetData(0, T[0], T[1], T[2]);
	setData(P[0], P[1], P[2], N[0], N[1], N[2]);
}

void e_TriangleData::setUvSetData(int setId, const Vec2f& a, const Vec2f& b, const Vec2f& c)
{
	half2 a1 = half2(a), b1 = half2(b), c1 = half2(c);
	m_sHostData.UV_Sets[setId].TexCoord[0] = *(ushort2*)&a1;
	m_sHostData.UV_Sets[setId].TexCoord[1] = *(ushort2*)&b1;
	m_sHostData.UV_Sets[setId].TexCoord[2] = *(ushort2*)&c1;
}

void e_TriangleData::setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
							 const Vec3f& n0, const Vec3f& n1, const Vec3f& n2)
{
	uint3 uvset = m_sDeviceData.UVSets[0];
	Vec2f   t0 = Vec2f(half(unsigned short(uvset.x)), half(unsigned short(uvset.x >> 16))),
			t1 = Vec2f(half(unsigned short(uvset.y)), half(unsigned short(uvset.y >> 16))),
			t2 = Vec2f(half(unsigned short(uvset.z)), half(unsigned short(uvset.z >> 16)));

	Vec3f dP1 = v1 - v0, dP2 = v2 - v0;
	Vec2f dUV1 = t1 - t0, dUV2 = t2 - t0;
	Vec3f n = normalize(cross(dP1, dP2));
	float determinant = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
	Vec3f dpdu, dpdv;
	if (determinant == 0)
	{
		coordinateSystem(n, dpdu, dpdv);
	}
	else
	{
		float invDet = 1.0f / determinant;
		dpdu = (( dUV2.y * dP1 - dUV1.y * dP2) * invDet);
		dpdv = ((-dUV2.x * dP1 + dUV1.x * dP2) * invDet);
	}

	half3 a = half3(dpdu), b = half3(dpdv);
	m_sDeviceData.NorMatExtra.x = NormalizedFloat3ToUchar2(n0) | (NormalizedFloat3ToUchar2(n1) << 16);
	m_sDeviceData.NorMatExtra.y = NormalizedFloat3ToUchar2(n2) | (m_sDeviceData.NorMatExtra.y & 0xffff0000);
	m_sDeviceData.DpduDpdv.x = a.x.bits() | (a.y.bits() << 16);
	m_sDeviceData.DpduDpdv.y = a.z.bits() | (b.x.bits() << 16);
	m_sDeviceData.DpduDpdv.z = b.y.bits() | (b.z.bits() << 16);
}

void e_TriangleData::fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const
{
	uint2 nme = m_sDeviceData.NorMatExtra;
	Vec3f na = Uchar2ToNormalizedFloat3(nme.x), nb = Uchar2ToNormalizedFloat3(nme.x >> 16), nc = Uchar2ToNormalizedFloat3(nme.y);
	float w = 1.0f - dg.bary.x - dg.bary.y, u = dg.bary.x, v = dg.bary.y;
	dg.sys.n = u * na + v * nb + w * nc;
	uint3 dpd = m_sDeviceData.DpduDpdv;
	Vec3f dpdu = Vec3f(half(unsigned short(dpd.x)), half(unsigned short(dpd.x >> 16)), half(unsigned short(dpd.y)));
	Vec3f dpdv = Vec3f(half(unsigned short(dpd.y >> 16)), half(unsigned short(dpd.z)), half(unsigned short(dpd.z >> 16)));
	dg.sys.s = dpdu - dg.sys.n * dot(dg.sys.n, dpdu);
	dg.sys.t = cross(dg.sys.s, dg.sys.n);
	dg.sys = dg.sys * localToWorld;
	dg.n = normalize(worldToLocal.TransformTranspose(Vec4f(na + nb + nc, 0.0f)).getXYZ());
	dg.dpdu = (localToWorld.TransformDirection(dpdu));
	dg.dpdv = (localToWorld.TransformDirection(dpdv));
	for (int i = 0; i < NUM_UV_SETS; i++)
	{
		uint3 uvset = m_sDeviceData.UVSets[i];
		Vec2f   ta = Vec2f(half(unsigned short(uvset.x)), half(unsigned short(uvset.x >> 16))),
				tb = Vec2f(half(unsigned short(uvset.y)), half(unsigned short(uvset.y >> 16))),
				tc = Vec2f(half(unsigned short(uvset.z)), half(unsigned short(uvset.z >> 16)));
		dg.uv[i] = u * ta + v * tb + w * tc;
	}
	dg.extraData = nme.y >> 24;

	if (dot(dg.n, dg.sys.n) < 0.0f)
		dg.n = -dg.n;
}
#endif