#include "e_TriangleData.h"
#include "e_Node.h"
#include "..\Math\Compression.h"

#ifdef EXT_TRI

e_TriangleData::e_TriangleData(const float3* P, unsigned char matIndex, const float2* T, const float3* N, const float3* Tan, const float3* BiTan)
{
	m_sHostData.MatIndex = matIndex;
	setUvSetData(0, T[0], T[1], T[2]);
	setData(P[0], P[1], P[2], N[0], N[1], N[2]);
}

void e_TriangleData::setData(const float3& v0, const float3& v1, const float3& v2,
							 const float3& n0, const float3& n1, const float3& n2)
{
	float3 dP1 = v1 - v0, dP2 = v2 - v0;
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x >> 16)).ToFloat())
#endif
	uint3 val = m_sDeviceData.RowX[0];
	float2 t0 = tof2(val.x), t1 = tof2(val.y), t2 = tof2(val.z);
	float2 dUV1 = t1 - t0, dUV2 = t2 - t0;
	float3 n = normalize(cross(dP1, dP2));
	float determinant = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
	//float3 dpdu, dpdv;
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
	m_sDeviceData.Row0.x = NormalizedFloat3ToUchar2(n0) | (NormalizedFloat3ToUchar2(n1) << 16);
	m_sDeviceData.Row0.y = NormalizedFloat3ToUchar2(n2) | (NormalizedFloat3ToUchar2(dpdu) << 16);
	m_sDeviceData.Row0.z = NormalizedFloat3ToUchar2(dpdv) | (NormalizedFloat3ToUchar2(n) << 16);
}

void e_TriangleData::fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const
{
	uint4 q = m_sDeviceData.Row0;
	float3 na = Uchar2ToNormalizedFloat3(q.x), nb = Uchar2ToNormalizedFloat3(q.x >> 16), nc = Uchar2ToNormalizedFloat3(q.y);
	float3 /*dpdu = Uchar2ToNormalizedFloat3(q.y >> 16), dpdv = Uchar2ToNormalizedFloat3(q.z),*/ faceN = Uchar2ToNormalizedFloat3(q.z >> 16);
	float w = 1.0f - dg.bary.x - dg.bary.y, u = dg.bary.x, v = dg.bary.y;
	dg.sys.n = u * na + v * nb + w * nc;
	dg.sys.s = dpdu - dg.sys.n * dot(dg.sys.n, dpdu);
	dg.sys.t = cross(dg.sys.s, dg.sys.n);
	dg.sys = dg.sys * localToWorld;
	dg.n = normalize(!worldToLocal.TransformTranspose(make_float4(faceN, 0.0f)));
	dg.dpdu = (localToWorld.TransformDirection(dpdu));
	dg.dpdv = (localToWorld.TransformDirection(dpdv));
	for (int i = 0; i < NUM_UV_SETS; i++)
		dg.uv[i] = lerpUV(i, dg.bary);
	dg.extraData = lerpExtraData(dg.bary);

	if (dot(dg.n, dg.sys.n) < 0.0f)
		dg.n = -dg.n;
}

float2 e_TriangleData::lerpUV(int setId, const float2& bCoords) const
{
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
	#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x >> 16)).ToFloat())
#endif
	uint3 val = m_sDeviceData.RowX[setId];
	float2 a = tof2(val.x), b = tof2(val.y), c = tof2(val.z);
	//float u = bCoords.y, v = 1.0f - u - bCoords.x;
	//return a + u * (b - a) + v * (c - a);
	float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;
	return u * a + v * b + w * c;
#undef tof2
}

void e_TriangleData::getNormalDerivative(const float2& bCoords, float3& dndu, float3& dndv) const
{
	uint4 q = m_sDeviceData.Row0;
	float3 n0 = Uchar2ToNormalizedFloat3(q.x), n1 = Uchar2ToNormalizedFloat3(q.x >> 16), n2 = Uchar2ToNormalizedFloat3(q.y);
#ifdef ISCUDA
	#define tof2(x) make_float2(__half2float(x & 0xffff), __half2float(x >> 16))
#else
	#define tof2(x) make_float2(half((unsigned short)(x & 0xffff)).ToFloat(), half((unsigned short)(x & 0xffff0000)).ToFloat())
#endif
	float2 uv0 = tof2(m_sDeviceData.RowX[0].x), uv1 = tof2(m_sDeviceData.RowX[0].y), uv2 = tof2(m_sDeviceData.RowX[0].z);
	float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;

	float3 N = u * n1 + v * n2 + w * n0;
	float il = 1.0f / length(N);
	N *= il;
	dndu = (n1 - n0) * il; dndu -= N * dot(N, dndu);
	dndv = (n2 - n0) * il; dndv -= N * dot(N, dndv);

	float2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;
	float det = duv1.x * duv2.y - duv1.y * duv2.x;
	float invDet = 1.0f / det;
	float3 dndu_ = ( duv2.y * dndu - duv1.y * dndv) * invDet;
	float3 dndv_ = (-duv2.x * dndu + duv1.x * dndv) * invDet;
	dndu = dndu_; dndv = dndv_;
}
#endif