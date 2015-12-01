#pragma once

#include "MathFunc.h"
#include "Vector.h"

//Compresses normalized float3s to ushorts. Either by using spherical coordinates, which are heavy on ALUs.
//The alternative is scaling the x and y components to [0,255] and storing the z in one bit of the others data.

namespace CudaTracerLib {

#define SPHERICAL_COMPRESSION

#ifdef SPHERICAL_COMPRESSION

CUDA_FUNC_IN unsigned short NormalizedFloat3ToUchar2(const Vec3f& v)
{
	float theta = (acos(v.z)*(255.0f / PI));
	float phi = (atan2(v.y, v.x)*(255.0f / (2.0f*PI)));
	phi = phi < 0 ? (phi + 255) : phi;
	return ((unsigned short)theta << 8) | (unsigned short)phi;
}

CUDA_FUNC_IN Vec3f Uchar2ToNormalizedFloat3(unsigned short v)
{
	unsigned char x = v >> 8, y = v & 0xff;
	float theta = float(x)*(1.0f / 255.0f)*PI;
	float phi = float(y)*(1.0f / 255.0f)*PI*2.0f;
	float sinphi, cosphi, costheta, sintheta;
	sincos(phi, &sinphi, &cosphi);
	sincos(theta, &sintheta, &costheta);
	return Vec3f(sintheta*cosphi, sintheta * sinphi, costheta);
}

#else

#define SCALE 63.0f
CUDA_FUNC_IN unsigned short NormalizedFloat3ToUchar2(const Vec3f& v)
{
	int x = Round2Int(v.x * SCALE + SCALE);//x € [0, 126]
	int y = Round2Int(v.y * SCALE + SCALE) << 8;
	int z = v.z > 0.0f ? 32768 : 0;
	return (unsigned short)(x | y | z);
}

CUDA_FUNC_IN Vec3f Uchar2ToNormalizedFloat3(unsigned short v)
{
	Vec3f r;
	r.x = float((v & 127) - SCALE) / SCALE;
	r.y = float(((v >> 8) & 127) - SCALE) / SCALE;
	r.z = math::sqrt(math::clamp(1.0f - r.x * r.x - r.y * r.y, 0.01f, 1.0f)) * (v & 32768 ? 1.0f : -1.0f);
	return r;

}

#endif

CUDA_FUNC_IN uchar3 NormalizedFloat3ToUchar3(Vec3f& v)
{
#define CNV(x) x * 127.0f + 127.0f
	return make_uchar3(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}

CUDA_FUNC_IN Vec3f Uchar3ToNormalizedFloat3(uchar3 v)
{
#define CNV(x) (float(x) - 127.0f) / 127.0f
	return Vec3f(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}

}
