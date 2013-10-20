#pragma once

#include "cutil_math.h"

#define SCALE 63.0f
CUDA_FUNC_IN unsigned short NormalizedFloat3ToUchar2(const float3& v)
{/*
	float theta = (acos(v.z)*(255.0f/PI));
	float phi = (atan2(v.y,v.x)*(255.0f/(2.0f*PI)));
	phi = phi < 0 ? (phi + 255) : phi;
	return make_uchar2((unsigned char)theta, (unsigned char)phi);*/
	int x = Round2Int(v.x * SCALE + SCALE);//x € [0, 126]
	int y = Round2Int(v.y * SCALE + SCALE) << 8;
	int z = v.z > 0.0f ? 32768 : 0;
	return (unsigned short)(x | y | z);
	/*float2 q = normalize(make_float2(v.y,v.z));
	float d = v.x <= 0.0f ? q.x + 1.0f : 3.0f - q.x;
	return make_uchar2((unsigned char)(d / 4.0f * 255.0f), (unsigned char)(q.y * 127.0f + 127.0f));*/
	/*unsigned int x = unsigned int(v.x * 16.0f + 16.0f), y = unsigned int(v.y * 16.0f + 16.0f), z = unsigned int(v.z * 16.0f + 16.0f);
	unsigned short s = (unsigned short)((x << 10) | (y << 5) | z);
	return *(uchar2*)&s;*/
}

CUDA_FUNC_IN float3 Uchar2ToNormalizedFloat3(unsigned short v)
{/*
	float theta = float(v.x)*(1.0f/255.0f)*PI;
	float phi = float(v.y)*(1.0f/255.0f)*PI*2.0f;
	float sinphi, cosphi, costheta, sintheta;
	sincos(phi, &sinphi, &cosphi);
	sincos(theta, &sintheta, &costheta);
	return make_float3(sintheta*cosphi, sintheta * sinphi, costheta);*/
	float3 r;
	r.x = float((v & 127) - SCALE) / SCALE;
	r.y = float(((v >> 8) & 127) - SCALE) / SCALE;
	//r.x = clamp01((float(v.x & 63)) / (2.0f * SCALE)) * 2.0f - 1.0f;
	//r.y = clamp01((float(v.y & 63)) / (2.0f * SCALE)) * 2.0f - 1.0f;
	r.z = sqrtf(clamp(1.0f - r.x * r.x - r.y * r.y, 0.01f, 1.0f)) * (v & 32768 ? 1.0f : -1.0f);
	return r;
	/*float d = float(v.x) / 255.0f * 4.0f;
	float y = d < 2.0f ? d - 1.0f : 3.0f - d;
	float x = sqrtf(1.0f - y * y), z = (float(v.y) - 127.0f) / 127.0f;
	return make_float3(x, y, z);*/
	/*unsigned short s = *(unsigned short*)&v;
	unsigned int x = (s >> 10) & 31, y = (s >> 5) & 31, z = s & 31;
	return make_float3(float(x - 16.0f) / 16.0f, float(y - 16.0f) / 16.0f, float(z - 16.0f) / 16.0f);*/
}

CUDA_FUNC_IN uchar3 NormalizedFloat3ToUchar3(float3& v)
{
#define CNV(x) x * 127.0f + 127.0f
	return make_uchar3(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}

CUDA_FUNC_IN float3 Uchar3ToNormalizedFloat3(uchar3 v)
{
#define CNV(x) (float(x) - 127.0f) / 127.0f
	return make_float3(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}