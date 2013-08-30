#pragma once

#include "cutil_math.h"

CUDA_FUNC_IN float y(float3& v)
{
	const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
	return YWeight[0] * v.x + YWeight[1] * v.y + YWeight[2] * v.z;
}

CUDA_FUNC_IN unsigned int toABGR(const float4& v)
{
    return
        (unsigned int)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((unsigned int)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((unsigned int)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        ((unsigned int)(fminf(fmaxf(v.w, 0.0f), 1.0f) * 255.0f) << 24);
}

CUDA_FUNC_IN unsigned int toABGR(const float3& v)
{
    return
        (unsigned int)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((unsigned int)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((unsigned int)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        (255 << 24);
}

typedef uchar4 RGBCOL; 
#define toInt(x) (int((float)pow(clamp01(x),1.0f/1.2f)*255.0f+0.5f))
//#define toInt(x) (unsigned char(x * 255.0f))

CUDA_FUNC_IN RGBCOL Float4ToCOLORREF(const float4& c)
{
	return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), toInt(c.w));
}

CUDA_FUNC_IN float4 COLORREFToFloat4(RGBCOL c)
{
	return make_float4((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, (float)c.w / 255.0f);
}

CUDA_FUNC_IN RGBCOL Float3ToCOLORREF(const float3& c)
{
	return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), 255);
}

CUDA_FUNC_IN float3 COLORREFToFloat3(RGBCOL c)
{
	return make_float3((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f);
}
#undef toInt

typedef uchar4 RGBE;

CUDA_FUNC_IN RGBE Float3ToRGBE(float3& c)
{
	float v = fmaxf(c);
	if(v < 1e-32)
		return make_uchar4(0,0,0,0);
	else
	{
		int e;
		v = frexp(v, &e) * 256.0f / v;
		return make_uchar4((unsigned char)(c.x * v), (unsigned char)(c.y * v), (unsigned char)(c.z * v), e + 128);
	}
}

CUDA_FUNC_IN float3 RGBEToFloat3(RGBE a)
{
	float f = ldexp(1.0f, a.w - (int)(128+8));
	return make_float3((float)a.x * f, (float)a.y * f, (float)a.z * f);
}

CUDA_FUNC_IN float Luminance(float3& c)
{
	return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
}

CUDA_FUNC_IN float3 RGBToXYZ(float3& c)
{
	float3 r;
	r.x = dot(make_float3(0.5767309,  0.1855540,  0.1881852), c);
	r.y = dot(make_float3(0.2973769,  0.6273491,  0.0752741), c);
	r.z = dot(make_float3(0.0270343,  0.0706872,  0.9911085), c);
	return r;
}

CUDA_FUNC_IN float3 XYZToRGB(float3& c)
{
	float3 r;
	r.x = dot(make_float3(2.0413690, -0.5649464, -0.3446944), c);
	r.y = dot(make_float3(-0.9692660,  1.8760108,  0.0415560), c);
	r.z = dot(make_float3(0.0134474, -0.1183897,  1.0154096), c);
	return r;
}

CUDA_FUNC_IN float3 XYZToYxy(float3& c)
{
	float s = c.x + c.y + c.z;
	return make_float3(c.y, c.x / s, c.y / s);
}

CUDA_FUNC_IN float3 YxyToXYZ(float3& c)
{
	float3 r;
	r.x = c.x * c.y / c.z;
	r.y = c.x;
	r.z = c.x * (1.0f - c.y - c.z) / c.z;
	return r;
}