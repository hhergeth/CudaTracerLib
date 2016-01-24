#include "TriIntersectorData.h"
#include <MathTypes.h>

namespace CudaTracerLib {

void TriIntersectorData::setData(const Vec3f& a, const Vec3f& b, const Vec3f& c)
{
	//if(absdot(normalize(cross(a - c, b - c)), make_float3(1, 0, 0)) < 0.05f)
	/*{
	float l = length((a + b + c) / 3.0f - a) / 200.0f;
	//e += make_float3(0,0.02f,-0.02f);
	//p *= 1.02f;
	//q *= 1.02f;
	p = p + make_float3(l, -l, l);
	q = q + make_float3(-l, l, -l);
	e = -p - q;
	e = e + make_float3(0,l,-l)/200.0f;
	}*/
	float4x4 m;
	m.col(0, Vec4f(a - c, 0));
	m.col(1, Vec4f(b - c, 0));
	m.col(2, Vec4f(cross(a - c, b - c), 0));
	m.col(3, Vec4f(c, 1));
	m = m.inverse();
	this->a = Vec4f(m(2, 0), m(2, 1), m(2, 2), -m(2, 3));
	this->b = m.row(0);
	this->c = m.row(1);
	Vec3f v1, v2, v3;
	getData(v1, v2, v3);
	//*(float2*)t2 = make_float2(m[0].x, m[0].y);
	//*(half2*)(((int*)t2) + 2) = half2(m[0].z, m[0].w);
	//this->b = make_float4(m[1].x, m[1].y, m[1].z, m[1].w);
	//if(this->a.x == -0.0f)
	//	this->a.x = 0.0f;
	//this->a = make_float4(m[2].y, m[2].z, -m[2].w, m[0].y);
	//this->b = make_float4(m[0].z, m[0].w, m[1].y, m[1].z);
	//t2->c = m[1].w;
	//t2->setXs(m[2].x, m[0].x, m[1].x);
}

void TriIntersectorData::getData(Vec3f& v0, Vec3f& v1, Vec3f& v2) const
{
	float4x4 m = float4x4::Identity();
	m.row(0, b);
	m.row(1, c);
	m.row(2, a);
	m(2, 3) *= -1.0f;
	m = m.inverse();
	Vec3f e02 = m.col(0).getXYZ(), e12 = m.col(1).getXYZ();
	v2 = m.col(3).getXYZ();
	v0 = v2 + e02;
	v1 = v2 + e12;
}

bool TriIntersectorData::Intersect(const Ray& r, float* dist, Vec2f* bary) const
{
	float Oz = a.w - r.ori().x*a.x - r.ori().y*a.y - r.ori().z*a.z;
	float invDz = 1.0f / (r.dir().x*a.x + r.dir().y*a.y + r.dir().z*a.z);
	float t = Oz * invDz;
	float tmax = dist ? *dist : FLT_MAX;
	if (t > 0.0001f && t < tmax)
	{
		float Ox = b.w + r.ori().x*b.x + r.ori().y*b.y + r.ori().z*b.z;
		float Dx = r.dir().x*b.x + r.dir().y*b.y + r.dir().z*b.z;
		float u = Ox + t*Dx;
		if (u >= 0.0f)
		{
			float Oy = c.w + r.ori().x*c.x + r.ori().y*c.y + r.ori().z*c.z;
			float Dy = r.dir().x*c.x + r.dir().y*c.y + r.dir().z*c.z;
			float v = Oy + t*Dy;
			if (v >= 0.0f && u + v <= 1.0f)
			{
				if (dist)
					*dist = t;
				if (bary)
					*bary = Vec2f(u, v);
				return true;
			}
		}
	}
	return false;
}

}