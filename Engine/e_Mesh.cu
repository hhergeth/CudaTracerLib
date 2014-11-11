#include "e_Mesh.h"
#include "..\Base\StringUtils.h"
#include "e_TraceResult.h"

void e_TriIntersectorData::setData(const float3& a, const float3& b, const float3& c)
{
	//if(AbsDot(normalize(cross(a - c, b - c)), make_float3(1, 0, 0)) < 0.05f)
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
	m.col(0, make_float4(a - c, 0));
	m.col(1, make_float4(b - c, 0));
	m.col(2, make_float4(cross(a - c, b - c), 0));
	m.col(3, make_float4(c, 1));
	m = m.inverse();
	this->a = make_float4(m(2, 0), m(2, 1), m(2, 2), -m(2, 3));
	this->b = m.row(0);
	this->c = m.row(1);
	float3 v1, v2, v3;
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

void e_TriIntersectorData::getData(float3& v0, float3& v1, float3& v2) const
{
	float4x4 m = float4x4::Identity();
	m.row(0, b);
	m.row(1, c);
	m.row(2, a);
	m(2, 3) *= -1.0f;
	m = m.inverse();
	float3 e02 = !m.col(0), e12 = !m.col(1);
	v2 = !m.col(3);
	v0 = v2 + e02;
	v1 = v2 + e12;
}

bool e_TriIntersectorData::Intersect(const Ray& r, TraceResult* a_Result) const
{
	float Oz = a.w - r.origin.x*a.x - r.origin.y*a.y - r.origin.z*a.z;
	float invDz = 1.0f / (r.direction.x*a.x + r.direction.y*a.y + r.direction.z*a.z);
	float t = Oz * invDz;
	if (t > 0.0001f && t < a_Result->m_fDist)
	{
		float Ox = b.w + r.origin.x*b.x + r.origin.y*b.y + r.origin.z*b.z;
		float Dx = r.direction.x*b.x + r.direction.y*b.y + r.direction.z*b.z;
		float u = Ox + t*Dx;
		if (u >= 0.0f)
		{
			float Oy = c.w + r.origin.x*c.x + r.origin.y*c.y + r.origin.z*c.z;
			float Dy = r.direction.x*c.x + r.direction.y*c.y + r.direction.z*c.z;
			float v = Oy + t*Dy;
			if (v >= 0.0f && u + v <= 1.0f)
			{
				a_Result->m_fDist = t;
				a_Result->m_fUV = make_float2(u, v);
				return true;
			}
		}
	}
	return false;
}