#include "e_Mesh.h"

void e_TriIntersectorData::setData(float3& a, float3& b, float3& c)
{
	float3 p = a - c, q = b - c, e = -p - q;
	float3 d = cross(a - c, b - c);
	if(AbsDot(normalize(cross(a - c, b - c)), make_float3(1, 0, 0)) < 0.2f)
	{
		e += make_float3(0,0.2f,-0.2f);
		p *= 1.05f;
		q *= 1.05f;
	}
	//e = d - make_float3(1,0,0);
	float4x4 m( p.x, q.x, e.x + 1, c.x,
				p.y, q.y, e.y + 0, c.y,
				p.z, q.z, e.z + 0, c.z,
					0,   0,   0,   1  );
	m = m.Inverse();
	this->a = make_float4(m[2].x, m[2].y, m[2].z, -m[2].w);
	this->b = make_float4(m[0].x, m[0].y, m[0].z, m[0].w);
	this->c = make_float4(m[1].x, m[1].y, m[1].z, m[1].w);
	if(this->a.x == -0.0f)
		this->a.x = 0.0f;
}

void e_TriIntersectorData::getData(float3& v0, float3& v1, float3& v2) const
{
	float4x4 m(b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, a.x, a.y, a.z, -a.w, 0, 0, 0, 1);
	m = m.Inverse();
	float3 e02 = make_float3(m.X.x, m.Y.x, m.Z.x), e12 = make_float3(m.X.y, m.Y.y, m.Z.y);
	v2 = make_float3(m.X.w, m.Y.w, m.Z.w);
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