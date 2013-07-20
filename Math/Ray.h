#pragma once
#include "Vector.h"

struct e_TriangleData;
class e_Node;
struct e_KernelBSDF;
struct e_KernelMaterial;
struct e_KernelBSSRDF;
struct TraceResult
{
	float m_fDist;
	float2 m_fUV;
	const e_TriangleData* m_pTri;
	const e_Node* m_pNode;
	CUDA_FUNC_IN bool hasHit() const
	{
		return m_pTri != 0;
	}
	CUDA_FUNC_IN void Init()
	{
		m_fDist = FLT_MAX;
		m_fUV = make_float2(0,0);
		m_pNode = 0;
		m_pTri = 0;
	}
	CUDA_FUNC_IN operator bool() const
	{
		return hasHit();
	}

	CUDA_FUNC_IN Onb lerpOnb();
	CUDA_FUNC_IN unsigned int getMatIndex();
	CUDA_FUNC_IN float2 lerpUV();
	CUDA_FUNC_IN e_KernelBSDF GetBSDF(const e_KernelMaterial* a_Mats);
	CUDA_FUNC_IN void GetBSDF(const e_KernelMaterial* a_Mats, e_KernelBSDF* bsdf);
	CUDA_FUNC_IN bool GetBSSRDF(const e_KernelMaterial* a_Mats, e_KernelBSSRDF* bssrdf);
	//CUDA_FUNC_IN DifferentialGeometry lerpDG(const Ray& r);
	//CUDA_FUNC_IN DifferentialGeometry lerpDG(const RayDifferential& r);
};

class Ray
{
public:  //Data

  // basic information about a ray
  float3 origin;
  float3 direction;
  
public:  // Methods
	CUDA_FUNC Ray()
	{
	}
	// Set up the ray with a give origin and direction (and optionally a ray depth)
	CUDA_FUNC_IN Ray( const float3 &orig, const float3 &dir )
		: origin(orig), direction(dir)
	{
	}
	// Normalize ray direction
	CUDA_FUNC_IN void NormalizeRayDirection( void ) { direction = normalize(direction); }	

	CUDA_FUNC_IN float3 getOrigin() const{return origin;}
	CUDA_FUNC_IN float3 getDirection() const{return direction;}
	CUDA_FUNC_IN Ray operator *(const float4x4& m) const
	{
		return Ray(m * origin, m.TransformNormal(direction));
	}
	CUDA_FUNC_IN float3 operator()(float d) const
	{
		return origin + d * direction;
	}
};

class RayDifferential : public Ray
{
	float3 rxOrigin, ryOrigin;
    float3 rxDirection, ryDirection;

	CUDA_FUNC_IN RayDifferential()
	{
	}

	CUDA_FUNC_IN RayDifferential(const float3 &orig, const float3 &dir, const float3& rxo, const float3& ryo, const float3& rxd, const float3& ryd)
		: Ray(orig, dir)
	{
		rxOrigin = rxo;
		ryOrigin = ryo;
		rxDirection = rxd;
		ryDirection = ryd;
	}
};

class DifferentialGeometry
{
    float3 p;
    float3 nn;
    float u, v;
    float3 dpdu, dpdv;
    float3 dndu, dndv;
    float3 dpdx, dpdy;
    float dudx, dvdx, dudy, dvdy;

	CUDA_FUNC_IN DifferentialGeometry()
	{

	}
};

