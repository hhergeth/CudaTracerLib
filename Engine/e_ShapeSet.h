#pragma once

#include <MathTypes.h>
#include "..\Engine\e_Mesh.h"
#include "..\Math\Sampling.h"
#include "e_Samples.h"

#define MAX_SHAPE_LENGTH 32

struct CUDA_ALIGN(16) ShapeSet
{
	CUDA_ALIGN(16) struct triData
	{
		CUDA_ALIGN(16) e_TriIntersectorData dat;
		CUDA_ALIGN(16) float3 p[3];
		CUDA_ALIGN(16) float3 n;
		CUDA_ALIGN(16) float area;
		CUDA_ALIGN(16) const e_TriIntersectorData* datRef;

		CUDA_FUNC_IN void UniformSampleTriangle(float u1, float u2, float *u, float *v) const
		{
			float su1 = sqrtf(u1);
			*u = 1.f - su1;
			*v = u2 * su1;
		}
		
		triData(){}
		triData(const e_TriIntersectorData* a_Int, float4x4& mat)
		{
			datRef = a_Int;
			Recalculate(mat);
		}
		CUDA_FUNC_IN float3 rndPoint(float b1, float b2) const
		{
			return b1 * p[0] + b2 * p[1] + (1.f - b1 - b2) * p[2];
		}
		CUDA_FUNC_IN float3 Sample(float u1, float u2, float3* Ns) const
		{
			float b1, b2;
			UniformSampleTriangle(u1, u2, &b1, &b2);
			*Ns = nor();
			return rndPoint(b1, b2);
		}
		CUDA_FUNC_IN float3 nor() const
		{
			return n;
		}
		AABB box() const;
		void Recalculate(float4x4& mat);
	};
public:
    ShapeSet(e_TriIntersectorData** indices, unsigned int indexCount, float4x4& mat)
	{
		if(indexCount > MAX_SHAPE_LENGTH)
			throw 1;
		count = indexCount;
		float areas[MAX_SHAPE_LENGTH];
		sumArea = 0;
		for(int i = 0; i < count; i++)
		{
			tris[i] = triData(indices[i], mat);
			areas[i] = tris[i].area;
			sumArea += tris[i].area;
		}
		areaDistribution = Distribution1D<MAX_SHAPE_LENGTH>(areas, count);
	}
    CUDA_FUNC_IN float Area() const { return sumArea; }
    CUDA_FUNC_IN float3 Sample(const LightSample &ls, float3 *Ns, const e_TriIntersectorData* a_Int) const
	{
		const triData& sn = tris[areaDistribution.SampleDiscrete(ls.uComponent, NULL)];
		return sn.Sample(ls.uPos[0], ls.uPos[1], Ns);
	}
    CUDA_FUNC_IN float3 Sample(const float3 &p, const LightSample &ls, float3 *Ns, const e_TriIntersectorData* a_Int) const
	{
		int sn = areaDistribution.SampleDiscrete(ls.uComponent, NULL);
		float3 pt = tris[sn].Sample(ls.uPos[0], ls.uPos[1], Ns);
		TraceResult r2;
		r2.Init();
		float3 n;
		Ray r(p, normalize(pt - p));
		for (int i = 0; i < count; ++i)
			if(tris[i].dat.Intersect(r, &r2))//a_Int[tris[i].index]
				n = tris[i].n;
		if(r2.hasHit() && Ns)
			*Ns = n;
		return r(r2.m_fDist);
		return pt;
	}
    CUDA_FUNC_IN float Pdf(const float3 &p, const float3 &wi, const e_TriIntersectorData* a_Int) const
	{
		float pdf = 0.f;
		TraceResult r;
		for (int i = 0; i < count; i++)
		{
			r.Init();
			if(tris[i].dat.Intersect(Ray(p, wi), &r))
			{
				float pdf2 = DistanceSquared(p, p + wi * r.m_fDist) / (AbsDot(tris[i].nor(), -1.0f * wi) * tris[i].area);//we trust in the compiler
				if (pdf2 == pdf2)//NAN check
					pdf += tris[i].area * pdf2;
			}
		}
		return pdf / sumArea;
	}
    CUDA_FUNC_IN float Pdf(const float3 &p) const
	{
		return float(count) / sumArea;
	}
	AABB getBox() const
	{
		AABB b = AABB::Identity();
		for(int i = 0; i < count; i++)
			b.Enlarge(tris[i].box());
		return b;
	}
	void Recalculate(float4x4& mat)
	{
		float areas[MAX_SHAPE_LENGTH];
		sumArea = 0;
		for(int i = 0; i < count; i++)
		{
			tris[i].Recalculate(mat);
			areas[i] = tris[i].area;
			sumArea += tris[i].area;
		}
		areaDistribution = Distribution1D<MAX_SHAPE_LENGTH>(areas, count);
	}
private:
    CUDA_ALIGN(16) triData tris[MAX_SHAPE_LENGTH];
    float sumArea;
    CUDA_ALIGN(16) Distribution1D<MAX_SHAPE_LENGTH> areaDistribution;
	int count;
};