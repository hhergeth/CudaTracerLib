#include <StdAfx.h>
#include "e_Core.h"
#include "../CudaMemoryManager.h"
#include "..\MathTypes.h"
#include "e_RoughTransmittance.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>
#include "e_FileTexture.h"
#include <crtdbg.h>
#include "../Kernel/k_Tracer.h"

CUDA_FUNC_IN AABB calculatePlaneAABBInCell(const AABB& cell, const Vec3f& p, const Vec3f& n, float r)
{
	Vec3f cellSize = cell.Size();
	AABB res = AABB::Identity();
	for (int a = 0; a < 3; a++)
	{
		if (n[a] != 0)
		{
			Vec3f d(0.0f);
			for (int l = 0; l < 4; l++)
			{
				Vec3f a1(0.0f); a1[(a + 1) % 3] = cellSize[(a + 1) % 3];
				Vec3f o = l == 0 ? cell.minV : (l == 2 ? cell.maxV : (l == 1 ? cell.minV + a1 : cell.maxV - a1));
				d[a] = l < 2 ? 1 : -1;
				float lambda = (p[a] - o[a]) / d[a];//plus or minus one!
				if (lambda > 0 && lambda < cellSize[a])
				{
					Vec3f x = o + lambda * d;
					float l2 = distance(p, x);
					x = p + (x - p).normalized() * min(l2, r);
					res = res.Extend(x);
				}
			}
		}
	}
	return res;
}
CUDA_FUNC_IN void calculatePlaneFrameInCell(const AABB& plane_box, Frame* frame, const Vec3f* p = 0, const Vec2f* xy = 0)
{
	Vec3f v_f = Vec3f(plane_box.maxV.x - plane_box.minV.x, 0, 0), 
		  v_g = Vec3f(0, plane_box.maxV.y - plane_box.minV.y, plane_box.maxV.z - plane_box.minV.z);
	if (frame)
		*frame = Frame(v_f, v_g, cross(v_f, v_g));
	if (p)
	{
		float d = dot(v_f, v_g);
		if (math::abs(d) > 1e-2f)
			printf("non ortho! f = [%f;%f;%f], g = [%f;%f;%f]\n", v_f.x, v_f.y, v_f.z, v_g.x, v_g.y, v_g.z);
		Vec2f xy = Vec2f(dot(v_f, *p - plane_box.minV), dot(v_g, *p - plane_box.minV));
		float err = dot(normalize(cross(v_f, v_g)), *p - plane_box.minV);
		Vec3f ap = *p - plane_box.minV;
		if (xy.x < 0 || xy.x > 1 || xy.y < 0 || xy.y > 1 || err < 0 || err > 0.1f)
			printf("xy={%f,%f}, err = %f\n", xy.x, xy.y, err);//f=[%f;%f;%f], g=[%f;%f;%f], p-a=[%f;%f;%f]) v_f.x, v_f.y, v_f.z, v_g.x, v_g.y, v_g.z, ap.x, ap.y, ap.z
	}
}

void InitializeCuda4Tracer(const std::string& dataPath)
{
	AABB box(Vec3f(0.0f), Vec3f(1.0f));
	Vec3f p(0.5f), n = Vec3f(1, 0, 0);
	//auto plane_box = calculatePlaneAABBInCell(box, p, n, FLT_MAX);
	Vec2f xy;
	//calculatePlaneFrameInCell(plane_box, 0, &p, &xy);

#ifndef NDEBUG
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetDbgFlag(_CrtSetDbgFlag(0) | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
	cudaError er = cudaFree(0);
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	e_RoughTransmittanceManager::StaticInitialize(dataPath);
}

void DeInitializeCuda4Tracer()
{
	k_TracerBase::g_sRngs.Free();
	FreeImage_DeInitialise();
	SpectrumHelper::StaticDeinitialize();
	e_RoughTransmittanceManager::StaticDeinitialize();
#ifndef NDEBUG
	//_CrtDumpMemoryLeaks();
#endif
}



std::map<void*, CudaMemoryEntry> CudaMemoryManager::alloced_entries;
std::vector<CudaMemoryEntry> CudaMemoryManager::freed_entries;

cudaError_t CudaMemoryManager::Cuda_malloc_managed(void** v, size_t i, const std::string& callig_func)
{
	cudaError_t r = cudaMalloc(v, i);
	if(r == CUDA_SUCCESS)
	{
		CudaMemoryEntry e;
		e.address = *v;
		e.free_func = "";
		e.length = i;
		e.malloc_func = callig_func;
		alloced_entries[*v] = e;
	}
	else ThrowCudaErrors(r);
	return r;
}

cudaError_t CudaMemoryManager::Cuda_free_managed(void* v, const std::string& callig_func)
{
	if(alloced_entries.count(v))
	{
		CudaMemoryEntry e = alloced_entries[v];
		//alloced_entries.erase(v);
		e.free_func = callig_func;
		freed_entries.push_back(e);
	}
	cudaError_t r = cudaFree(v);
	ThrowCudaErrors(r);
	return r;
}