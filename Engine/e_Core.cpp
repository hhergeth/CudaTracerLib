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
#include <ctime>

void testPrisma()
{
	float x, y, Y;
	CudaRNG rng;
	rng.Initialize(12434363, 6242574, (int)time(0));

	Spectrum s(rng.randomFloat(), rng.randomFloat(), rng.randomFloat());
	s.toYxy(Y, x, y);
	s *= 10000;
	Spectrum sum(0.0f);
	float sumw = 0;
	const int N = 1000;
	for (int i = 0; i < N; i++)
	{
		float w;
		sum += s.SampleSpectrum(w, rng.randomFloat()) / float(N);
		sumw += w / float(N);
	}
	Spectrum diff = sum - s;
	Spectrum div = sum / s;
	float d = diff.abs().max();
}

#include "e_Grid.h"
CUDA_FUNC_IN float SH(float f)
{
	return f >= 0 ? 1 : 0;
}
int test(CudaRNG& rng, const k_HashGrid_Reg& grid, const AABB& box, Ray& r, float& rad, float& ttEnd)
{
	r.origin = Vec3f(88.116699, -67.444046, 99.186798);
	r.direction = Vec3f(-0.511254, 0.556284, -0.655109);
	float m_fGridSize = grid.m_fGridSize;
	float m_fCurrentRadiusVol = 1;
	float tEnd = math::lerp(0.5f, 0.9f, rng.randomFloat()) * box.Size().length();
	ttEnd = tEnd;
	tEnd = 270.619f;

	Vec3f m_vCellSize = box.Size() / (m_fGridSize - 1);
	Vec3i Step(sign<int>(r.direction.x), sign<int>(r.direction.y), sign<int>(r.direction.z));
	Vec3f inv_d = r.direction;
	const float ooeps = math::exp2(-40.0f);//80 is too small, will create underflow on GPU
	inv_d.x = 1.0f / (math::abs(inv_d.x) > ooeps ? inv_d.x : copysignf(ooeps, inv_d.x));
	inv_d.y = 1.0f / (math::abs(inv_d.y) > ooeps ? inv_d.y : copysignf(ooeps, inv_d.y));
	inv_d.z = 1.0f / (math::abs(inv_d.z) > ooeps ? inv_d.z : copysignf(ooeps, inv_d.z));
	Vec3f DeltaT = abs(m_vCellSize * inv_d);

	Vec3f NextCrossingT[5];
	Vec3u Pos[5];
	float rayT[5];
	float maxT[5];
	//coordinate system which has left axis pointing towards (-1, 0, 0) and up to (0, 1, 0)
	Frame T(Vec3f(-math::abs(r.direction.z), 0, math::sign(r.direction.z) * r.direction.x),
		Vec3f(-r.direction.x * r.direction.y, math::sqr(r.direction.x) + math::sqr(r.direction.z), -r.direction.y * r.direction.z),
		r.direction);

	int nRaysTerminated = 0;
	float r_ = rng.randomFloat() * m_vCellSize.length() * 0.9f;
	rad = r_;
	r_ = 2.71542f;
	for (int i = 0; i < 5; i++)
	{
		Vec3f pos = i == 0 ? r.origin : (r.origin + T.toWorld(Vec3f(-r_ + ((i - 1) / 2) * 2 * r_, -r_ + ((i - 1) % 2) * 2 * r_, 0)));
		if (!box.Intersect(Ray(pos, r.direction), rayT + i, maxT + i))
		{
			rayT[i] = -1;
			nRaysTerminated++;
			continue;
		}
		rayT[i] = math::clamp(rayT[i], 0.0f, tEnd);
		maxT[i] = math::clamp(maxT[i], 0.0f, tEnd);
		Vec3f q = (r.direction * rayT[i] + pos - box.minV) / box.Size() * (m_fGridSize - 1);
		Pos[i] = clamp(Vec3u(unsigned int(q.x), unsigned int(q.y), unsigned int(q.z)), Vec3u(0), Vec3u(m_fGridSize - 1));
		auto A = box.minV + (Vec3f(Pos[i].x, Pos[i].y, Pos[i].z) + Vec3f(SH(r.direction.x), SH(r.direction.y), SH(r.direction.z))) * m_vCellSize,
			B = pos - r.direction * rayT[i];
		NextCrossingT[i] = max(Vec3f(0.0f), Vec3f(rayT[i]) + (A - B) * inv_d);
	}
	int cv = 0;
	while (nRaysTerminated != 5)
	{
		cv++;
		/*Vec3u minG(UINT_MAX), maxG(0);
		for (int i = 0; i < 5; i++)
		if (rayT[i] >= 0)
		{
		minG = min(minG, Pos[i]);
		maxG = max(maxG, Pos[i]);
		}
		for (unsigned int a = minG.x; a <= maxG.x; a++)
		for (unsigned int b = minG.y; b <= maxG.y; b++)
		for (unsigned int c = minG.z; c <= maxG.z; c++)
		{
		m_sStorage.store(Vec3u(a, b, c), beam_idx);
		}*/
		//if (rayT[0] > 0)
			//m_sStorage.store(Pos[0], beam_idx);
		for (int i = 0; i < 5; i++)
		{
			if (rayT[i] < 0)
				continue;
			int bits = ((NextCrossingT[i][0] < NextCrossingT[i][1]) << 2) + ((NextCrossingT[i][0] < NextCrossingT[i][2]) << 1) + ((NextCrossingT[i][1] < NextCrossingT[i][2]));
			int stepAxis = (0x00000a66 >> (2 * bits)) & 3;
			Pos[i][stepAxis] += Step[stepAxis];
			if (Pos[i][stepAxis] >= m_fGridSize || rayT[i] > maxT[i])
			{
				nRaysTerminated++;
				rayT[i] = -1;
				continue;
			}
			rayT[i] = NextCrossingT[i][stepAxis];
			NextCrossingT[i][stepAxis] += DeltaT[stepAxis];
		}
	}
	return cv;
}
void testGrid()
{
	CudaRNG rng;
	rng.Initialize(123, 456, 789);
	k_HashGrid_Reg grid(AABB(Vec3f(-100), Vec3f(100)), 0, 100 * 100 * 100);
	//Ray r(Vec3f(0), Vec3f(0, 0, 1));
	AABB box = grid.getAABB();
	std::cout << "Traversal test started : \n";
	int maxq = 0;
	for (int i = 0; i < 1000000; i++)
	{
	Ray r(math::lerp(box.minV, box.maxV, rng.randomFloat3()), Warp::squareToUniformSphere(rng.randomFloat2()));
	float rad, tend;
	int q = test(rng, grid, box, r, rad, tend);
	if (q > 250)
	{
	std::cout << "rad = " << rad << ", tend = " << tend << "\n";
	std::cout << format("r.origin = Vec3f(%f, %f, %f )", r.origin.x, r.origin.y, r.origin.z);
	std::cout << format("r.direction = Vec3f(%f, %f, %f )", r.direction.x, r.direction.y, r.direction.z);
	}
	maxq = max(q, maxq);
	std::cout << " : ";
	}
	std::cout << "\nTraversal test passed!\nQ = " << maxq << "\n";
}

void InitializeCuda4Tracer(const std::string& dataPath)
{
#ifndef NDEBUG
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetDbgFlag(_CrtSetDbgFlag(0) | _CRTDBG_CHECK_ALWAYS_DF);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
	ThrowCudaErrors(cudaFree(0));
	SpectrumHelper::StaticInitialize();
	FreeImage_Initialise();
	e_RoughTransmittanceManager::StaticInitialize(dataPath);

	//testPrisma();
	//testGrid();
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



