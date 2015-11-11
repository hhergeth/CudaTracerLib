#pragma once

#include "k_TraceHelper.h"
#include "k_BlockSampler_device.h"
#include <Engine/e_Material.h>

namespace CudaTracerLib {

class e_DynamicScene;
class k_BlockSampler;

typedef void(*SliderCreateCallback)(float, float, bool, float*, std::string);

CUDA_FUNC_IN float CalcZBufferDepth(float n, float f, float z)
{
	return (f / (f - n) * z - f * n / (f - n)) / z;
}

class k_TracerBase
{
public:
	static CudaRNGBuffer g_sRngs;
	static AABB GetEyeHitPointBox(e_DynamicScene* s, bool recursive);
	static float GetLightVisibility(e_DynamicScene* s, int recursion_depth);
	static TraceResult TraceSingleRay(Ray r, e_DynamicScene* s);
	static void InitRngs(unsigned int N = 1 << 16);
	static void RenderDepth(e_Image* img, e_DynamicScene* s);

	CUDA_DEVICE static Vec2i getPixelPos(unsigned int xoff, unsigned int yoff)
	{
#ifdef ISCUDA
		unsigned int x = xoff + blockIdx.x * threadsPerBlock.x + threadIdx.x;
		unsigned int y = yoff + blockIdx.y * threadsPerBlock.y + threadIdx.y;
		return Vec2i(x, y);
#else
		return Vec2i(xoff, yoff);
#endif
	}

	k_TracerBase();
	virtual ~k_TracerBase();
	virtual void InitializeScene(e_DynamicScene* a_Scene)
	{
		m_pScene = a_Scene;
	}
	virtual void Resize(unsigned int _w, unsigned int _h) = 0;
	virtual void DoPass(e_Image* I, bool a_NewTrace) = 0;
	virtual void Debug(e_Image* I, const Vec2i& pixel)
	{

	}
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{

	}
	virtual void CreateSliders(SliderCreateCallback a_Callback) const
	{

	}
	virtual bool isMultiPass() const = 0;
	virtual bool usesBlockSampler() const = 0;
	virtual unsigned int getNumPassesDone() const
	{
		return m_uPassesDone;
	}
	virtual unsigned int getRaysInLastPass() const
	{
		return m_uLastNumRaysTraced;
	}
	virtual float getLastTimeSpentRenderingSec() const
	{
		return m_fLastRuntime;
	}
	virtual unsigned int getAccRays() const
	{
		return m_uAccNumRaysTraced;
	}
	virtual float getAccTimeSpentRenderingSec() const
	{
		return m_fAccRuntime;
	}
	virtual IBlockSampler* getBlockSampler() const
	{
		return m_pBlockSampler;
	}
	k_BlockSampleImage getDeviceBlockSampler() const;
protected:
	float m_fLastRuntime;
	unsigned int m_uLastNumRaysTraced;
	float m_fAccRuntime;
	unsigned int m_uAccNumRaysTraced;
	unsigned int m_uPassesDone;
	unsigned int w, h;
	e_DynamicScene* m_pScene;
	cudaEvent_t start, stop;
	IBlockSampler* m_pBlockSampler;
	void allocateBlockSampler(e_Image* I);
};

template<bool USE_BLOCKSAMPLER, bool PROGRESSIVE> class k_Tracer : public k_TracerBase
{
public:
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		w = _w;
		h = _h;
		if (USE_BLOCKSAMPLER)
		{
			if (m_pBlockSampler)
			{
				m_pBlockSampler->Free();
				delete m_pBlockSampler;
			}
			m_pBlockSampler = 0;
		}
	}
	virtual void DoPass(e_Image* I, bool a_NewTrace)
	{
		if (USE_BLOCKSAMPLER && !m_pBlockSampler)
			allocateBlockSampler(I);
		g_sRngs.NextPass();
		ThrowCudaErrors(cudaEventRecord(start, 0));
		if (a_NewTrace || !PROGRESSIVE)
		{
			m_uPassesDone = 0;
			m_uAccNumRaysTraced = 0;
			m_fAccRuntime = 0;
			I->Clear();
			if (USE_BLOCKSAMPLER)
				m_pBlockSampler->Clear();
			StartNewTrace(I);
		}
		k_INITIALIZE(m_pScene, g_sRngs);
		k_setNumRaysTraced(0);
		m_uPassesDone++;
		DoRender(I);
		I->DoUpdateDisplay(getSplatScale());
		if (USE_BLOCKSAMPLER)
			m_pBlockSampler->AddPass();
		ThrowCudaErrors(cudaEventRecord(stop, 0));
		ThrowCudaErrors(cudaEventSynchronize(stop));
		if (start != stop)
			ThrowCudaErrors(cudaEventElapsedTime(&m_fLastRuntime, start, stop));
		else m_fLastRuntime = 0;
		m_fLastRuntime /= 1000.0f;
		m_uLastNumRaysTraced = k_getNumRaysTraced();
		m_fAccRuntime += m_fLastRuntime;
		m_uAccNumRaysTraced += m_uLastNumRaysTraced;
	}
	virtual bool isMultiPass() const
	{
		return PROGRESSIVE;
	}
	virtual bool usesBlockSampler() const
	{
		return USE_BLOCKSAMPLER;
	}

protected:
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
	{

	}
	virtual void DoRender(e_Image* I)
	{
		/*
		xxxxxxxx
		x      x	Warp := 8x4
		x      x
		xxxxxxxx

		WW
		WW	Block := 16x8

		BBBB
		BBBB
		BBBB
		BBBB	Block
		BBBB
		BBBB
		BBBB
		BBBB

		*/
		if (USE_BLOCKSAMPLER)
		{
			unsigned int nBlocks = m_pBlockSampler->NumBlocks();
			for (unsigned int idx = 0; idx < nBlocks; idx++)
			{
				unsigned int x, y, bw, bh;
				m_pBlockSampler->getBlockCoords(idx, x, y, bw, bh);
				RenderBlock(I, x, y, bw, bh);
			}
		}
		else
		{
			int nx = (I->getWidth() + blockSize - 1) / blockSize, ny = (I->getHeight() + blockSize - 1) / blockSize;
			for (int ix = 0; ix < nx; ix++)
				for (int iy = 0; iy < ny; iy++)
				{
					int x = ix * blockSize, y = iy * blockSize;
					int x2 = (ix + 1) * blockSize, y2 = (iy + 1) * blockSize;
					int bw = min(int(w), x2) - x, bh = min(int(h), y2) - y;
					RenderBlock(I, x, y, bw, bh);
				}
		}
	}
	virtual void StartNewTrace(e_Image* I)
	{

	}
	virtual float getSplatScale()
	{
		if (PROGRESSIVE)
			return 1.0f / float(m_uPassesDone);
		else return 0;
	}
};

}