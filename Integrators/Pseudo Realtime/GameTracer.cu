#include "GameTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <CudaMemoryManager.h>

namespace CudaTracerLib {

CUDA_DEVICE DeviceDepthImage g_DepthImageGT;

#define BLOCK_SIZE 16
__global__ void primaryKernelBlocked(int width, int height, Image g_Image, bool depthImage, Spectrum* lastImage1, Spectrum* lastImage2, Sensor lastSensor, int nIteration)
{
	CudaRNG rng = g_RNGData();
	int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x), y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
	if (x < width && y < height)
	{
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);

		Ray primaryRay;
		TraceResult primaryRes;
		Spectrum Le(0.0f);
		Spectrum primaryThrough[4];
		for (int i = 0; i < 4; i++)
		{
			int x2 = x + i % 2, y2 = y + i / 2;
			primaryThrough[i] = g_SceneData.m_Camera.sampleRay(primaryRay, Vec2f(x2, y2), rng.randomFloat2());
			primaryRes = Traceray(primaryRay);
			if (primaryRes.hasHit())
			{
				primaryRes.getBsdfSample(primaryRay, bRec, ETransportMode::ERadiance, &rng);
				Le = primaryRes.Le(dg.P, bRec.dg.sys, -primaryRay.direction);
				primaryThrough[i] *= primaryRes.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (depthImage && x2 < width && y2 < height)
					g_DepthImageGT.Store(x2, y2, primaryRes.m_fDist);
			}
			else primaryThrough[i] = g_SceneData.EvalEnvironment(primaryRay);
		}

		Spectrum through_indicrect(1.0f);
		Spectrum L_indirect(0.0f);
		if (primaryRes.hasHit())
		{
			int N_INDIRECT = 4;
			for (int i = 0; i < N_INDIRECT; i++)
			{
				Ray r_indicrect = Ray(dg.P, bRec.getOutgoing());
				TraceResult r2_indicrect = Traceray(r_indicrect);
				through_indicrect = Transmittance(r_indicrect, 0, r2_indicrect.m_fDist);
				if (r2_indicrect.hasHit())
				{
					DifferentialGeometry dg_indicrect;
					BSDFSamplingRecord bRec_indicrect(dg_indicrect);
					r2_indicrect.getBsdfSample(r_indicrect, bRec_indicrect, ETransportMode::ERadiance, &rng);
					L_indirect += UniformSampleOneLight(bRec_indicrect, r2_indicrect.getMat(), rng);
				}
				else
				{
					L_indirect += g_SceneData.EvalEnvironment(r_indicrect);
				}
			}
			L_indirect /= N_INDIRECT;

			/*if (nIteration > 2)
			{
			DirectSamplingRecord dRec(primaryRay(primaryRes.m_fDist), Vec3f(0.0f));
			lastSensor.sampleDirect(dRec, Vec2f(0, 0));
			if (dRec.pdf)
			{
			int oy = int(dRec.uv.y) / 2, ox = int(dRec.uv.x) / 2, w2 = width / 2;
			Spectrum lu = lastImage1[oy * w2 + ox], ru = lastImage1[oy * w2 + min(w2 - 1, ox + 1)],
			ld = lastImage1[min(height / 2 - 1, oy + 1) * w2 + ox], rd = lastImage1[min(height / 2 - 1, oy + 1) * w2 + min(w2 - 1, ox + 1)];
			Spectrum lL = math::bilerp(dRec.uv - dRec.uv.floor(), lu, ru, ld, rd);
			//Spectrum lL = lastImage1[oy * w2 + ox];
			//Spectrum lL = (lu + ru + ld + rd) / 4.0f;
			const float p = 0.5f;
			L_indirect = p * lL + (1 - p) * L_indirect;
			}
			}
			lastImage2[y / 2 * width / 2 + x / 2] = L_indirect;*/
		}
		else lastImage2[y / 2 * width / 2 + x / 2] = primaryThrough[0];

		/*CUDA_SHARED Spectrum accuData;
		accuData = Spectrum(0.0f);
		__syncthreads();
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			atomicAdd(&accuData[i], L_indirect[i]);
		__syncthreads();
		L_indirect = accuData / (blockDim.x * blockDim.y);

		CUDA_SHARED RGBCOL indirectData[BLOCK_SIZE*BLOCK_SIZE];
		indirectData[threadIdx.y * blockDim.x + threadIdx.x] = L_indirect.toRGBCOL();
		__syncthreads();
		int filterW = 2;
		GaussianFilter filt(filterW * 2, filterW * 2, 0.5f);
		int xstart = math::clamp((int)threadIdx.x - filterW / 2, filterW / 2 + 1, BLOCK_SIZE - filterW / 2 - 1),
			ystart = math::clamp((int)threadIdx.y - filterW / 2, filterW / 2 + 1, BLOCK_SIZE - filterW / 2 - 1);
		Spectrum filterVal(0.0f);
		float filterWeight = 0.0f;
		for (int i = -filterW / 2; i <= filterW / 2; i++)
			for (int j = -filterW / 2; j <= filterW / 2; j++)
			{
				float f = filt.Evaluate(i, j);
				Spectrum v;
				v.fromRGBCOL(indirectData[(ystart + j) * blockDim.x + xstart + i]);
				filterVal += v * f;
				filterWeight += f;
			}
		L_indirect = filterVal / filterWeight;*/

		for (int i = 0; i < 4; i++)
		{
			Spectrum L = primaryRes.hasHit() ? Transmittance(primaryRay, 0, primaryRes.m_fDist) * (Le + primaryThrough[i] * through_indicrect * L_indirect) : primaryThrough[i];
			int x2 = x + i % 2, y2 = y + i / 2;
			if (x2 < width && y2 < height)
				g_Image.AddSample(x2, y2, L);
		}
	}
	g_RNGData(rng);
}

void GameTracer::DoRender(Image* I)
{
	if (hasDepthBuffer())
		CopyToSymbol(g_DepthImageGT, getDeviceDepthBuffer());
	swapk(&m_pDeviceLastImage1, &m_pDeviceLastImage2);
	primaryKernelBlocked << <dim3(w / (2 * BLOCK_SIZE) + 1, h / (2 * BLOCK_SIZE) + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(w, h, *I, hasDepthBuffer(), m_pDeviceLastImage1, m_pDeviceLastImage2, lastSensor, iterations++);
	lastSensor = g_SceneData.m_Camera;
}

void GameTracer::Debug(Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	Ray r, rX, rY;
	g_SceneData.sampleSensorRay(r, rX, rY, Vec2f(pixel.x, pixel.y), rng.randomFloat2());
}

void GameTracer::Resize(unsigned int _w, unsigned int _h)
{
	if (m_pDeviceLastImage1)
		CUDA_FREE(m_pDeviceLastImage1);
	if (m_pDeviceLastImage2)
		CUDA_FREE(m_pDeviceLastImage2);
	CUDA_MALLOC(&m_pDeviceLastImage1, _w * _h * sizeof(Spectrum));
	CUDA_MALLOC(&m_pDeviceLastImage2, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastImage1, 0, _w * _h * sizeof(Spectrum));
	cudaMemset(m_pDeviceLastImage2, 0, _w * _h * sizeof(Spectrum));
	Platform::SetMemory(&lastSensor, sizeof(lastSensor));
	iterations = 0;
	Tracer<false, false>::Resize(_w, _h);
}

}