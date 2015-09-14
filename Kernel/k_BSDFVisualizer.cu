#include "../Engine/e_Buffer.h"
#include "k_BSDFVisualizer.h"
#include "../Kernel/k_TraceHelper.h"
#include <time.h>
#include "../Kernel/k_TraceAlgorithms.h"
#include "../Engine/e_BSDF.h"
#include "../Engine/e_Light.h"
#include "../Engine/e_FileTexture.h"
#include "../Base/FileStream.h"

CUDA_FUNC_IN Vec3f hemishphere(const Vec2f& q)
{
	Vec2f f = q * 2 - Vec2f(1);
	return normalize(Vec3f(f, math::sqrt(1 - f.x * f.x - f.y * f.y)));
}

CUDA_FUNC_IN Spectrum func(BSDFALL& bsdf, Vec3f wo, int w, int h, int x, int y, bool cosTheta)
{
	DifferentialGeometry dg;
	dg.bary = Vec2f(0.5f);
	dg.hasUVPartials = false;
	dg.n = Vec3f(0, 0, 1);
	dg.P = Vec3f(0);
	dg.uv[0] = Vec2f(0);
	dg.sys = Frame(dg.n);
	BSDFSamplingRecord bRec(dg);
	bRec.eta = 1;
	bRec.mode = ERadiance;
	bRec.typeMask = EAll;
	bRec.wi = wo;
	Vec2f xy = Vec2f(x, y) / Vec2f(w, h);
	if (length(2.0f * xy - Vec2f(1)) > 1)
		return Spectrum(100, 149, 237) / 255.0f;
	bRec.wo = hemishphere(xy);//bsdf.As()->hasComponent(EDelta) ? EDiscrete : ESolidAngle
	Spectrum f = bsdf.f(bRec, EDiscrete) + bsdf.f(bRec, ESolidAngle);
	return f / (cosTheta ? 1 : Frame::cosTheta(bRec.wo));
}

CUDA_FUNC_IN Spectrum func2(BSDFALL& bsdf, e_InfiniteLight& light, CudaRNG& rng, float3 wo, int w, int h, int x, int y, bool cosTheta)
{
	DifferentialGeometry dg;
	dg.bary = Vec2f(0.5f);
	dg.hasUVPartials = false;
	dg.n = Vec3f(0, 0, 1);
	dg.P = Vec3f(0);
	dg.uv[0] = Vec2f(0);
	dg.sys = Frame(dg.n);
	BSDFSamplingRecord bRec(dg);
	bRec.eta = 1;
	bRec.mode = ERadiance;
	bRec.typeMask = EAll;
	Vec2f xy = Vec2f(x, y) / Vec2f(w, h);
	//if (length(2.0f * xy - Vec2f(1)) > 1)
	//	return Spectrum(100, 149, 237) / 255.0f;
	//bRec.wi = hemishphere(xy);
	Vec3f tar(xy.x, 0, xy.y);
	Vec3f cam(0, 1, 0);
	bRec.wi = normalize(tar - cam);
	Spectrum q(0.0f);
	int N = 20;
	for (int i = 0; i < N; i++)
	{
		Spectrum f = bsdf.sample(bRec, rng.randomFloat2());
		q += f * light.evalEnvironment(Ray(Vec3f(0), bRec.wo));
	}
	return q / float(N);
}

CUDA_DEVICE BSDFALL g_BSDF;
CUDA_DEVICE CUDA_ALIGN(256) char g_Light[sizeof(e_InfiniteLight)];
CUDA_GLOBAL void BSDFCalc(Vec3f wo, e_Image I, Vec2i off, Vec2i size, float scale, bool cosTheta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < size.y)
	{
		Spectrum f = func(g_BSDF, wo, size.x, size.y, x, y, cosTheta) * scale;
		I.SetSample(x + off.x, y + off.y, f.toRGBCOL());
	}
}

CUDA_GLOBAL void BSDFCalc2(Vec3f wo, e_Image I, Vec2i off, Vec2i size, float scale, bool cosTheta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < size.y)
	{
		CudaRNG rng = g_RNGData();
		Spectrum f = func2(g_BSDF, *(e_InfiniteLight*)g_Light, rng, wo, size.x, size.y, x, y, cosTheta) * scale;
		I.AddSample(x + off.x, y + off.y, f);
		g_RNGData(rng);
	}
}

void k_BSDFVisualizer::DrawRegion(e_Image* I, const Vec2i& off, const Vec2i& size)
{
	if (!m_Bsdf)
		return;
	cudaMemcpyToSymbol(g_RNGDataDevice, &k_Tracer::g_sRngs, sizeof(CudaRNGBuffer));
	int p = 16;
	cudaMemcpyToSymbol(g_BSDF, m_Bsdf, sizeof(BSDFALL));
	if (drawEnvMap)
	{
		cudaMemcpyToSymbol(g_Light, m_pLight, sizeof(e_InfiniteLight));
		BSDFCalc2 << <dim3(size.x / p + 1, size.y / p + 1, 1), dim3(p, p, 1) >> >(m_wo, *I, off, size, LScale, cosTheta);
		I->DoUpdateDisplay(0);
	}
	else
	{
		I->Clear();
		BSDFCalc << <dim3(size.x / p + 1, size.y / p + 1, 1), dim3(p, p, 1) >> >(m_wo, *I, off, size, LScale, cosTheta);
	}
}

void k_BSDFVisualizer::DoRender(e_Image* I)
{
	DrawRegion(I, Vec2i(0, 0), Vec2i(w, h));
}

void k_BSDFVisualizer::Debug(e_Image* I, const Vec2i& p)
{
	g_RNGDataHost = k_Tracer::g_sRngs;
	CudaRNG rng = g_RNGData();
	if (drawEnvMap)
		func2(*m_Bsdf, *m_pLight, rng, m_wo, w, h, p.x, p.y, cosTheta);
	func(*m_Bsdf, m_wo, w, h, p.x, p.y, cosTheta);
}

void k_BSDFVisualizer::setSkydome(const char* compiledPath)
{
	if (m_pBuffer)
		m_pBuffer->Free();
	if (m_pBuffer2)
		m_pBuffer2->Free();
	m_pBuffer = new e_Stream<char>(1024 * 1024 * 32);
	m_pBuffer2 = new e_Buffer<e_MIPMap, e_KernelMIPMap>(3);
	FileInputStream in(compiledPath);
	e_BufferReference<e_MIPMap, e_KernelMIPMap> mip = m_pBuffer2->malloc(1); 
	m_pMipMap = new (mip.operator->())e_MIPMap(in.getFilePath(), in);
	in.Close();	
	m_pLight = new e_InfiniteLight(m_pBuffer, mip, Spectrum(1.0f), new AABB(Vec3f(0), Vec3f(1)));
}

k_BSDFVisualizer::~k_BSDFVisualizer()
{
	if (m_pBuffer)
		m_pBuffer->Free();
	if (m_pBuffer2)
		m_pBuffer2->Free();
	if (m_pLight)
		delete m_pLight;
}