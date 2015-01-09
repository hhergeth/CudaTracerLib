#include "k_BDPT.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_FUNC_IN float PdfWtoA(const float aPdfW, const float aDist2, const float aCosThere)
{
	return aPdfW * std::abs(aCosThere) / aDist2;
}

CUDA_FUNC_IN float PdfAtoW(const float aPdfA, const float aDist2, const float aCosThere)
{
	return aPdfA * aDist2 / std::abs(aCosThere);
}

CUDA_FUNC_IN float revPdf(const e_KernelMaterial& mat, BSDFSamplingRecord& bRec)
{
	swapk(bRec.wo, bRec.wi);
	float pdf = mat.bsdf.pdf(bRec);
	swapk(bRec.wo, bRec.wi);
	return pdf;
}

struct BPTSubPathState
{
	Ray r;
	Spectrum throughput;
	bool delta;

	float dVCM;
	float dVC;
};

struct BPTVertex
{
	const e_KernelMaterial* mat;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec;
	Spectrum throughput;

	float dVCM;
	float dVC;

	CUDA_FUNC_IN BPTVertex()
		: bRec(dg)
	{

	}
};

CUDA_FUNC_IN void sampleEmitter(BPTSubPathState& v, CudaRNG& rng)
{
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;
	Spectrum Le = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2());
	const e_KernelLight* l = (const e_KernelLight*)pRec.object;
	Le *= l->sampleDirection(dRec, pRec, rng.randomFloat2());

	v.delta = 0;
	v.throughput = Le;
	v.r = Ray(pRec.p, dRec.d);

	DirectSamplingRecord directRec;
	directRec.d = directRec.refN = make_float3(1, 0, 0);
	directRec.n = make_float3(-1, 0, 0);
	directRec.measure = EArea;

	float directPdfW = l->pdfDirect(directRec);
	float emissionPdfW = pRec.pdf * dRec.pdf;

	v.dVCM = directPdfW / emissionPdfW;
	if (l->As()->IsDegenerate())
	{
		float usedCosLight = dot(pRec.n, -dRec.d);
		v.dVC = usedCosLight / emissionPdfW;
	}
	else v.dVC = 0;
}

CUDA_FUNC_IN void sampleCamera(BPTSubPathState& v, CudaRNG& rng, const float2& pixelPosition)
{
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;

	Spectrum imp = g_SceneData.m_Camera.samplePosition(pRec, rng.randomFloat2(), &pixelPosition);
	imp *= g_SceneData.m_Camera.sampleDirection(dRec, pRec, rng.randomFloat2(), &pixelPosition);

	v.delta = 1;
	v.throughput = imp;
	v.r = Ray(pRec.p, dRec.d);
	v.dVCM = 1.0f / dRec.pdf;
	v.dVC = 0;
}

CUDA_FUNC_IN bool sampleScattering(BPTSubPathState& v, BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, CudaRNG& rng)
{
	float bsdfDirPdfW, cosThetaOut;
	Spectrum f = mat.bsdf.sample(bRec, bsdfDirPdfW, rng.randomFloat2());
	cosThetaOut = Frame::cosTheta(bRec.wo);
	if (f.isZero())
		return false;
	float bsdfRevPdfW = bsdfDirPdfW;
	if ((bRec.sampledType & EDelta1D) == 0)
	{
		bsdfRevPdfW = revPdf(mat, bRec);

		//the pdf's are actually already multiplied with cosThetaOut
		bsdfDirPdfW /= cosThetaOut;
		bsdfRevPdfW /= Frame::cosTheta(bRec.wi);
	}


	if (bRec.sampledType & EDelta1D)
	{
		v.dVCM = 0;
		v.delta &= 1;
	}
	else
	{
		v.dVC = cosThetaOut / bsdfDirPdfW * (v.dVC * bsdfRevPdfW + v.dVCM);
		v.dVCM = 1.0f / (bsdfDirPdfW);
		v.delta &= 0;
	}

	v.r = Ray(bRec.dg.P, bRec.getOutgoing());
	v.throughput *= f;
	return true;
}

CUDA_FUNC_IN Spectrum gatherLight(const BPTSubPathState& cameraState, BSDFSamplingRecord& bRec, const TraceResult& r2, CudaRNG& rng, int subPathLength)
{
	e_KernelLight* l = &g_SceneData.m_sLightData[r2.LightIndex()];
	float pdfLight = g_SceneData.pdfLight(l);
	PositionSamplingRecord pRec(bRec.dg.P, bRec.dg.sys.n, 0);
	float directPdfA = l->pdfPosition(pRec);
	DirectionSamplingRecord dRec(-cameraState.r.direction);
	float emissionPdfW = l->pdfDirection(dRec, pRec) * directPdfA;
	Spectrum L = l->eval(bRec.dg.P, bRec.dg.sys, -cameraState.r.direction);

	if (L.isZero())
		return Spectrum(0.0f);

	if (subPathLength == 1)
		return L;

	directPdfA *= pdfLight;
	emissionPdfW *= pdfLight;

	float wCamera = directPdfA * cameraState.dVCM + emissionPdfW * cameraState.dVC;

	const float misWeight = 1.f / (1.f + wCamera);

	return misWeight * L;
}

CUDA_FUNC_IN void connectToCamera(const BPTSubPathState& lightState, BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, e_Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	Spectrum directFactor = g_SceneData.m_Camera.sampleDirect(dRec, rng.randomFloat2());
	bRec.wo = bRec.dg.toLocal(dRec.d);
	Spectrum bsdfFactor = mat.bsdf.f(bRec);
	float bsdfRevPdfW = mat.bsdf.pdf(bRec);
	float cosToCamera = Frame::cosTheta(bRec.wo);
	float cameraPdfA = dRec.pdf;
	if (dRec.measure != EArea)
		cameraPdfA = PdfWtoA(cameraPdfA, DistanceSquared(bRec.dg.P, dRec.p), cosToCamera);
	float wLight = cameraPdfA * (lightState.dVCM + lightState.dVC * bsdfRevPdfW);
	float miWeight = 1.f / (wLight + 1.f);
	Spectrum contrib = miWeight * lightState.throughput * bsdfFactor * directFactor;
	if (!contrib.isZero() && ::V(dRec.p, dRec.ref))
		g_Image.Splat(dRec.uv.x, dRec.uv.y, contrib);
}

CUDA_FUNC_IN Spectrum connectToLight(const BPTSubPathState& cameraState, BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, CudaRNG& rng)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	const Spectrum directFactor = g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
	const e_KernelLight* l = (const e_KernelLight*)dRec.object;
	if (!l)
		return Spectrum(0.0f);
	float directPdfW = dRec.pdf;
	DirectionSamplingRecord dirRec(-dRec.d);
	const float emissionPdfW = l->pdfDirection(dirRec, dRec);
	const float cosAtLight = dot(dRec.n, -dRec.d);

	bRec.wo = bRec.dg.toLocal(dRec.d);
	const float cosToLight = Frame::cosTheta(bRec.wo);
	const Spectrum bsdfFactor = mat.bsdf.f(bRec);
	const float bsdfDirPdfW = mat.bsdf.pdf(bRec) / cosToLight;
	const float bsdfRevPdfW = revPdf(mat, bRec) / Frame::cosTheta(bRec.wi);

	if (directFactor.isZero() || bsdfFactor.isZero())
		return Spectrum(0.0f);

	if (dRec.measure != ESolidAngle)
		directPdfW = PdfAtoW(directPdfW, DistanceSquared(bRec.dg.P, dRec.p), cosAtLight);

	const float wLight = bsdfDirPdfW / directPdfW;
	const float wCamera = emissionPdfW * cosToLight / (directPdfW * cosAtLight) * (cameraState.dVCM + cameraState.dVC * bsdfRevPdfW);

	const float misWeight = 1.f / (wLight + 1.f + wCamera);

	Spectrum contrib = misWeight * directFactor * bsdfFactor;

	if (contrib.isZero() || !::V(bRec.dg.P, dRec.p))
		return Spectrum(0.0f);
	return contrib;
}

CUDA_FUNC_IN Spectrum connectVertices(BPTVertex& emitterVertex, const BPTSubPathState& cameraState, BSDFSamplingRecord& bRec, const e_KernelMaterial& mat)
{
	const float dist2 = DistanceSquared(emitterVertex.dg.P, bRec.dg.P);
	float3 direction = normalize(emitterVertex.dg.P - bRec.dg.P);

	bRec.wo = bRec.dg.toLocal(direction);
	const Spectrum cameraBsdf = mat.bsdf.f(bRec);
	const float cosCamera = Frame::cosTheta(bRec.wo);
	const float cameraBsdfDirPdfW = mat.bsdf.pdf(bRec) / cosCamera;
	const float cameraBsdfRevPdfW = revPdf(mat, bRec) / Frame::cosTheta(bRec.wi);

	emitterVertex.bRec.wo = emitterVertex.bRec.dg.toLocal(-direction);
	const Spectrum emitterBsdf = emitterVertex.mat->bsdf.f(emitterVertex.bRec);
	const float cosLight = Frame::cosTheta(emitterVertex.bRec.wo);
	const float lightBsdfDirPdfW = emitterVertex.mat->bsdf.pdf(emitterVertex.bRec) / cosLight;
	const float lightBsdfRevPdfW = revPdf(*emitterVertex.mat, emitterVertex.bRec) / Frame::cosTheta(emitterVertex.bRec.wi);

	if (cameraBsdf.isZero() || emitterBsdf.isZero())
		return Spectrum(0.0f);

	const float cameraBsdfDirPdfA = PdfWtoA(cameraBsdfDirPdfW, dist2, cosLight);
	const float lightBsdfDirPdfA = PdfWtoA(lightBsdfDirPdfW, dist2, cosCamera);

	const float wLight = cameraBsdfDirPdfA * (emitterVertex.dVCM + emitterVertex.dVC * lightBsdfRevPdfW);
	const float wCamera = lightBsdfDirPdfA * (cameraState.dVCM + cameraState.dVC * cameraBsdfRevPdfW);

	const float misWeight = 1.f / (wLight + 1.f + wCamera);

	const float geometryTerm = 1.0f / dist2;
	const Spectrum contrib = (misWeight * geometryTerm) * cameraBsdf * emitterBsdf;
	if (contrib.isZero() || !::V(bRec.dg.P, emitterVertex.dg.P))
		return Spectrum(0.0f);
	return contrib;
}

CUDA_FUNC_IN void BPT(const float2& pixelPosition, e_Image& g_Image, CudaRNG& rng,
					  bool use_mis, int force_s, int force_t, float LScale)
{
	const int NUM_V_PER_PATH = 5;
	BPTVertex lightPath[NUM_V_PER_PATH];
	BPTSubPathState lightPathState;
	sampleEmitter(lightPathState, rng);
	int emitterPathLength = 1, emitterVerticesStored = 0;
	for (; emitterVerticesStored < NUM_V_PER_PATH; emitterPathLength++)
	{
		TraceResult r2 = k_TraceRay(lightPathState.r);
		if (!r2.hasHit())
			break;

		BPTVertex& v = lightPath[emitterVerticesStored];
		r2.getBsdfSample(lightPathState.r, rng, &v.bRec);

		if (emitterPathLength > 0)
			lightPathState.dVCM *= r2.m_fDist * r2.m_fDist;
		lightPathState.dVCM /= fabsf(Frame::cosTheta(v.bRec.wi));
		lightPathState.dVC /= fabsf(Frame::cosTheta(v.bRec.wi));

		//store in list
		if (r2.getMat().bsdf.hasComponent(ESmooth))
		{
			v.dVCM = lightPathState.dVCM;
			v.dVC = lightPathState.dVC;
			v.throughput = lightPathState.throughput;
			v.mat = &r2.getMat();
			emitterVerticesStored++;
		}

		//connect to camera
		if (r2.getMat().bsdf.hasComponent(ESmooth))
			connectToCamera(lightPathState, v.bRec, r2.getMat(), g_Image, rng);

		if (!sampleScattering(lightPathState, v.bRec, r2.getMat(), rng))
			break;
	}

	BPTSubPathState cameraState;
	sampleCamera(cameraState, rng, pixelPosition);
	Spectrum acc(0.0f);
	for (int camPathLength = 0; camPathLength < NUM_V_PER_PATH; camPathLength++)
	{
		TraceResult r2 = k_TraceRay(cameraState.r);
		if (!r2.hasHit())
		{
			//sample environment map

			break;
		}

		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		r2.getBsdfSample(cameraState.r, rng, &bRec);

		cameraState.dVCM *= r2.m_fDist * r2.m_fDist;
		cameraState.dVCM /= fabsf(Frame::cosTheta(bRec.wi));
		cameraState.dVC /= fabsf(Frame::cosTheta(bRec.wi));

		if (r2.LightIndex() != 0xffffffff)
		{
			acc += cameraState.throughput * gatherLight(cameraState, bRec, r2, rng, camPathLength + 1);
			break;
		}

		if (r2.getMat().bsdf.hasComponent(ESmooth))
			acc += cameraState.throughput * connectToLight(cameraState, bRec, r2.getMat(), rng);

		if (r2.getMat().bsdf.hasComponent(ESmooth))
		for (int emitterVertexIdx = 0; emitterVertexIdx < emitterVerticesStored; emitterVertexIdx++)
			acc += cameraState.throughput * lightPath[emitterVertexIdx].throughput * connectVertices(lightPath[emitterVertexIdx], cameraState, bRec, r2.getMat());

		if (!sampleScattering(cameraState, bRec, r2.getMat(), rng))
			break;
	}

	g_Image.AddSample(pixelPosition.x, pixelPosition.y, acc);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image,
		bool use_mis, int force_s, int force_t, float LScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		BPT(make_float2(x, y), g_Image, rng, use_mis, force_s, force_t, LScale);
	g_RNGData(rng);
}

void k_BDPT::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene, g_sRngs);
	int p = 16;
	if(w < 200 && h < 200)
		pathKernel << < dim3((w + p - 1) / p, (h + p - 1) / p, 1), dim3(p, p, 1) >> >(w, h, 0, 0, *I, use_mis, force_s, force_t, LScale);
	else
	{
		unsigned int q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				pathKernel << < dim3(q, q, 1), dim3(p, p, 1) >> >(w, h, pq * i, pq * j, *I, use_mis, force_s, force_t, LScale);
	}
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(1.0f / float(m_uPassesDone));
}

void k_BDPT::Debug(e_Image* I, int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	BPT(make_float2(pixel), *I, rng, use_mis, force_s, force_t, LScale);
	g_RNGData(rng);
}