#include "k_BDPT.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

struct BidirVertex
{
	float3 p;
	float3 n;
	Spectrum cumulative;
	float cumulativePdf;
	int type;
	union
	{
		TraceResult r2;
		const e_Sensor* s;
		const e_KernelLight* l;
	};

	CUDA_FUNC_IN BidirVertex()
	{

	}

	CUDA_FUNC_IN BidirVertex(TraceResult r2, const float3& p, const float3& n, const Spectrum& c, float pdf)
		: p(p), n(normalize(n)), r2(r2), cumulativePdf(pdf)
	{
		cumulative = c;
		type = 1;
	}

	CUDA_FUNC_IN BidirVertex(const e_Sensor* s, const float3& p, const float3& n, float pdf)
		: p(p), n(normalize(n)), s(s), cumulativePdf(pdf)
	{
		cumulative = Spectrum(1);
		type = 2;
	}

	CUDA_FUNC_IN BidirVertex(const e_KernelLight* l, const float3& p, const float3& n, float pdf)
		: p(p), n(normalize(n)), l(l), cumulativePdf(pdf)
	{
		cumulative = Spectrum(1);
		type = 3;
	}

	CUDA_FUNC_IN bool isDelta() const
	{
		if (type == 1)
			return r2.getMat().bsdf.hasComponent(EDeltaReflection) || r2.getMat().bsdf.hasComponent(EDeltaTransmission);
		else return false;
	}

	CUDA_FUNC_IN Spectrum f(const BidirVertex* prev, const BidirVertex* next, CudaRNG& rng) const
	{
		if (type == 1)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(normalize(p - prev->p), p, rng, &bRec);
			bRec.wo = dg.toLocal(normalize(next->p - p));
			return r2.getMat().bsdf.f(bRec, isDelta() ? EDiscrete : ESolidAngle) / (isDelta() ? 1.0f : Frame::cosTheta(bRec.wo) != 0 ? abs(Frame::cosTheta(bRec.wo)) : 1);
		}
		else if (type == 2 || type == 3)
		{
			PositionSamplingRecord pRec;
			pRec.p = p;
			pRec.n = n;
			pRec.measure = EDiscrete;
			DirectionSamplingRecord dRec;
			dRec.d = normalize(next->p - p);
			dRec.measure = ESolidAngle;
			Spectrum eval(0.0f);
			if (type == 2)
				eval = s->evalPosition(pRec) * s->evalDirection(dRec, pRec);
			else if (type == 3)
				eval = l->evalPosition(pRec) * l->evalDirection(dRec, pRec);
			float dp = AbsDot(pRec.n, dRec.d);
			return eval / (dp != 0 ? dp : 1);
		}
		else printf("Invalid vertex type : %d\n.", type);
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdf(const BidirVertex* prev, const BidirVertex* next, bool superSample, CudaRNG& rng) const
	{
		if (superSample)
		{
			PositionSamplingRecord pRec;
			pRec.p = p;
			pRec.n = n;
			pRec.measure = EDiscrete;
			if (type == 1)
				return g_SceneData.m_sLightData[r2.LightIndex()].pdfPosition(pRec);
			return type == 2 ? s->pdfPosition(pRec) : l->pdfPosition(pRec);
		}
		const e_KernelLight* l = this->l;
		bool isLight = false;
		if (type == 1 && (prev == 0 || next == 0) && r2.LightIndex() != 0xffffffff)
		{
			isLight = true;
			l = &g_SceneData.m_sLightData[r2.LightIndex()];
		}
		if (type == 1 && !isLight)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(normalize(p - prev->p), p, rng, &bRec);
			bRec.wo = dg.toLocal(normalize(next->p - p));
			return r2.getMat().bsdf.pdf(bRec, isDelta() ? EDiscrete : ESolidAngle) * (isDelta() ? 1.0f : ::G(n, next->n, p, next->p));
		}
		else if (type == 2 || type == 3 || isLight)
		{
			PositionSamplingRecord pRec;
			pRec.p = p;
			pRec.n = n;
			DirectionSamplingRecord dRec;
			dRec.d = normalize(next->p - p);
			dRec.measure = ESolidAngle;
			return (type == 2 ? s->pdfDirection(dRec, pRec) : l->pdfDirection(dRec, pRec)) * ::G(n, next->n, p, next->p);
		}
		else printf("Invalid vertex type : %d\n.", type);
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Le(const BidirVertex* next, CudaRNG& rng) const
	{
		if (type == 1)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(make_float3(0), p, rng, &bRec);
			return r2.Le(p, dg.sys, normalize(next->p - p));
		}
		else return 0.0f;
	}
};

template<int NUM_V_PER_PATH> struct BidirSampleT
{
	BidirVertex eyePath[NUM_V_PER_PATH];
	BidirVertex lightPath[NUM_V_PER_PATH];
	int n_E;
	int n_L;
};

typedef BidirSampleT<5> BidirSample;

struct BidirPath
{
	int t;//eye length
	int s;//light length
	const BidirSample& sample;

	CUDA_FUNC_IN BidirPath(const BidirSample& sample, int t, int s)
		: t(t), s(s), sample(sample)
	{

	}

	CUDA_FUNC_IN int k() const
	{
		return s + t - 1;
	}

	CUDA_FUNC_IN const BidirVertex& operator[](int i) const
	{
		return i < t ? sample.eyePath[i] : sample.lightPath[s - 1 - (i - t)];
	}

	CUDA_FUNC_IN const BidirVertex* vertexOrNull(int i) const
	{
		if (i >= 0 && i <= k())
			return &operator[](i);
		else return 0;
	}

	CUDA_FUNC_IN float p_forward(int i, CudaRNG& rng) const
	{
		return operator[](MAX(i, 0)).pdf(vertexOrNull(i - 1), vertexOrNull(i + 1), i == -1, rng);
	}

	CUDA_FUNC_IN float p_backward(int i, CudaRNG& rng) const
	{
		return operator[](MIN(k(), i)).pdf(vertexOrNull(i + 1), vertexOrNull(i - 1), i == k() + 1, rng);
	}

	CUDA_FUNC_IN float p_L(int i, CudaRNG& rng) const
	{
		if (i == 0)
			return 1;
		else if (i == 1)
			return operator[](k()).pdf(0, 0, true, rng);
		else return operator[](k() - (i - 1)).pdf(vertexOrNull(k() - (i - 2)), vertexOrNull(k() - i), false, rng);
	}

	CUDA_FUNC_IN float p_E(int i, CudaRNG& rng) const
	{
		if (i == 0)
			return 1;
		else if (i == 1)
			return operator[](0).pdf(0, 0, true, rng);
		else return operator[](i - 1).pdf(vertexOrNull(i - 2), vertexOrNull(i), false, rng);
	}

	CUDA_FUNC_IN float pdf(int k, CudaRNG& rng) const
	{
		//int s = k, t = this->k() - s;
		//return p_E(s, rng) * p_L(t, rng);
		float r = 1;
		for (int i = -1; i <= k - 1; i++)
			r *= p_forward(i, rng);
		for (int i = k + 2; i <= this->k() + 1; i++)
			r *= p_backward(i, rng);
		return r;
	}

	CUDA_FUNC_IN float G(int i, int j) const
	{
		const BidirVertex& v_i = operator[](i), &v_j = operator[](j);
		float g = ::G(v_i.n, v_j.n, v_i.p, v_j.p), f = ::V(v_i.p, v_j.p);
		return g * f;
	}

	CUDA_FUNC_IN Spectrum f(CudaRNG& rng) const
	{
		const BidirVertex& ev = sample.eyePath[t - 1], &lv = sample.lightPath[s - 1];
		Spectrum f_ev = ev.f(t > 1 ? &sample.eyePath[t - 2] : 0, &lv, rng);
		Spectrum f_lv = lv.f(s > 1 ? &sample.lightPath[s - 2] : 0, &ev, rng);
		return ev.cumulative * f_ev * ::V(ev.p, lv.p) * ::G(ev.n, lv.n, ev.p, lv.p) * f_lv * lv.cumulative;//(1.0f / DistanceSquared(ev.p, lv.p))
		/*Spectrum W_t = operator[](0).f(0, vertexOrNull(1), rng), L_t = operator[](k()).f(0, vertexOrNull(k() - 1), rng);
		Spectrum F_t(1.0f);
		float G_t = 1;
		for (int j = 0; j <= k() - 1; j++)
			G_t *= G(j, j + 1);
		for (int j = 1; j <= k() - 1; j++)
			F_t *= operator[](j).f(vertexOrNull(j - 1), vertexOrNull(j + 1), rng);
		return  W_t * G_t * F_t * L_t / pdf(t - 1, rng);*/
	}
};

CUDA_FUNC_IN void sampleSubPath(BidirVertex* vertices, int* N, Spectrum cumulative, float cumulativePdf, Ray r, CudaRNG& rng)
{
	Spectrum initialCumulative = cumulative;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while (*N < 5)
	{
		TraceResult r2 = k_TraceRay(r);
		if (!r2.hasHit())
			return;
		r2.getBsdfSample(r, rng, &bRec);
		vertices[*N] = BidirVertex(r2, dg.P, dg.sys.n, cumulative, cumulativePdf);
		*N = *N + 1;
		float pdf;
		Spectrum f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());
		cumulativePdf *= pdf;
		cumulative *= f;
		float p = (cumulative / initialCumulative).max();
		if (rng.randomFloat() < p)
			cumulative /= p;
		else break;
		r = Ray(dg.P, bRec.getOutgoing());
		r2.Init();
	}
}

CUDA_FUNC_IN float misWeight(BidirPath& P, CudaRNG& rng)
{
	//return 1;
	/*float div = 0;
	for (int i = 0; i <= P.k(); i++)
		div += P.pdf(i, rng);
	return P.pdf(P.t - 1, rng) / div;*/
	int s = P.t - 1;
	float pdf_s = P.pdf(s, rng);
	float div = pdf_s;
	float pdf_i = pdf_s;
	for (int i = s; i < P.k(); i++)
	{
		pdf_i = pdf_i * P.p_forward(i, rng) / P.p_backward(i + 2, rng);
		div += pdf_i;
	}
	pdf_i = pdf_s;
	for (int i = s; i > 0; i--)
	{
		pdf_i = pdf_i * P.p_backward(i + 1, rng) / P.p_forward(i - 1, rng);//attention the indices shifted! i -> i - 1
		div += pdf_i;
	}
	return pdf_s / div;
}

CUDA_FUNC_IN void SBDPT(const float2& pixelPosition, e_Image& g_Image, CudaRNG& rng)
{
	BidirSample bSample;
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;
	float2 sample = rng.randomFloat2();
	Spectrum imp = g_SceneData.m_Camera.samplePosition(pRec, sample, &pixelPosition);
	imp *= g_SceneData.m_Camera.sampleDirection(dRec, pRec, sample, &pixelPosition);
	float pdfEye = pRec.pdf * dRec.pdf;
	bSample.eyePath[0] = BidirVertex(&g_SceneData.m_Camera, pRec.p, pRec.n, 1);
	Ray rEye(pRec.p, dRec.d);

	sample = rng.randomFloat2();
	Spectrum Le = g_SceneData.sampleEmitterPosition(pRec, sample);
	Le *= ((const e_KernelLight*)pRec.object)->sampleDirection(dRec, pRec, sample);
	float pdfLight = pRec.pdf * dRec.pdf;
	bSample.lightPath[0] = BidirVertex((const e_KernelLight*)pRec.object, pRec.p, pRec.n, 1);
	Ray rLight(pRec.p, dRec.d);

	bSample.n_E = bSample.n_L = 1;
	sampleSubPath(bSample.eyePath, &bSample.n_E, imp, pdfEye, rEye, rng);
	sampleSubPath(bSample.lightPath, &bSample.n_L, Le, pdfLight, rLight, rng);

	Spectrum acc(0.0f), accLight(0.0f);
	for (int t = 2; t <= bSample.n_E; t++)
	{
		for (int s = 1; s <= bSample.n_L; s++)
		{
			BidirPath path(bSample, t, s);
			float miWeight = misWeight(path, rng);
			if (t >= 2)
				acc += path.f(rng) * miWeight;
			else accLight += path.f(rng) * miWeight;
		}

		//edge case with s = 0 -> eye path hit a light source
		if (bSample.eyePath[t - 1].r2.LightIndex() != 0xffffffff)
		{
			BidirPath path(bSample, t, 0);
			acc += bSample.eyePath[t - 1].Le(&bSample.eyePath[t - 2], rng) * bSample.eyePath[t - 1].cumulative * misWeight(path, rng);
		}
	}
	g_Image.Splat(pixelPosition.x, pixelPosition.y, accLight);
	g_Image.AddSample(pixelPosition.x, pixelPosition.y, acc);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		//Li(g_Image, rng, x, y);
		SBDPT(make_float2(x, y),g_Image, rng);
	g_RNGData(rng);
}

void k_BDPT::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene, g_sRngs);
	int p = 16;
	if(w < 200 && h < 200)
		pathKernel<<< dim3((w + p - 1) / p, (h + p - 1) / p,1), dim3(p, p, 1)>>>(w, h, 0, 0, *I);
	else
	{
		unsigned int q = 8, pq = p * q;
		int nx = w / pq + 1, ny = h / pq + 1;
		for(int i = 0; i < nx; i++)
			for(int j = 0; j < ny; j++)
				pathKernel<<< dim3(q, q,1), dim3(p, p, 1)>>>(w, h, pq * i, pq * j, *I);
	}
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(float(w*h) / float(m_uPassesDone * w * h));
}

void k_BDPT::Debug(e_Image* I, int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	SBDPT(make_float2(pixel),*I, rng);
}