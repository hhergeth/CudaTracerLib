#include "k_BDPT.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

struct BidirVertex
{
	float3 p;
	float3 n;
	Spectrum cumulative;
	int type;
	float d_i;
	union
	{
		TraceResult r2;
		const e_Sensor* s;
		const e_KernelLight* l;
	};

	CUDA_FUNC_IN BidirVertex()
	{

	}

	CUDA_FUNC_IN BidirVertex(TraceResult r2, const float3& p, const float3& n, const Spectrum& c)
		: p(p), n(normalize(n)), r2(r2)
	{
		cumulative = c;
		type = 1;
	}

	CUDA_FUNC_IN BidirVertex(const e_Sensor* s, const float3& p, const float3& n)
		: p(p), n(normalize(n)), s(s)
	{
		cumulative = Spectrum(1);
		type = 2;
	}

	CUDA_FUNC_IN BidirVertex(const e_KernelLight* l, const float3& p, const float3& n)
		: p(p), n(normalize(n)), l(l)
	{
		cumulative = Spectrum(1);
		type = 3;
	}

	CUDA_FUNC_IN bool isDelta() const
	{
		if (type == 1)
			return r2.getMat().bsdf.hasComponent(EDeltaReflection) || r2.getMat().bsdf.hasComponent(EDeltaTransmission);
		else return false;
		//unsigned int t = type == 2 ? s->As()->m_Type : l->As()->m_Type;
		//return t & EDeltaDirection;
	}

	CUDA_FUNC_IN bool isOnSurface() const
	{
		if (type == 1)
			return true;
		else return type == 2 ? s->As()->IsOnSurface() : l->As()->IsOnSurface();
	}

	CUDA_FUNC_IN Spectrum f(const BidirVertex* prev, const BidirVertex* next, bool pdff = false) const
	{
		if (type == 1)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(normalize(p - prev->p), p, &bRec);
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
			float pdf = 1;
			if (type == 2)
			{
				eval = s->evalPosition(pRec) * s->evalDirection(dRec, pRec);
				pdf = s->pdfPosition(pRec) * s->pdfDirection(dRec, pRec);
			}
			else if (type == 3)
			{
				eval = l->evalPosition(pRec) * l->evalDirection(dRec, pRec);
				pdf = l->pdfPosition(pRec) * l->pdfDirection(dRec, pRec);
			}
			float dp = AbsDot(pRec.n, dRec.d);
			e_AbstractEmitter* emitter = type == 2 ? (e_AbstractEmitter*)s->As() : (e_AbstractEmitter*)l->As();
			if (emitter->IsOnSurface() && dp != 0)
				eval /= dp;
			return eval / (pdff && pdf != 0 ? pdf : 1.0f);
		}
		else printf("Invalid vertex type : %d\n.", type);
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN static float G(const BidirVertex* a, const BidirVertex* b)
	{
		float3 theta = normalize(b->p - a->p);
		float r = 1.0f / DistanceSquared(a->p, b->p);
		if (a->isOnSurface() && !a->isDelta())
			r *= AbsDot(a->n, theta);
		if (b->isOnSurface() && !b->isDelta())
			r *= AbsDot(b->n, -theta);
		return r;
	}

	CUDA_FUNC_IN float pdf(const BidirVertex* prev, const BidirVertex* next, bool superSample) const
	{
		if (superSample)
		{
			PositionSamplingRecord pRec;
			pRec.p = p;
			pRec.n = n;
			pRec.measure = EDiscrete;
			return type == 2 ? s->pdfPosition(pRec) : l->pdfPosition(pRec);
		}
		if (type == 1)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(normalize(p - prev->p), p, &bRec);
			bRec.wo = dg.toLocal(normalize(next->p - p));
			return r2.getMat().bsdf.pdf(bRec, isDelta() ? EDiscrete : ESolidAngle) * (isDelta() ? 1.0f : 1.0f / DistanceSquared(p, next->p));
		}
		else if (type == 2 || type == 3)
		{
			PositionSamplingRecord pRec;
			pRec.p = p;
			pRec.n = n;
			DirectionSamplingRecord dRec;
			dRec.d = normalize(next->p - p);
			dRec.measure = ESolidAngle;
			return (type == 2 ? s->pdfDirection(dRec, pRec) : l->pdfDirection(dRec, pRec)) / DistanceSquared(p, next->p);
		}
		else printf("Invalid vertex type : %d\n.", type);
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Le(const BidirVertex* next) const
	{
		if (type == 1)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(make_float3(0), p, &bRec);
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
	static const int Num_Vertices_Per_Path = NUM_V_PER_PATH;
};

typedef BidirSampleT<5> BidirSample;

struct BidirPath
{
	int t;//eye length
	int s;//light length
	BidirSample& sample;

	CUDA_FUNC_IN BidirPath(BidirSample& sample, int t, int s)
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

	CUDA_FUNC_IN BidirVertex& operator[](int i)
	{
		return i < t ? sample.eyePath[i] : sample.lightPath[s - 1 - (i - t)];
	}

	CUDA_FUNC_IN const BidirVertex* vertexOrNull(int i) const
	{
		if (i >= 0 && i <= k())
			return &operator[](i);
		else return 0;
	}

	CUDA_FUNC_IN float p_forward(int i) const
	{
		return operator[](MAX(i, 0)).pdf(vertexOrNull(i - 1), vertexOrNull(i + 1), i == -1);
	}

	CUDA_FUNC_IN float p_backward(int i) const
	{
		return operator[](MIN(k(), i)).pdf(vertexOrNull(i + 1), vertexOrNull(i - 1), i == k() + 1);
	}

	CUDA_FUNC_IN float pdf(int k) const
	{
		float r = 1;
		for (int i = -1; i <= k - 1; i++)
			r *= p_forward(i);
		for (int i = k + 2; i <= this->k() + 1; i++)
			r *= p_backward(i);
		return r;
	}

	CUDA_FUNC_IN Spectrum f() const
	{
		const BidirVertex& ev = sample.eyePath[t - 1], &lv = sample.lightPath[s - 1];
		Spectrum f_ev = ev.f(t > 1 ? &sample.eyePath[t - 2] : 0, &lv);
		Spectrum f_lv = lv.f(s > 1 ? &sample.lightPath[s - 2] : 0, &ev);
		if (f_ev.isZero() || f_lv.isZero())
			return 0.0f;
		return ev.cumulative * f_ev * ::V(ev.p, lv.p) * BidirVertex::G(&ev, &lv) * f_lv * lv.cumulative;//(1.0f / DistanceSquared(ev.p, lv.p))
		/*Spectrum W_t = operator[](0).f(0, vertexOrNull(1), rng), L_t = operator[](k()).f(0, vertexOrNull(k() - 1), rng);
		Spectrum F_t(1.0f);
		float G_t = 1;
		for (int j = 0; j <= k() - 1; j++)
			G_t *= G(j, j + 1);
		for (int j = 1; j <= k() - 1; j++)
			F_t *= operator[](j).f(vertexOrNull(j - 1), vertexOrNull(j + 1), rng);
		return  W_t * G_t * F_t * L_t / pdf(t - 1, rng);*/
	}

	CUDA_FUNC_IN float misWeight(bool use_mis, int force_s, int force_t)
	{
		if (force_s != -1 && force_t != -1 && (this->t != force_t || this->s != force_s))
			return 0;
		if (!use_mis)
			return 1;

		int s = t - 1;
		float inv_w_s = 1;
		if (s == 0)
		{
			float dL_1 = s + 1 == k() ? 1.0f / p_backward(k() + 1) : (1.0f + p_forward(1) * operator[](2).d_i) / p_backward(2);
			inv_w_s = 1.0f + dL_1 * p_forward(0);
		}
		else if (s == k())
		{
			float dE_k = k() == 1 ? 1.0f / p_forward(0) : (1 + p_backward(k()) * operator[](k() - 1).d_i) / p_forward(k() - 1);
			inv_w_s = dE_k * p_backward(k() + 1) + 1;
		}
		else
		{
			float dE_s = s == 1 ? 1.0f / p_forward(0) : (1 + p_backward(s) * operator[](s).d_i) / p_forward(s - 1);
			float dL_s1 = (s + 1 == k()) ? 1.0f / p_backward(k() + 1) : (1 + p_forward(s + 1) * operator[](s + 2).d_i) / p_backward(s + 2);
			inv_w_s = dE_s * p_backward(s + 1) + 1 + dL_s1 * p_forward(s);
		}
		return 1.0f / inv_w_s;

		/*if (this->t == 0 || this->s == 0)
			return 1;
		float pdf_s = pdf(s);
		float div = pdf_s;
		float pdf_i = pdf_s;
		for (int i = s; i < k(); i++)
		{
			pdf_i = pdf_i * p_forward(i) / p_backward(i + 2);
			div += pdf_i;
		}
		pdf_i = pdf_s;
		for (int i = s; i > 0; i--)
		{
			pdf_i = pdf_i * p_backward(i + 1) / p_forward(i - 1);//attention the indices shifted! i -> i - 1
			div += pdf_i;
		}
		return pdf_s / div;*/
	}
};

CUDA_FUNC_IN void sampleSubPath(BidirVertex* vertices, int* N, Spectrum cumulative, Ray r, CudaRNG& rng, float pdf)
{
	Spectrum initialCumulative = cumulative;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	float lastPdf_fwd = pdf, lastPdf_bwd = 0, d_i1 = 0;
	while (*N < 5)
	{
		TraceResult r2 = k_TraceRay(r);
		if (!r2.hasHit())
			return;
		r2.getBsdfSample(r, rng, &bRec);

		vertices[*N] = BidirVertex(r2, dg.P, dg.sys.n, cumulative);
		float g = vertices[*N - 1].isDelta() ? 1 : 1.0f / DistanceSquared(vertices[*N - 1].p, vertices[*N].p);
		d_i1 = vertices[*N - 1].d_i = (1.0f + lastPdf_bwd * d_i1) / (lastPdf_fwd * g);//the probability was with respect to solid angle
		*N = *N + 1;

		float pdf;
		Spectrum f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());
		r = Ray(dg.P, bRec.getOutgoing());

		lastPdf_fwd = pdf;
		swapk(bRec.wo, bRec.wi);
		lastPdf_bwd = r2.getMat().bsdf.pdf(bRec, vertices[*N - 1].isDelta() ? EDiscrete : ESolidAngle);

		cumulative *= f;
		float p = (cumulative / initialCumulative).max();
		if (rng.randomFloat() < p)
			cumulative /= p;
		else break;
		r2.Init();
	}
	/*vertices[0].d_i = 1 / vertices[0].pdf(0, vertices + 1, false);
	for (int i = 1; i < *N - 1; i++)
	{
		vertices[i].d_i = (1 + vertices[i].pdf(vertices + (i + 1), vertices + (i - 1), false) * vertices[i - 1].d_i) / 
						  vertices[i - 1].pdf(i > 1 ? vertices + (i - 2) : 0, vertices + i,false);
	}*/
}

CUDA_FUNC_IN void SBDPT(const float2& pixelPosition, e_Image& g_Image, CudaRNG& rng,
						bool use_mis, int force_s, int force_t, float LScale)
{
	BidirSample bSample;
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;

	Spectrum imp = g_SceneData.m_Camera.samplePosition(pRec, rng.randomFloat2(), &pixelPosition);
	imp *= g_SceneData.m_Camera.sampleDirection(dRec, pRec, rng.randomFloat2(), &pixelPosition);
	bSample.eyePath[0] = BidirVertex(&g_SceneData.m_Camera, pRec.p, pRec.n);
	float pdf_0 = dRec.pdf;
	Ray rEye(pRec.p, dRec.d);

	Spectrum Le = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2());
	Le *= ((const e_KernelLight*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());
	bSample.lightPath[0] = BidirVertex((const e_KernelLight*)pRec.object, pRec.p, pRec.n);
	float pdf_k = dRec.pdf;
	Ray rLight(pRec.p, dRec.d);

	bSample.n_E = bSample.n_L = 1;
	sampleSubPath(bSample.eyePath, &bSample.n_E, imp, rEye, rng, pdf_0);
	sampleSubPath(bSample.lightPath, &bSample.n_L, Le, rLight, rng, pdf_k);

	Spectrum acc(0.0f);
	for (int t = 1; t <= bSample.n_E; t++)
	{
		for (int s = 1; s <= bSample.n_L; s++)
		{
			BidirPath path(bSample, t, s);
			if (t == 1)
			{
				DirectSamplingRecord dRec(path[1].p, path[1].n);
				Spectrum value = path[1].cumulative * g_SceneData.m_Camera.sampleDirect(dRec, rng.randomFloat2());
				Spectrum f = path[1].f(path.vertexOrNull(2), &path[0], true);
				float miWeight = path.misWeight(use_mis, force_s, force_t);
				if (!value.isZero() && ::V(dRec.p, dRec.ref))
					g_Image.Splat(dRec.uv.x, dRec.uv.y, value * f * miWeight * LScale);
			}
			else if (s == 1)
			{
				int q = path.k() - 1;
				DirectSamplingRecord dRec(path[q].p, path[q].n);
				Spectrum value = path[q].cumulative * path[path.k()].l->sampleDirect(dRec, rng.randomFloat2());
				BidirVertex tmp = path[path.k()];
				path[path.k()] = BidirVertex(tmp.l, dRec.p, dRec.n);
				value *= path[q].f(path.vertexOrNull(q - 1), &path[path.k()], true);
				float miWeight = path.misWeight(use_mis, force_s, force_t);
				path[path.k()] = tmp;
				if (!value.isZero() && ::V(dRec.p, dRec.ref))
					acc += value * miWeight;
			}
			else
			{
				acc += path.f() * path.misWeight(use_mis, force_s, force_t);
			}
		}
		
		//edge case with s = 0 -> eye path hit a light source
		if (t >= 2 && bSample.eyePath[t - 1].r2.LightIndex() != 0xffffffff)
		{
			BidirPath path(bSample, t, 0);
			BidirVertex tmp = bSample.eyePath[t - 1];
			bSample.eyePath[t - 1] = BidirVertex(&g_SceneData.m_sLightData[tmp.r2.LightIndex()], tmp.p, tmp.n);
			float miWeight = path.misWeight(use_mis, force_s, force_t);
			Spectrum le = bSample.eyePath[t - 1].f(0, &bSample.lightPath[t - 2]);
			//acc += le * tmp.cumulative * miWeight;
			bSample.eyePath[t - 1] = tmp;
		}
	}
	g_Image.AddSample(pixelPosition.x, pixelPosition.y, acc * LScale);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image,
		bool use_mis, int force_s, int force_t, float LScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		SBDPT(make_float2(x, y), g_Image, rng, use_mis, force_s, force_t, LScale);
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
	I->DoUpdateDisplay(1.0f / float(m_uPassesDone * 3));
}

void k_BDPT::Debug(e_Image* I, int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	SBDPT(make_float2(pixel), *I, rng, use_mis, force_s, force_t, LScale);
	g_RNGData(rng);
}