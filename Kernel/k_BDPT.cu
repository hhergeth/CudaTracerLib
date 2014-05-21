#include "k_BDPT.h"
#include "k_TraceHelper.h"
#include <time.h>
#include "k_TraceAlgorithms.h"

CUDA_FUNC_IN bool V(const float3& a, const float3& b)
{
	float3 d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l);
}
/*
#define MAX_SUBPATH_LENGTH 5
struct BidirVertex
{
	float cosi, coso, pdf, pdfR, tPdf, tPdfR, dAWeight, dARWeight, rr, rrR, d2, padding;
	// flux: flux from the beginning of the path up to this vertex, excluding scattering at this vertex, probability weighted
	Spectrum flux;
	// bsdf: the BSDF at that vertex
	TraceResult r2;
	// wi: direction towards the light
	// wo: direcion towards the eye
	float3 wi, wo;
	// p: the location of this vertex
	float3 p;
	EBSDFType flags;

	CUDA_FUNC_IN void EyeConnect(e_Image& img, CudaRNG& rng, const Spectrum& col)
	{
		DirectSamplingRecord dRec(p, wi, make_float2(0));
		Spectrum localLe = col * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
		if(V(dRec.p, p))
		{
			unsigned int w,h;
			img.getExtent(w, h);
			if(dRec.uv.x >= 0 && dRec.uv.x < w && dRec.uv.y >= 0 && dRec.uv.y < h)
			{
				img.Splat((int)dRec.uv.x, (int)dRec.uv.y, localLe);
			}
		}
	}
};

struct Path
{
	BidirVertex eyePath[MAX_SUBPATH_LENGTH];
	BidirVertex lightPath[MAX_SUBPATH_LENGTH];
	unsigned int eyePathLength;
	unsigned int lightPathLength;
	const e_KernelLight* light;
	Spectrum Le;
	CUDA_FUNC_IN Path()
	{
		eyePathLength = 0;
		lightPathLength = 0;
	}
};

#define maxEyeDepth 5u
#define maxLightDepth 5u
#define rrStart 3
#define BSDF_SPECULAR EDelta
#define lightThreshold 0.0f
#define eyeThreshold 0.0f
#define directSamplingCount 1
#define shadowRayCount  1
#define lightRayCount 1

CUDA_FUNC_IN float WeightPath(BidirVertex* eye, u_int nEye, BidirVertex* light, u_int nLight, float pdfLightDirect, bool isLightDirect)
{
	const float pBase = (nLight == 1 && isLightDirect) ?
		fabsf(light[0].dAWeight) / pdfLightDirect : 1.f;
	float weight = 1.f, p = pBase;
	if (nLight == 1)
	{
		if (isLightDirect)
		{
			if ((light[0].flags & EDelta) == 0 && maxLightDepth > 0)
				weight += pBase * pBase;
		} else {
			const float pDirect = pdfLightDirect / fabsf(light[0].dAWeight);
			weight += pDirect * pDirect;
		}
	}
	const u_int nLightExt = MIN(nEye, maxLightDepth - MIN(maxLightDepth, nLight));
	for (u_int i = 1; i <= nLightExt; ++i) {
		// Exit if the path is impossible
		if (!(eye[nEye - i].dARWeight > 0.f && eye[nEye - i].dAWeight > 0.f))
			break;
		// Compute new path relative probability
		p *= eye[nEye - i].dAWeight / eye[nEye - i].dARWeight;
		// Adjust for round robin termination
		if (nEye - i > rrStart)
			p /= eye[nEye - i - 1].rrR;
		if (nLight + i > rrStart + 1) {
			if (i == 1)
				p *= light[nLight - 1].rr;
			else
				p *= eye[nEye - i + 1].rr;
		}
		// The path can only be obtained if none of the vertices
		// is specular
		if ((eye[nEye - i].flags & BSDF_SPECULAR) == 0 &&
			(i == nEye || (eye[nEye - i - 1].flags & BSDF_SPECULAR) == 0))
			weight += p * p;
	}
	p = pBase;
	u_int nEyeExt = MIN(nLight, maxEyeDepth - MIN(maxEyeDepth, nEye));
	for (u_int i = 1; i <= nEyeExt; ++i) {
		// Exit if the path is impossible
		if (!(light[nLight - i].dARWeight > 0.f && light[nLight - i].dAWeight > 0.f))
				break;
		// Compute new path relative probability
		p *= light[nLight - i].dARWeight / light[nLight - i].dAWeight;
		// Adjust for round robin termination
		if (nLight - i > rrStart)
			p /= light[nLight - i - 1].rr;
		if (nEye + i > rrStart + 1) {
			if (i == 1)
				p *= eye[nEye - 1].rrR;
			else
				p *= light[nLight - i + 1].rrR;
		}
		// The path can only be obtained if none of the vertices
		// is specular
		if ((light[nLight - i].flags & BSDF_SPECULAR) == 0 &&
			(i == nLight || (light[nLight - i - 1].flags & BSDF_SPECULAR) == 0))
			weight += p * p;
		// Check for direct path
		// Light path has at least 2 vertices here
		// The path can only be obtained if the second vertex
		// is not specular.
		// Even if the light source vertex is specular,
		// the special sampling for direct lighting will get it
		if (i == nLight - 1 && (light[1].flags & BSDF_SPECULAR) == 0) {
			const float pDirect = p * pdfLightDirect / fabsf(light[0].dAWeight);
			weight += pDirect * pDirect;
		}
	}
	return weight;
}

CUDA_FUNC_IN bool EvalPath(CudaRNG& rng, BidirVertex* eye, u_int nEye, BidirVertex* light, u_int nLight, float pdfLightDirect, bool isLightDirect, float *weight, Spectrum *L)
{
	const float epsilon = EPSILON;
	*weight = 0.f;
	BidirVertex &eyeV(eye[nEye - 1]);
	BidirVertex &lightV(light[nLight - 1]);
	
	BSDFSamplingRecord lRec;
	lightV.r2.getBsdfSample(Ray(lightV.p, -lightV.wi), rng, &lRec, lightV.wo);
	lRec.typeMask = lightV.flags;
	const e_KernelMaterial& lMat = lightV.r2.getMat();
	BSDFSamplingRecord eRec;
	eyeV.r2.getBsdfSample(Ray(eyeV.p, -eyeV.wo), rng, &eRec, eyeV.wi);
	eRec.typeMask = eyeV.flags;
	const e_KernelMaterial& eMat = eyeV.r2.getMat();

	eyeV.flags = EBSDFType(~BSDF_SPECULAR);
	const float3 ewi(normalize(lightV.p - eyeV.p));
	const Spectrum ef = eMat.bsdf.f(eRec);
	if (ef.isZero())
		return false;
	lightV.flags = EBSDFType(~BSDF_SPECULAR);
	const float3 lwo(-1.0f * ewi);
	const Spectrum lf = lMat.bsdf.f(lRec);
	if (lf.isZero())
		return false;
	swapk(eRec.wo, eRec.wi);
	const float epdfR = eMat.bsdf.pdf(eRec);
	swapk(eRec.wo, eRec.wi);
	const float lpdf = lMat.bsdf.pdf(lRec);
	float ltPdf = 1.f;
	float etPdfR = 1.f;
	const float d2 = DistanceSquared(eyeV.p, lightV.p);
	if (d2 < epsilon)
		return false;
	// Connect eye and light vertices
	*L *= lightV.flux * lf * ef * eyeV.flux / d2;
	if (L->isZero())
		return false;
	const float ecosi = AbsDot(ewi, eRec.map.sys.n);//eyeV.bsdf->ng
	const float epdf = eMat.bsdf.pdf(eRec);
	if (nEye == 1)
		eyeV.rr = 1.f;
	else if (ecosi * epdf > epsilon)
		eyeV.rr = MIN(1.f, MAX(lightThreshold, ef.average() * eyeV.coso / (ecosi * epdf)));
	else
		eyeV.rr = 0.f;
	if (epdfR > epsilon)
		eyeV.rrR = MIN(1.f, MAX(eyeThreshold, ef.average() / epdfR));
	else
		eyeV.rrR = 0.f;
	eyeV.dAWeight = lpdf * ltPdf / d2;
	eyeV.dAWeight *= ecosi;
	const float eWeight = nEye > 1 ? eye[nEye - 2].dAWeight : 0.f;
	if (nEye > 1) {
		eye[nEye - 2].dAWeight = epdf * eyeV.tPdf / eye[nEye - 2].d2;
		eye[nEye - 2].dAWeight *= eye[nEye - 2].cosi;
	}
	// Evaluate factors for light path weighting
	const float lcoso = AbsDot(lwo, lRec.map.sys.n);//lightV.bsdf->ng
	swapk(lRec.wo, lRec.wi);
	const float lpdfR = lMat.bsdf.pdf(lRec);
	swapk(lRec.wo, lRec.wi);
	if (lpdf > epsilon)
		lightV.rr = MIN(1.f, MAX(lightThreshold, lf.average() / lpdf));
	else
		lightV.rr = 0.f;
	if (nLight == 1)
		lightV.rrR = 1.f;
	else if (lcoso * lpdfR > epsilon)
		lightV.rrR = MIN(1.f, MAX(eyeThreshold, lf.average() * lightV.cosi / (lcoso * lpdfR)));
	else
		lightV.rrR = 0.f;
	lightV.dARWeight = epdfR * etPdfR / d2;
	lightV.dARWeight *= lcoso;
	if (nLight > 1) {
		light[nLight - 2].dARWeight = lpdfR * lightV.tPdfR / light[nLight - 2].d2;
		light[nLight - 2].dARWeight *= light[nLight - 2].coso;
	}
	const float w = 1.f / WeightPath(eye, nEye, light, nLight,
		pdfLightDirect, isLightDirect);
	*weight = w;
	*L *= w;
	if (nEye > 1)
		eye[nEye - 2].dAWeight = eWeight;
	// return back some eye data
	eyeV.wi = ewi;
	eyeV.d2 = d2;

	return true;
}

CUDA_FUNC_IN bool GetDirectLight(CudaRNG& rng, BidirVertex* eye, u_int length, const e_KernelLight *light, float lightWeight, float directWeight, Spectrum *Ld, float *weight)
{
	BidirVertex &vE(eye[length - 1]);
	BidirVertex vL;
	BSDFSamplingRecord eRec;
	vE.r2.getBsdfSample(Ray(vE.p, -vE.wo), rng, &eRec, vE.wi);
	eRec.typeMask = vE.flags;
	DirectSamplingRecord dRec(vE.p, eRec.map.sys.n, eRec.map.uv); 
	*Ld = light->sampleDirect(dRec, rng.randomFloat2());
	float ePdfDirect = light->pdfDirect(dRec);
	vL.dAWeight = light->pdfPosition(dRec);
	vL.p = dRec.p;
	vL.wi = dRec.n;
	vL.cosi = 1.0f;
	vL.dAWeight *= lightWeight;
	vL.flux = Spectrum(1.f / directWeight);
	vL.tPdf = 1.f;
	vL.tPdfR = 1.f;
	if (light->IsDeltaLight())
		vL.dAWeight = -vL.dAWeight;
	ePdfDirect *= directWeight;
	if (!EvalPath(rng, eye, length, &vL, 1, ePdfDirect, true, weight, Ld))
		return false;
	return true;
}

CUDA_FUNC_IN void Li(e_Image& img, CudaRNG& rng,int x, int y)
{
	Ray ray;
	const e_Sensor* sens;
	BidirVertex eyePath[MAX_SUBPATH_LENGTH];
	BidirVertex lightPath[MAX_SUBPATH_LENGTH];
	Spectrum W = g_SceneData.sampleSensorRay(ray, sens, rng.randomFloat2(), rng.randomFloat2());
	BidirVertex &eye0(eyePath[0]);
	eye0.p = ray.origin;
	eye0.wo = ray.direction;
	eye0.coso = 1;
	eye0.dARWeight = 0.f;
	eye0.flux = W;
	eye0.r2.Init();
	u_int nEye = 1;

	for (u_int l = 0; l < directSamplingCount; ++l)
	{
		const u_int offset = l * (1 + shadowRayCount * 3);
		Spectrum Ld;
		float dWeight, dPdf;
		const e_KernelLight* light = g_SceneData.sampleLight(dPdf, rng.randomFloat2());
		if (!light)
			break;
		dPdf *= shadowRayCount;
		const float lPdf = g_SceneData.pdfLight(light) * lightRayCount;
		for (u_int s = 0;s < shadowRayCount; ++s)
		{
			if(GetDirectLight(rng, eyePath, 1, light, lPdf, dPdf, &Ld, &dWeight))
			{
				eye0.EyeConnect(img, rng, Ld * dWeight);
			}
		}
	}

	if (maxEyeDepth > 1)
	{

	}
}
*/
#define MAX_SUBPATH_LENGTH 5
struct PathVertex
{
	float3 p;
	float3 wi;
	float3 wo;
	TraceResult r2;
	Spectrum cumulative;
};
struct Path
{
	PathVertex EyePath[MAX_SUBPATH_LENGTH];
	PathVertex LightPath[MAX_SUBPATH_LENGTH];
	unsigned int s;
	unsigned int t;
	CUDA_FUNC_IN Path()
	{
		s = 0;
		t = 0;
	}
};

CUDA_FUNC_IN float G(const float3& N_x, const float3& N_y, const float3& x, const float3& y)
{
	float3 theta = normalize(y - x);
	return AbsDot(N_x, theta) * AbsDot(N_y, -theta) / DistanceSquared(x, y);
}

CUDA_FUNC_IN void randomWalk(PathVertex* vertices, unsigned int* N, Ray r, CudaRNG& rng, bool eye)
{
	Spectrum cumulative(1.0f); 
	while(*N < MAX_SUBPATH_LENGTH)
	{
		TraceResult r2 = k_TraceRay(r);
		if(!r2.hasHit())
			return;
		PathVertex& v = vertices[*N];
		(*N)++;
		v.r2 = r2;
		v.p = r(r2.m_fDist);
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		if(eye)
		{
			v.wo = -r.direction;
			r.direction = v.wi = normalize(bRec.getOutgoing());
			cumulative *= f;
		}
		else
		{
			v.wi = -r.direction;
			r.direction = v.wo = normalize(bRec.getOutgoing());
			cumulative *= f;
		}
		v.cumulative = cumulative;
		if(cumulative.isZero())
			return;
		r.origin = v.p;
	}
}

CUDA_FUNC_IN Spectrum evalPath(const Path& P, int nEye, int nLight, CudaRNG& rng)
{
	const PathVertex& ev = P.EyePath[nEye - 1];
	const PathVertex& lv = P.LightPath[nLight - 1];
	Spectrum L(1.0f);
	if(nEye > 1)
		L *= P.EyePath[nEye - 2].cumulative;
	if(nLight > 1)
		L *= P.LightPath[nLight - 2].cumulative;

	float3 dir = normalize(lv.p - ev.p);
	BSDFSamplingRecord bRec;
	ev.r2.getBsdfSample(Ray(ev.p, -1.0f * ev.wi), rng, &bRec, dir);
	L *= ev.r2.getMat().bsdf.f(bRec);
	float pdf_i = ev.r2.getMat().bsdf.pdf(bRec);
	float3 N_x = bRec.map.sys.n;
	lv.r2.getBsdfSample(Ray(lv.p, dir), rng, &bRec, lv.wo);
	L *= lv.r2.getMat().bsdf.f(bRec);
	float3 N_y = bRec.map.sys.n;
	float g = G(N_x, N_y, ev.p, lv.p);
	L *= g;

	const float misWeight = MonteCarlo::PowerHeuristic(1, 1, 1, pdf_i);
	//L *= misWeight;

	return L;
}

CUDA_FUNC_IN float pathWeight(int i, int j)
{
	return 1;
}

CUDA_FUNC_IN void BDPT(int x, int y, int w, int h, e_Image& g_Image, CudaRNG& rng)
{
	Path P;
	Ray r;
	Spectrum imp = g_SceneData.sampleSensorRay(r, make_float2(x, y), rng.randomFloat2());
	randomWalk(P.EyePath, &P.s, r, rng, true);
	const e_KernelLight* light;
	Spectrum Le  = g_SceneData.sampleEmitterRay(r, light, rng.randomFloat2(), rng.randomFloat2());
	randomWalk(P.LightPath, &P.t, r, rng, false);
	
	Spectrum L(0.0f);
	BSDFSamplingRecord bRec;
	for(unsigned int i = 1; i < P.s + 1; i++)
	{
		const PathVertex& ev = P.EyePath[i - 1];
		if(!ev.r2.hasHit())
			break;//urgs wtf?

		//case ii
		ev.r2.getBsdfSample(Ray(ev.p, -1.0f * ev.wo), rng, &bRec);
		DirectSamplingRecord dRec(ev.p, bRec.map.sys.n);
		Spectrum localLe = light->sampleDirect(dRec, rng.randomFloat2());
		bRec.wo = bRec.map.sys.toLocal(dRec.d);
		if(V(ev.p, dRec.p))
		{
			if(i > 1)
				localLe *= P.EyePath[i - 2].cumulative;
			const float bsdfPdf = ev.r2.getMat().bsdf.pdf(bRec);
			const float misWeight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
			L += localLe * ev.r2.getMat().bsdf.f(bRec) * pathWeight(i, 0);
		}

		//case iv
		for(unsigned int j = 1; j < P.t + 1; j++)
		{
			const PathVertex& lv = P.LightPath[j - 1];
			if(V(ev.p, lv.p))
				L += Le * evalPath(P, i, j, rng) * pathWeight(i, j);
		}
	}
	
	for(unsigned int j = 1; j < P.t + 1; j++)
	{
		const PathVertex& lv = P.LightPath[j - 1];
		if(!lv.r2.hasHit())
			break;//urgs wtf?
		
		lv.r2.getBsdfSample(Ray(lv.p, -1.0f * lv.wi), rng, &bRec);
		DirectSamplingRecord dRec(lv.p, bRec.map.sys.n);
		Spectrum localLe = Le * g_SceneData.sampleSensorDirect(dRec, rng.randomFloat2());
		if(V(dRec.p, lv.p))
		{
			if(j > 1)
				localLe *= P.LightPath[j - 2].cumulative;
			bRec.wo = bRec.map.sys.toLocal(dRec.d);
			localLe *= lv.r2.getMat().bsdf.f(bRec);
			if(dRec.uv.x >= 0 && dRec.uv.x < w && dRec.uv.y >= 0 && dRec.uv.y < h)
				g_Image.Splat((int)dRec.uv.x, (int)dRec.uv.y, localLe * pathWeight(0, j) / float(P.t));
		}
	}

	g_Image.AddSample(x, y, L);
}

__global__ void pathKernel(unsigned int w, unsigned int h, int xoff, int yoff, e_Image g_Image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + xoff, y = blockIdx.y * blockDim.y + threadIdx.y + yoff;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
		//Li(g_Image, rng, x, y);
		BDPT(x,y,w,h,g_Image, rng);
	g_RNGData(rng);
}

static e_Image* gI;
void k_BDPT::DoRender(e_Image* I)
{
	gI = I;
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

void k_BDPT::Debug(int2 pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//Li(*gI, g_RNGData(), pixel.x, pixel.y);
	BDPT(pixel.x,pixel.y,w,h,*gI, g_RNGData());
}