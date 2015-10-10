#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

CUDA_CONST e_SpatialLinkedMap<k_pPpmPhoton> g_SurfMap;
CUDA_CONST unsigned int g_NumPhotonEmitted2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax4(sizeof(k_PointStorage), sizeof(k_BeamGrid), sizeof(k_BeamBeamGrid), sizeof(k_BeamBVHStorage))];

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat)
{
	Spectrum Lp = Spectrum(0.0f);
	const float r2 = a_rSurfaceUNUSED * a_rSurfaceUNUSED;
	Frame sys = Frame(bRec.dg.n);
	sys.t *= a_rSurfaceUNUSED;
	sys.s *= a_rSurfaceUNUSED;
	Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
	g_SurfMap.ForAll(low, high, [&](unsigned int p_idx, k_pPpmPhoton& ph)
	{
		float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
		if (dist2 < r2)//&& dot(n, bRec.dg.sys.n) > 0.8f
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			Spectrum bsdfFactor = mat->bsdf.f(bRec);
			float ke = k_tr(a_rSurfaceUNUSED, math::sqrt(dist2));
			Lp += PI * ke * ph.getL() * bsdfFactor / Frame::cosTheta(bRec.wo);
		}
	});
	return Lp / g_NumPhotonEmitted2;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const e_KernelMaterial* mat, k_AdaptiveStruct& A, int x, int y,
	const Spectrum& importance, int a_PassIndex)
{
	//Adaptive Progressive Photon Mapping Implementation
	k_AdaptiveEntry ent = A(x,y);
	float r2 = ent.r * ent.r, maxr = max(ent.r, ent.rd), rd2 = ent.rd * ent.rd, rd = ent.rd, r = ent.r;
	Frame sys = bRec.dg.sys;
	sys.t *= maxr;
	sys.s *= maxr;
	sys.n *= maxr;
	Vec3f ur = bRec.dg.sys.t * rd, vr = bRec.dg.sys.s * rd;
	Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
	Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
	Spectrum Lp = 0.0f;
	g_SurfMap.ForAll(low, high, [&](unsigned int p_idx, k_pPpmPhoton& ph)
	{
		float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
		if (dot(ph.getNormal(), bRec.dg.sys.n) > 0.95f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			Spectrum bsdfFactor = mat->bsdf.f(bRec);
			float psi = Spectrum(importance * bsdfFactor * ph.getL()).getLuminance();
			if (dist2 < rd2)
			{
				const Vec3f e_l = bRec.dg.P - ph.Pos;
				float cc = k_tr(rd, e_l);
				float laplu = k_tr(rd, e_l + ur) + k_tr(rd, e_l - ur) - 2.0f * cc,
					laplv = k_tr(rd, e_l + vr) + k_tr(rd, e_l - vr) - 2.0f * cc,
					lapl = psi / rd2 * (laplu + laplv);
				ent.I += lapl;
				ent.I2 += lapl * lapl;
			}
			if (dist2 < r2)
			{
				float kri = k_tr(r, math::sqrt(dist2));
				Lp += PI * kri * ph.getL() * bsdfFactor / Frame::cosTheta(bRec.wo);
				ent.psi += psi;
				ent.psi2 += psi * psi;
				ent.pl += kri;
			}
		}
	});
	float NJ = a_PassIndex * g_NumPhotonEmitted2;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_I = ent.I / NJ;
	float E_pl = ent.pl / a_PassIndex;

	if (VAR_Lapl)
	{
		ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(a_PassIndex, -1.0f / 8.0f);
		ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);
	}

	if (VAR_Lapl && VAR_Phi)
	{
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float ta = (2.0f * math::sqrt(VAR_Phi)) / (PI * float(g_NumPhotonEmitted2) * E_pl * k_22 * E_I * E_I);
		ent.r = math::pow(ta, 1.0f / 6.0f) * math::pow(a_PassIndex, -1.0f / 6.0f);
		ent.r = math::clamp(ent.r, A.r_min, A.r_max);
	}
	A(x,y) = ent;
	//return 0.0f;
	return Lp / float(g_NumPhotonEmitted2);
	//return L_Surface(bRec, ent.r, mat, photonMap, photonMap.m_sSurfaceMap);
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage img, bool DIRECT)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2i pixel = k_TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		Vec2f screenPos = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		Ray r, rX, rY;
		Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2()), importance = throughput;
		TraceResult r2;
		r2.Init();
		int depth = -1;
		Spectrum L(0.0f);
		while (k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
		{
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			if (depth == 0)
				dg.computePartials(r, rX, rY);
			if (g_SceneData.m_sVolume.HasVolumes())
			{
				float tmin, tmax;
				if (g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
				{
					Spectrum Tr(1.0f);
					L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, r, tmin, tmax, VolHelper<true>(), Tr);
					throughput = throughput * Tr;
				}
			}
			if (DIRECT)
				L += throughput * UniformSampleOneLight(bRec, r2.getMat(), rng);
			L += throughput * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);//either it's the first bounce or it's a specular reflection
			const e_KernelBSSRDF* bssrdf;
			if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = k_TraceRay(rTrans);
				Spectrum Tr;
				e_VolumeRegion reg;
				e_PhaseFunction func;
				func.SetData(e_IsotropicPhaseFunction());
				reg.SetData(e_HomogeneousVolumeDensity(func, float4x4::Translate(Vec3f(0.5f)) % float4x4::Scale(Vec3f(1e5f)), bssrdf->sig_a, bssrdf->sigp_s, Spectrum(0.0f)));
				const float a = 1e10f;
				reg.As<e_HomogeneousVolumeDensity>()->WorldToVolume = float4x4::Translate(Vec3f(0.5f)) % float4x4::Scale(Vec3f(0.5f/a));
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, rTrans, 0, r3.m_fDist, VolHelper<false>(&reg), Tr);
				//throughput = throughput * Tr;//break;
			}
			bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
				hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasSmooth)
			{
				L += throughput * (hasGlossy ? 0.5f : 1) * L_Surface(bRec, a_rSurface, &r2.getMat());
				//L += throughput * L_Surface(bRec, a_rSurfaceUNUSED, &r2.getMat(), a_AdpEntries, x, y, importance, a_PassIndex, photonMap);
				if (!hasSpecGlossy)
					break;
			}
			if (hasSpecGlossy)
			{
				bRec.sampledType = 0;
				bRec.typeMask = EDelta | EGlossy;
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (!bRec.sampledType)
					break;
				throughput = throughput * t_f * (hasGlossy ? 0.5f : 1);
				importance = t_f;
				r = Ray(bRec.dg.P, bRec.getOutgoing());
				r2.Init();
			}
			else break;
		}

		float tmin, tmax;
		if (!r2.hasHit() && g_SceneData.m_sVolume.HasVolumes() && g_SceneData.m_sVolume.IntersectP(r, 0, r2.m_fDist, &tmin, &tmax))
		{
			Spectrum Tr(1);
			L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, r, tmin, tmax, VolHelper<true>(), Tr);
			L += Tr * throughput * g_SceneData.EvalEnvironment(r);
		}
		img.Add(screenPos.x, screenPos.y, L);
		//Spectrum qs;
		//float t = A.E[y * w + x].r / a_rSurfaceUNUSED;
		//t = (A.E[y * w + x].r - A.r_min) / (A.r_max - A.r_min);
		//qs.fromHSL(1.0f / 3.0f - t / 3.0f, 1, 0.5f);
		//g_Image.AddSample(screenPos.x, screenPos.y, qs);
		//auto& ent = a_AdpEntries(x, y);
		/*float NJ = a_PassIndex * photonMap.m_uPhotonNumEmitted;
		float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
		float VAR_Phi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
		float E_I = ent.I / NJ;
		float E_pl = ent.pl / NJ;
		g_Image.AddSample(screenPos.x, screenPos.y, Spectrum(VAR_Phi*100));*/
		//float v = (ent.I2 - (ent.I * ent.I) / a_PassIndex) / a_PassIndex * 1e-10f;
		//float v = (ent.rd - a_AdpEntries.r_min) / (a_AdpEntries.r_max - a_AdpEntries.r_min);
		//img.Add(x, y, Spectrum(math::abs(v)));
	}
	g_RNGData(rng);
}

void k_sPpmTracer::RenderBlock(e_Image* I, int x, int y, int blockW, int blockH)
{
	float radius2 = getCurrentRadius(2);
	float radius3 = getCurrentRadius(3);

	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmitted2, &m_uPhotonEmittedPass, sizeof(m_uPhotonEmittedPass)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	k_AdaptiveStruct A(r_min, r_max, m_pEntries, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	k_BlockSampleImage img = m_pBlockSampler->getBlockImage();
	if (dynamic_cast<k_PointStorage*>(m_pVolumeEstimator))
		k_EyePass<k_PointStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect);
	else if (dynamic_cast<k_BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<k_BeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect);
	else if (dynamic_cast<k_BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<k_BeamBeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect);
	else if (dynamic_cast<k_BeamBVHStorage*>(m_pVolumeEstimator))
		k_EyePass<k_BeamBVHStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect);

	ThrowCudaErrors(cudaThreadSynchronize());
}

__global__ void k_StartPass(int w, int h, float r, float rd, k_AdaptiveEntry* E)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * w + x;
	if(x < w && y < h)
	{
		E[i].r = r;
		E[i].rd = rd;
		E[i].psi = E[i].psi2 = E[i].I = E[i].I2 = E[i].pl = 0.0f;
	}
}

void k_sPpmTracer::doStartPass(float r, float rd)
{
	int p = 32;
	if (m_pEntries)
		k_StartPass<<<dim3(w / p + 1, h / p + 1, 1), dim3(p,p,1)>>>(w, h, r, rd, m_pEntries);
}