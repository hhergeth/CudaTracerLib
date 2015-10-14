#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include <Math/half.h>

CUDA_CONST e_SpatialLinkedMap<k_pPpmPhoton> g_SurfMap;
CUDA_CONST unsigned int g_NumPhotonEmitted2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax4(sizeof(k_PointStorage), sizeof(k_BeamGrid), sizeof(k_BeamBeamGrid), sizeof(k_BeamBVHStorage))];

template<bool USE_GLOBAL> Spectrum k_BeamBeamGrid::L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
		{
			k_Beam B = m_pDeviceBeams[beam_idx];
			float t1, t2;
			bool sk = skew_lines(r, Ray(B.pos, B.dir), t1, t2);
			//Vec3f p_b = B.pos + t2 * B.dir, p_c = r.origin + t1 * r.direction;m_sStorage.hashMap.Transform(p_b) == cell_pos && m_sStorage.hashMap.Transform(p_c) == cell_pos && 
			if (sk && t1 > 0 && t2 > 0 && t2 < B.t && t1 < cellEndT)
			{
				float sin_theta = math::sin(math::safe_acos(dot(-B.dir.normalized(), r.direction.normalized())));
				Spectrum photon_tau = vol.tau(Ray(B.pos, B.dir), 0, t2);
				Spectrum camera_tau = vol.tau(r, 0, t1);
				Spectrum camera_sc = vol.sigma_s(r(t1), r.direction);
				float p = vol.p(r(t1), r.direction, B.dir, rng);
				L_n += camera_sc * p * B.Phi * (-photon_tau).exp() * (-camera_tau).exp() / sin_theta;
			}
		});
		float localDist = cellEndT - rayT;
		Spectrum tauD = vol.tau(r, rayT, cellEndT);
		Tau += tauD;
		L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
	});
	Tr = (-Tau).exp();
	return L_n / (a_r * m_uNumEmitted);
}

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
		if (dist2 < r2 && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
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

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage img, bool DIRECT, bool USE_RI)
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
			const e_VolumeRegion* bssrdf;
			if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = k_TraceRay(rTrans);
				Spectrum Tr;
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr);
				//throughput = throughput * Tr;//break;
			}
			bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
				hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasSmooth)
			{
				float r = math::clamp(getCurrentRadius(A(pixel.x, pixel.y).r, a_PassIndex, 2), A.r_min, A.r_max);
				L += throughput * (hasGlossy ? 0.5f : 1) * L_Surface(bRec, USE_RI ? a_rSurface : r, &r2.getMat());
				//L += throughput * L_Surface(bRec, a_rSurface, &r2.getMat(), a_AdpEntries, x, y, importance, a_PassIndex, photonMap);
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

__global__ void k_EyePass2(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct A, k_BlockSampleImage img, float rMax, float rMin)
{
	Vec2i pixel = k_TracerBase::getPixelPos(off.x, off.y);
	/*Ray r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	k_BeamBeamGrid* grid = (k_BeamBeamGrid*)g_VolEstimator2;
	int n = 0;
#ifdef ISCUDA
	TraverseGrid(r, grid->m_sStorage.hashMap, 0, FLT_MAX, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		grid->m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
		{
			n += beam_idx == 1234;
			//if(pixel.x == 200 && pixel.y == 200)
			//	printf("(%u, %d), ", ABC, beam_idx);
		});
	});
#endif
	img.Add(pixel.x, pixel.y, Spectrum(n!=0));*/
	float rq = (getCurrentRadius(A(pixel.x, pixel.y).r, a_PassIndex, 2) - a_rSurface) / getCurrentRadius(rMax, a_PassIndex, 2);
	img.Add(pixel.x, pixel.y, Spectrum(rq > 0 ? rq : 0, rq < 0 ? -rq : 0, 0));
	//float ab = getCurrentRadius(A(pixel.x, pixel.y).r, a_PassIndex, 2) < a_rSurface;
	//img.Add(pixel.x, pixel.y, Spectrum(ab));
}

__global__ void debugEye(int x, int y)
{
	Ray r = g_SceneData.GenerateSensorRay(x, y);
	k_BeamBeamGrid* grid = (k_BeamBeamGrid*)g_VolEstimator2;
#ifdef ISCUDA
	TraverseGrid(r, grid->m_sStorage.hashMap, 0, FLT_MAX, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		grid->m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
		{
			printf("(%u, %d), ", ABC, beam_idx);
		});
	});
#endif
}

void k_sPpmTracer::Debug(e_Image* I, const Vec2i& pixel)
{
	/*k_Beam b;
	k_BeamBeamGrid& grid = *((k_BeamBeamGrid*)m_pVolumeEstimator);
	cudaMemcpy(&b, grid.m_pDeviceBeams + 1, sizeof(k_Beam), cudaMemcpyDeviceToHost);
	std::cout << Ray(b.pos, b.dir) << "\n";
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));
	debugEye << <1, 1 >> >(pixel.x, pixel.y);*/
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
		k_EyePass<k_PointStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<k_BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<k_BeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<k_BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<k_BeamBeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<k_BeamBVHStorage*>(m_pVolumeEstimator))
		k_EyePass<k_BeamBVHStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	//k_EyePass2 << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_fIntitalRadMin, m_fIntitalRadMax);

	ThrowCudaErrors(cudaThreadSynchronize());
}

CUDA_DEVICE int g_MaxRad, g_MinRad;
CUDA_FUNC_IN int floatToOrderedInt(float floatVal) {
	int intVal = float_as_int_(floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}
CUDA_FUNC_IN float orderedIntToFloat(int intVal) {
	return int_as_float_((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__global__ void k_PerPixelRadiusEst(int w, int h, float r_max, float r_1, k_AdaptiveStruct adpt, int k_toFind)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h)
	{
		auto& e = adpt(x, y);
		//adaptive progressive intit
		e.r = r_max;
		e.rd = r_max;
		e.psi = e.psi2 = e.I = e.I2 = e.pl = 0.0f;

		//initial per pixel rad estimate
		CudaRNG rng = g_RNGData();
		DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = k_TraceRay(r);
		if (r2.hasHit())
		{
			const float search_rad = r_1;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			Frame sys = Frame(bRec.dg.n);
			sys.t *= search_rad;
			sys.s *= search_rad;
			Vec3f a = -1.0f * sys.t - sys.s, b = sys.t - sys.s, c = -1.0f * sys.t + sys.s, d = sys.t + sys.s;
			Vec3f low = min(min(a, b), min(c, d)) + bRec.dg.P, high = max(max(a, b), max(c, d)) + bRec.dg.P;
			int k_found = 0;
#ifdef ISCUDA
			g_SurfMap.ForAll(low, high, [&](unsigned int p_idx, const k_pPpmPhoton& ph)
			{
				float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
				if (dist2 < search_rad * search_rad && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
					k_found++;
			});
#endif
			float density = max(k_found, 1) / (PI * search_rad * search_rad);
			e.r = math::sqrt(k_toFind / (PI * density));
			atomicMin(&g_MinRad, floatToOrderedInt(e.r));
			atomicMax(&g_MaxRad, floatToOrderedInt(e.r));
		}
		else e.r = r_1;
		g_RNGData(rng);
	}
}

void k_sPpmTracer::doPerPixelRadiusEstimation()
{
	int a = floatToOrderedInt(FLT_MAX), b = floatToOrderedInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxRad, &b, sizeof(b)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MinRad, &a, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	int p = 32;
	if (m_pEntries)
		k_PerPixelRadiusEst << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(w, h, r_max, m_fInitialRadius, k_AdaptiveStruct(r_min, r_max, m_pEntries, w, m_uPassesDone), k_Intial);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
	std::cout << "m_fIntitalRadMin = " << m_fIntitalRadMin << ", m_fIntitalRadMax = " << m_fIntitalRadMax << "\n";
}