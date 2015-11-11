#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>

namespace CudaTracerLib {

CUDA_CONST SpatialLinkedMap<PPPMPhoton> g_SurfMap;
CUDA_CONST unsigned int g_NumPhotonEmitted2;
CUDA_CONST CUDA_ALIGN(16) unsigned char g_VolEstimator2[Dmax4(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid), sizeof(BeamBVHStorage))];

template<bool USE_GLOBAL> Spectrum BeamBeamGrid::L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
{
	Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
	TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
		{
			Beam B = m_pDeviceBeams[beam_idx];
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

				L_n += camera_sc * p * B.Phi * (-photon_tau).exp() * (-camera_tau).exp() / (m_fCurrentRadiusVol * m_uNumEmitted);
			}
		});
		float localDist = cellEndT - rayT;
		Spectrum tauD = vol.tau(r, rayT, cellEndT);
		Tau += tauD;
		L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
	});
	Tr = (-Tau).exp();
	return L_n;
}

template<bool USE_GLOBAL> Spectrum BeamBVHStorage::L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const
{
	Spectrum L_n = Spectrum(0.0f);
	iterateBeams(Ray(r(tmin), r.direction), tmax - tmin, [&](const Beam& b)
	{
		float t1, t2;
		if (skew_lines(r, Ray(b.pos, b.dir), t1, t2) < a_r)
		{
			//float sin_theta = math::sin(math::safe_acos(dot(-b.dir.normalized(), r.direction.normalized())));
			float sin_theta = math::safe_sqrt(1 - math::sqr(dot(-b.dir, r.direction)));
			Spectrum photon_tau = vol.tau(Ray(b.pos, b.dir), 0, t2);
			Spectrum camera_tau = vol.tau(r, 0, t1);
			Spectrum camera_sc = vol.sigma_s(r(t1), r.direction);
			float p = vol.p(r(t1), r.direction, b.dir, rng);
			L_n += camera_sc * p * b.Phi * (-photon_tau).exp() * (-camera_tau).exp() / sin_theta;
		}
	});
	Tr = (-vol.tau(r, tmin, tmax)).exp();
	return L_n / (m_fCurrentRadiusVol * m_uNumEmitted);
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float r, const Material* mat)
{
	Spectrum Lp = Spectrum(0.0f);
	Vec3f a = r*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, c = r*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	g_SurfMap.ForAll(min(a, b, c, d), max(a, b, c, d), [&](unsigned int p_idx, PPPMPhoton& ph)
	{
		float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
		if (dist2 < r * r && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			Spectrum bsdfFactor = mat->bsdf.f(bRec);
			float ke = k_tr(r, math::sqrt(dist2));
			Lp += ke * ph.getL() / g_NumPhotonEmitted2 * bsdfFactor;
		}
	});
	return Lp;
}

CUDA_FUNC_IN Spectrum L_Surface(BSDFSamplingRecord& bRec, float a_rSurfaceUNUSED, const Material* mat, k_AdaptiveStruct& A, int x, int y,
	const Spectrum& importance, int a_PassIndex, BlockSampleImage& img)
{
	//Adaptive Progressive Photon Mapping Implementation
	k_AdaptiveEntry ent = A(x, y);
	Vec3f ur = bRec.dg.sys.t * ent.rd, vr = bRec.dg.sys.s * ent.rd;
	float r = max(2 * ent.rd, ent.r);
	Vec3f a = r*(-bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, b = r*(bRec.dg.sys.t - bRec.dg.sys.s) + bRec.dg.P, c = r*(-bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P, d = r*(bRec.dg.sys.t + bRec.dg.sys.s) + bRec.dg.P;
	Spectrum Lp = 0.0f;
	g_SurfMap.ForAll<200>(min(a, b, c, d), max(a, b, c, d), [&](unsigned int p_idx, PPPMPhoton& ph)
	{
		float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
		//if (dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
		{
			bRec.wo = bRec.dg.toLocal(ph.getWi());
			Spectrum bsdfFactor = mat->bsdf.f(bRec);
			float psi = Spectrum(importance * bsdfFactor * ph.getL() / float(g_NumPhotonEmitted2)).getLuminance();//this 1/J has nothing to do with E[X], it is scaling for radiance distribution
			const Vec3f e_l = bRec.dg.P - ph.Pos;
			float k_rd = k_tr(ent.rd, e_l);
			float laplu = k_tr(ent.rd, e_l + ur) + k_tr(ent.rd, e_l - ur) - 2 * k_rd,
				laplv = k_tr(ent.rd, e_l + vr) + k_tr(ent.rd, e_l - vr) - 2 * k_rd,
				lapl = psi / (ent.rd * ent.rd) * (laplu + laplv);
			ent.I += laplu / (ent.rd * ent.rd);
			ent.I2 += lapl * lapl;
			ent.n1++;
			if (dist2 < ent.r * ent.r)
			{
				ent.n2++;
				float kri = k_tr(ent.r, math::sqrt(dist2));
				Lp += kri * ph.getL() / float(g_NumPhotonEmitted2) * bsdfFactor;
				ent.psi += psi;
				ent.psi2 += psi * psi;
				ent.pl += kri;
			}
		}
	});
	float NJ = a_PassIndex * g_NumPhotonEmitted2;
	float VAR_Lapl = ent.I2 / NJ - ent.I / NJ * ent.I / NJ;
	float E_I = ent.I / NJ;
	float VAR_Psi = ent.psi2 / NJ - ent.psi / NJ * ent.psi / NJ;
	float E_pl = ent.pl / NJ;

	//ent.rd = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(a_PassIndex, -1.0f / 8.0f);
	//ent.rd = math::clamp(ent.rd, A.r_min, A.r_max);

	float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
	float ta = (2.0f * math::sqrt(VAR_Psi)) / (PI * float(g_NumPhotonEmitted2) * E_pl * k_22 * E_I * E_I);
	float q = math::pow(ta, 1.0f / 6.0f) * math::pow(a_PassIndex, -1.0f / 6.0f);
	float p = 1.9635f * math::sqrt(VAR_Lapl) * math::pow(a_PassIndex, -1.0f / 8.0f);
	if (x == 488 && y == 654)
		printf("Var[Psi] = %5.5e, Var[Lapl] = %5.5e, E[pl] = %5.5e, E[I] = %5.5e, E[Psi] = %5.5e\n", VAR_Psi, VAR_Lapl, E_pl, E_I, ent.psi / NJ);
	//ent.r = math::clamp(q, A.r_min, A.r_max);

	Spectrum qs;
	//float t = E_pl / (PPM_MaxRecursion / (PI * 4 * math::sqr(g_SceneData.m_sBox.Size().length() / 2)));
	//float t = (ent.r - A.r_min) / (A.r_max - A.r_min);
	//float t = math::abs(E_I) * 1e-3f / (g_SceneData.m_sBox.Size().length() / 2);
	float t = ent.n2 / 500;
	qs.fromHSL(2.0f / 3.0f * (1 - math::clamp01(t)), 1, 0.5f);//0 -> 1 : Dark Blue -> Light Blue -> Green -> Yellow -> Red
	img.Add(x, y, qs);

	A(x, y) = ent;
	return Lp;
}

template<typename VolEstimator>  __global__ void k_EyePass(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct a_AdpEntries, BlockSampleImage img, bool DIRECT, bool USE_RI)
{
	CudaRNG rng = g_RNGData();
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	if (pixel.x < w && pixel.y < h)
	{
		Vec2f screenPos = Vec2f(pixel.x, pixel.y) + rng.randomFloat2();
		Ray r, rX, rY;
		Spectrum throughput = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2()), importance = throughput;
		TraceResult r2;
		r2.Init();
		int depth = -1;
		Spectrum L(0.0f);
		while (Traceray(r.direction, r.origin, &r2) && depth++ < 5)
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
			const VolumeRegion* bssrdf;
			if (r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				Spectrum t_f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				bRec.wo.z *= -1.0f;
				Ray rTrans = Ray(bRec.dg.P, bRec.getOutgoing());
				TraceResult r3 = Traceray(rTrans);
				Spectrum Tr;
				L += throughput * ((VolEstimator*)g_VolEstimator2)->L_Volume(a_rVolume, rng, rTrans, 0, r3.m_fDist, VolHelper<false>(bssrdf), Tr);
				//throughput = throughput * Tr;//break;
			}
			bool hasSmooth = r2.getMat().bsdf.hasComponent(ESmooth),
				hasSpecGlossy = r2.getMat().bsdf.hasComponent(EDelta | EGlossy),
				hasGlossy = r2.getMat().bsdf.hasComponent(EGlossy);
			if (hasSmooth)
			{
				float r = math::clamp(getCurrentRadius(a_AdpEntries(pixel.x, pixel.y).r, a_PassIndex, 2), a_AdpEntries.r_min, a_AdpEntries.r_max);
				L += throughput * (hasGlossy ? 0.5f : 1) * L_Surface(bRec, USE_RI ? r : a_rSurface, &r2.getMat());
				//L += throughput * L_Surface(bRec, a_rSurface, &r2.getMat(), a_AdpEntries, pixel.x, pixel.y, importance, a_PassIndex, img);
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
	}
	g_RNGData(rng);
}

__global__ void k_EyePass2(Vec2i off, int w, int h, float a_PassIndex, float a_rSurface, float a_rVolume, k_AdaptiveStruct A, BlockSampleImage img, float rMax, float rMin)
{
	Vec2i pixel = TracerBase::getPixelPos(off.x, off.y);
	/*Ray r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	BeamBeamGrid* grid = (BeamBeamGrid*)g_VolEstimator2;
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
	BeamBeamGrid* grid = (BeamBeamGrid*)g_VolEstimator2;
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

void PPPMTracer::Debug(Image* I, const Vec2i& pixel)
{
	/*Beam b;
	BeamBeamGrid& grid = *((BeamBeamGrid*)m_pVolumeEstimator);
	cudaMemcpy(&b, grid.m_pDeviceBeams + 1, sizeof(Beam), cudaMemcpyDeviceToHost);
	std::cout << Ray(b.pos, b.dir) << "\n";
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));
	debugEye << <1, 1 >> >(pixel.x, pixel.y);*/
}

void PPPMTracer::RenderBlock(Image* I, int x, int y, int blockW, int blockH)
{
	float radius2 = getCurrentRadius(2);
	float radius3 = getCurrentRadius(3);

	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NumPhotonEmitted2, &m_uPhotonEmittedPass, sizeof(m_uPhotonEmittedPass)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator2, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	k_AdaptiveStruct A(r_min, r_max, m_pEntries, w, m_uPassesDone);
	Vec2i off = Vec2i(x, y);
	BlockSampleImage img = m_pBlockSampler->getBlockImage();
	if (dynamic_cast<PointStorage*>(m_pVolumeEstimator))
		k_EyePass<PointStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
		k_EyePass<BeamBeamGrid> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
	else if (dynamic_cast<BeamBVHStorage*>(m_pVolumeEstimator))
		k_EyePass<BeamBVHStorage> << <numBlocks, threadsPerBlock >> >(off, w, h, m_uPassesDone, radius2, radius3, A, img, m_bDirect, m_bFinalGather);
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
		e.n1 = e.n2 = 0;

		//initial per pixel rad estimate
		CudaRNG rng = g_RNGData();
		/*DifferentialGeometry dg;
		BSDFSamplingRecord bRec(dg);
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = Traceray(r);
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
		g_SurfMap.ForAll(low, high, [&](unsigned int p_idx, const PPPMPhoton& ph)
		{
		float dist2 = distanceSquared(ph.Pos, bRec.dg.P);
		if (dist2 < search_rad * search_rad && dot(ph.getNormal(), bRec.dg.sys.n) > 0.9f)
		k_found++;
		});
		#endif
		float density = max(k_found, 1) / (PI * search_rad * search_rad);
		e.r = math::sqrt(k_toFind / (PI * density));
		}
		else e.r = r_1;*/
		atomicMin(&g_MinRad, floatToOrderedInt(e.r));
		atomicMax(&g_MaxRad, floatToOrderedInt(e.r));
		g_RNGData(rng);
	}
}

void PPPMTracer::doPerPixelRadiusEstimation()
{
	int a = floatToOrderedInt(FLT_MAX), b = floatToOrderedInt(0);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MaxRad, &b, sizeof(b)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_MinRad, &a, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	int p = 32;
	if (m_pEntries)
		k_PerPixelRadiusEst << <dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1) >> >(w, h, r_max * 0.1f, m_fInitialRadius, k_AdaptiveStruct(r_min, r_max, m_pEntries, w, m_uPassesDone), k_Intial);
	ThrowCudaErrors(cudaMemcpyFromSymbol(&a, g_MinRad, sizeof(a)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&b, g_MaxRad, sizeof(b)));
	m_fIntitalRadMin = orderedIntToFloat(a);
	m_fIntitalRadMax = orderedIntToFloat(b);
	std::cout << "m_fIntitalRadMin = " << m_fIntitalRadMin << ", m_fIntitalRadMax = " << m_fIntitalRadMax << "\n";
}

}