#pragma once

#include <Math/FunctionModel.h>
#include <Math/Frame.h>
#include <Kernel/TracerSettings.h>
#include "../PhotonMapHelper.h"
#include <Engine/Light.h>

namespace CudaTracerLib {

template<int DIM> CUDA_FUNC float _density_to_rad_(float kToFind, float density)
{
	return math::pow(kToFind / (Kernel::c_d<DIM>() * density), 1.0f / DIM);
}

//computes the radius from the density for the specified dimension after numIter iterations, the min max radii are specified at the specific iteration
template<int DIM> CUDA_FUNC float density_to_rad(float kToFind, float density, float r_min, float r_max, unsigned int numIter)
{
	float r = math::IsNaN(density) || isinf(density) || density == 0.0f ? (r_min + r_max) / 2.0f : _density_to_rad_<DIM>(kToFind, density);
	float r_it = getCurrentRadius(r, numIter, DIM);
	return math::clamp(r_it, r_min, r_max);
}
	
template<int N> struct DerivativeCollection
{
	float df_di[N];

	CUDA_FUNC_IN DerivativeCollection(float f = 0.0f)
	{
		for (int i = 0; i < N; i++)
			df_di[i] = f;
	}

#define DCL_OP(OP, RHS, SEL) \
	CUDA_FUNC_IN DerivativeCollection operator OP (RHS) const \
	{ \
		DerivativeCollection r; \
		for(int i = 0; i < N; i++) \
			r.df_di[i] = df_di[i] OP SEL; \
		return r; \
	}

	DCL_OP(+, const DerivativeCollection& rhs, rhs.df_di[i])
	DCL_OP(-, const DerivativeCollection& rhs, rhs.df_di[i])
	DCL_OP(*, const DerivativeCollection& rhs, rhs.df_di[i])
	DCL_OP(*, float rhs, rhs)
	DCL_OP(/ , float rhs, rhs)

#undef  DCL_OP
};

CUDA_FUNC_IN static float Lapl(const DerivativeCollection<1>& der)
{
	return der.df_di[0];
}

CUDA_FUNC_IN static float Lapl(const DerivativeCollection<3>& der, const Frame& sys, const Vec3f& l1, const Vec3f& l2)
{
	auto l = sys.toLocal(l1) + sys.toLocal(l2);
	float sum = 0;
	for (int i = 0; i < 3; i++)
		sum += der.df_di[i] * l[i];
	return sum;
}

template<int DIM_RD, int NUM_DERIV> struct APPM_QueryPointData
{
	VarAccumulator<double> psi;
	DerivativeCollection<NUM_DERIV> Sum_DI;
	VarAccumulator<double> E_DI;
	float Sum_pl;
	float num_psi;

	CUDA_FUNC_IN APPM_QueryPointData()
		: Sum_pl(0), num_psi(0)
	{

	}

	CUDA_FUNC_IN APPM_QueryPointData(const VarAccumulator<double>& psi, const DerivativeCollection<NUM_DERIV>& di, const VarAccumulator<double>& E_di, float pl)
		: psi(psi), Sum_DI(di), E_DI(E_di), Sum_pl(pl), num_psi(0)
	{

	}

	CUDA_FUNC_IN float compute_rd(int iteration, int J, int totalPhotons)
	{
		auto VAR_Lapl = E_DI.Var((float)iteration);
		if (VAR_Lapl <= 0) return -1.0f;
		const float alpha = Kernel::alpha<DIM_RD>(), alpha2 = alpha * alpha, beta = Kernel::beta<DIM_RD>();
		const float L = extract_val<DIM_RD - 1>({3.0f / (8.0f * 1.77245f), 1.0f / (2.0f * PI), 15.0f / (32.0f * 5.56833f)});//0 based indexing
		const float c = math::pow(DIM_RD * beta * 1.0f / alpha2 * 1.0f / L * 1.0f / iteration, 1.0f / 8);//Kaplanyan argues not to use (DIM_RD + 4) because this is the shrinkage rate of r
		return (float)(sqrt(VAR_Lapl) * c);
	}

	template<int DIM_R, typename F> CUDA_FUNC_IN float compute_r(int iteration, int J, int totalPhotons, const F& clb)
	{
		auto VAR_Psi = psi.Var(num_psi);
		float k_2 = Kernel::alpha<DIM_R>(), k_22 = k_2 * k_2;
		auto E_pl = (double)Sum_pl / totalPhotons;
		auto E_DI = clb(Sum_DI / totalPhotons);
		auto nom = (2 * VAR_Psi), denom = (KernelBase<1, 2, 3>::c_d<DIM_R>() * J * E_pl * k_22 * math::sqr(E_DI) * iteration);
		if (nom <= 0 || denom <= 0) return -1.0f;
		return (float)pow(nom / denom, 1.0f / 6.0f);
	}

	template<int DIM_R> CUDA_FUNC_IN float compute_r(int iteration, int J, int totalPhotons)
	{
		return compute_r<DIM_R>(iteration, J, totalPhotons, [](auto& d) {return Lapl(d); });
	}

#define DCL_OP(OP) \
	CUDA_FUNC_IN APPM_QueryPointData operator OP (const APPM_QueryPointData& rhs) const \
	{ \
		return APPM_QueryPointData(psi OP rhs.psi, Sum_DI OP rhs.Sum_DI, E_DI OP rhs.E_DI, Sum_pl OP rhs.Sum_pl); \
	}

	DCL_OP(+)
	DCL_OP(-)
	DCL_OP(*)
#undef  DCL_OP

	CUDA_FUNC_IN APPM_QueryPointData operator*(float f) const
	{
		return APPM_QueryPointData(psi * f, Sum_DI * f, E_DI * f, Sum_pl * f);
	}

	CUDA_FUNC_IN static APPM_QueryPointData Zero()
	{
		return APPM_QueryPointData(VarAccumulator<double>(), 0, VarAccumulator<double>(), 0);
	}

	CUDA_FUNC_IN float Sum() const
	{
		return Sum_pl;
	}
};

#define NUM_VOL_MODEL_BINS 8
typedef MergingModel<NUM_VOL_MODEL_BINS, APPM_QueryPointData<3, 3>> VolumeModel;
CUDA_FUNC_IN float model_t(float t, float tmin, float tmax)
{
	tmax = min(tmax, 1e6f);
	return t / (tmax - tmin);
}

struct APPM_PixelData
{
	APPM_QueryPointData<2, 1> m_surfaceData;
	VolumeModel m_volumeModel;
	Frame queryRayFrame;

	CUDA_FUNC_IN APPM_PixelData()
	{

	}

	CUDA_FUNC_IN void Initialize(float surfRadius, const NormalizedT<Vec3f>& camera_dir)
	{
		m_surfaceData = m_surfaceData.Zero();
		queryRayFrame = Frame(camera_dir);
	}
};

#define PTDM(X) X(Constant) X(kNN) X(Adaptive)
ENUMIZE(PPM_Radius_Type, PTDM)
#undef PTDM

struct PrevPhotonIdx
{
	unsigned int val;

	CUDA_FUNC_IN PrevPhotonIdx()
	{

	}

	CUDA_FUNC_IN PrevPhotonIdx(unsigned int idx, bool surf)
	{
		val = surf ? idx : (1u << 31) | idx;
	}

	CUDA_FUNC_IN bool isSurfPhoton() const
	{
		return (val & (1u << 31)) == 0;
	}

	CUDA_FUNC_IN unsigned int idx() const
	{
		return val & ~(1u << 31);
	}
};

}