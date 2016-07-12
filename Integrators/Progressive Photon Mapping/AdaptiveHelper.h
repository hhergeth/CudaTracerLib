#pragma once

#include <Math/FunctionModel.h>
#include <Math/Frame.h>
#include <Kernel/TracerSettings.h>
#include "../PhotonMapHelper.h"

namespace CudaTracerLib {
	
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

template<int DIM_RD PP_COMMA int NUM_DERIV> struct APPM_QueryPointData
{
	float Sum_psi, Sum_psi2;
	DerivativeCollection<NUM_DERIV> Sum_DI;
	float Sum_E_DI, Sum_E_DI2;
	float Sum_pl;
	float r_std;

	CUDA_FUNC_IN APPM_QueryPointData()
	{

	}

	CUDA_FUNC_IN APPM_QueryPointData(float psi, float psi2, const DerivativeCollection<NUM_DERIV>& di, float di1, float di2, float pl, float r)
		: Sum_psi(psi), Sum_psi2(psi2), Sum_DI(di), Sum_E_DI(di1), Sum_E_DI2(di2), Sum_pl(pl), r_std(r)
	{

	}

	CUDA_FUNC_IN float compute_rd(int iteration, int J, int totalPhotons)
	{
		float VAR_Lapl = Sum_E_DI2 / iteration - math::sqr(Sum_E_DI / iteration);
		if (VAR_Lapl <= 0) return -1.0f;
		const float alpha = Kernel::alpha<DIM_RD>(), alpha2 = alpha * alpha, beta = Kernel::beta<DIM_RD>();
		const float L = extract_val<DIM_RD - 1>(3.0f / (8.0f * 1.77245f), 1.0f / (2.0f * PI), 15.0f / (32.0f * 5.56833f));//0 based indexing
		const float c = math::pow(DIM_RD * beta * 1.0f / alpha2 * 1.0f / L * 1.0f / iteration, 1.0f / (DIM_RD + 4));
		return math::sqrt(VAR_Lapl) * c;
	}

	template<int DIM_R, typename F> CUDA_FUNC_IN float compute_r(int iteration, int J, int totalPhotons, const F& clb)
	{
		float VAR_Psi = Sum_psi2 / iteration - math::sqr(Sum_psi / iteration);
		float k_2 = Kernel::alpha<DIM_R>(), k_22 = k_2 * k_2;
		float E_pl = Sum_pl / iteration;//Kaplanyan claims this should be 1 / (N * J)
		float E_DI = clb(Sum_DI / (float)iteration);
		float nom = (2.0f * math::sqrt(VAR_Psi)), denom = (KernelBase<1, 2, 3>::c_d<DIM_R>() * J * E_pl * k_22 * math::sqr(E_DI) * iteration);
		if (nom <= 0 || denom <= 0) return -1.0f;
		return math::pow(nom / denom, 1.0f / 6.0f);
	}

#define DCL_OP(OP) \
	CUDA_FUNC_IN APPM_QueryPointData operator OP (const APPM_QueryPointData& rhs) const \
	{ \
		return APPM_QueryPointData(Sum_psi OP rhs.Sum_psi, Sum_psi2 OP rhs.Sum_psi2, Sum_DI OP rhs.Sum_DI, Sum_E_DI OP rhs.Sum_E_DI, Sum_E_DI2 OP rhs.Sum_E_DI2, Sum_pl OP rhs.Sum_pl, r_std OP rhs.r_std); \
	}

	DCL_OP(+)
	DCL_OP(-)
	DCL_OP(*)

	CUDA_FUNC_IN APPM_QueryPointData operator*(float f) const
	{
		return APPM_QueryPointData(Sum_psi * f, Sum_psi2 * f, Sum_DI * f, Sum_E_DI * f, Sum_E_DI2 * f, Sum_pl * f, r_std * f);
	}

	CUDA_FUNC_IN static APPM_QueryPointData Zero()
	{
		return APPM_QueryPointData(0, 0, 0, 0, 0, 0, 0);
	}

	CUDA_FUNC_IN float Sum() const
	{
		return Sum_pl;//Sum_psi + Sum_psi2 + Sum_DI + Sum_E_DI + Sum_E_DI2 + 
	}

#undef  DCL_OP
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
		m_surfaceData.r_std = surfRadius;
		queryRayFrame = Frame(camera_dir);
	}
};

#define PTDM(X) X(Constant) X(kNN) X(Adaptive)
ENUMIZE(PPM_Radius_Type, PTDM)
#undef PTDM

}