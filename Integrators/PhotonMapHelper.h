#pragma once

#include <Kernel/Tracer.h>
#include <Engine/SpatialStructures/Grid/HashGrid.h>
#include <Math/Compression.h>
#include <Base/FileStream.h>
#include <Base/CudaMemoryManager.h>
#include <Math/Kernel.h>

namespace CudaTracerLib {

#define PTDM(X) X(Constant) X(kNN)
ENUMIZE(PPM_Radius_Type, PTDM)
#undef PTDM

#define ALPHA (2.0f / 3.0f)

CUDA_FUNC_IN float getCurrentRadius(float initial_r, unsigned int iteration, float exp)
{
	return initial_r * math::pow((float)iteration, (ALPHA - 1) / exp);
}

template<int DIM> CUDA_FUNC float _density_to_rad_(float kToFind, float density)
{
	auto x = kToFind / (c_d<DIM>() * density);
	if (DIM == 1)
		return x;
	else if (DIM == 2)
		return math::sqrt(x);
	else return math::pow(x, 1.0f / DIM);
}

//computes the radius from the density for the specified dimension after numIter iterations, the min max radii are specified at the specific iteration
template<int DIM_DENS, int DIM_SHRINK = DIM_DENS, bool SHRINK_RAD_ITER = true> CUDA_FUNC float density_to_rad(float kToFind, float density, float r_min, float r_max, unsigned int numIter)
{
	if (density == 0.0f)
		return r_max;
	float r = math::IsNaN(density) || isinf(density) || density == 0.0f ? (r_min + r_max) / 2.0f : _density_to_rad_<DIM_DENS>(kToFind, density);
	float r_it = SHRINK_RAD_ITER ? getCurrentRadius(r, numIter, DIM_SHRINK) : r;
	return math::clamp(r_it, r_min, r_max);
}

typedef KernelWrapper<PerlinKernel> Kernel;

struct PPPMPhoton
{
private:
	RGBE L;
	unsigned short Wi;
	unsigned short Nor;
	unsigned int flag_pos;
public:
	CUDA_FUNC_IN PPPMPhoton(){}
	CUDA_FUNC_IN PPPMPhoton(const Spectrum& l, const NormalizedT<Vec3f>& wi, const NormalizedT<Vec3f>& n)
	{
		Nor = NormalizedFloat3ToUchar2(n);
		L = (l).toRGBE();
		Wi = NormalizedFloat3ToUchar2(wi);
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> getNormal() const
	{
		return Uchar2ToNormalizedFloat3(Nor);
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> getWi() const
	{
		return Uchar2ToNormalizedFloat3(Wi);
	}
	CUDA_FUNC_IN Spectrum getL() const
	{
		Spectrum s;
		s.fromRGBE(L);
		return s;
	}
	template<typename HASH> CUDA_FUNC_IN void setPos(const HASH& hash, const Vec3u& i, const Vec3f& p)
	{
		flag_pos = (flag_pos & 0xc0000000) | hash.EncodePos(p, i);
	}
	template<typename HASH> CUDA_FUNC_IN Vec3f getPos(const HASH& hash, const Vec3u& i) const
	{
		return hash.DecodePos(flag_pos & 0x3fffffff, i);
	}
	CUDA_FUNC_IN unsigned short& accessNormalStorage()
	{
		return Nor;
	}
	CUDA_FUNC_IN bool getFlag() const
	{
		return (flag_pos >> 31) != 0;
	}
	CUDA_FUNC_IN void setFlag()
	{
		flag_pos |= 0x80000000;
	}
};

}
