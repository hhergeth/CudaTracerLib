#pragma once

#include <Kernel/Tracer.h>
#include <Engine/Grid.h>
#include <Math/Compression.h>
#include <Base/FileStream.h>
#include <CudaMemoryManager.h>
#include <Math/Kernel.h>

namespace CudaTracerLib {

#define ALPHA (2.0f / 3.0f)

CUDA_FUNC_IN float getCurrentRadius(float initial_r, unsigned int iteration, float exp)
{
	return initial_r * math::pow((float)iteration, (ALPHA - 1) / exp);
}

typedef KernelWrapper<PerlinKernel> Kernel;

enum PhotonType
{
	pt_Diffuse = 0,
	pt_Caustic = 1,
};

struct PPPMPhoton
{
private:
	RGBE L;
	unsigned short Wi;
	unsigned short Nor;
	unsigned int flag_type_pos;
public:
	CUDA_FUNC_IN PPPMPhoton(){}
	CUDA_FUNC_IN PPPMPhoton(const Spectrum& l, const NormalizedT<Vec3f>& wi, const NormalizedT<Vec3f>& n, PhotonType type)
	{
		Nor = NormalizedFloat3ToUchar2(n);
		L = (l).toRGBE();
		Wi = NormalizedFloat3ToUchar2(wi);
		flag_type_pos = ((unsigned char)type << 30);
	}
	CUDA_FUNC_IN PhotonType getType() const
	{
		return (PhotonType)((flag_type_pos >> 30) & 1);
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
		flag_type_pos = (flag_type_pos & 0xc0000000) | hash.EncodePos(p, i);
	}
	template<typename HASH> CUDA_FUNC_IN Vec3f getPos(const HASH& hash, const Vec3u& i) const
	{
		return hash.DecodePos(flag_type_pos & 0x3fffffff, i);
	}
	CUDA_FUNC_IN unsigned short& accessNormalStorage()
	{
		return Nor;
	}
	CUDA_FUNC_IN bool getFlag() const
	{
		return (flag_type_pos >> 31) != 0;
	}
	CUDA_FUNC_IN void setFlag()
	{
		flag_type_pos |= 0x80000000;
	}
};

}
