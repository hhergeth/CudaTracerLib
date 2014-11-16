#pragma once

#include <MathTypes.h>
#include "e_DifferentialGeometry.h"

enum EMeasure {
	/// Invalid measure
	EInvalidMeasure = 0,
	/// Solid angle measure
	ESolidAngle = 1,
	/// Length measure
	ELength = 2,
	/// Area measure
	EArea = 3,
	/// Discrete measure
	EDiscrete = 4
};

enum ETransportMode
{
	ERadiance = 0,
	EImportance = 1,
};

enum EBSDFType {
	// =============================================================
	//                      BSDF lobe types
	// =============================================================

	/// 'null' scattering event, i.e. particles do not undergo deflection
	ENull                 = 0x00001,
	/// Ideally diffuse reflection
	EDiffuseReflection    = 0x00002,
	/// Ideally diffuse transmission
	EDiffuseTransmission  = 0x00004,
	/// Glossy reflection
	EGlossyReflection     = 0x00008,
	/// Glossy transmission
	EGlossyTransmission   = 0x00010,
	/// Reflection into a discrete set of directions
	EDeltaReflection      = 0x00020,
	/// Transmission into a discrete set of directions
	EDeltaTransmission    = 0x00040,
	/// Reflection into a 1D space of directions
	EDelta1DReflection    = 0x00080,
	/// Transmission into a 1D space of directions
	EDelta1DTransmission  = 0x00100,

	// =============================================================
	//!                   Other lobe attributes
	// =============================================================
	/// The lobe is not invariant to rotation around the normal
	EAnisotropic          = 0x01000,
	/// The BSDF depends on the UV coordinates
	ESpatiallyVarying     = 0x02000,
	/// Flags non-symmetry (e.g. transmission in dielectric materials)
	ENonSymmetric         = 0x04000,
	/// Supports interactions on the front-facing side
	EFrontSide            = 0x08000,
	/// Supports interactions on the back-facing side
	EBackSide             = 0x10000,
	/// Uses extra random numbers from the supplied sampler instance
	EUsesSampler          = 0x20000,
};

enum ETypeCombinations {
	/// Any reflection component (scattering into discrete, 1D, or 2D set of directions)
	EReflection   = EDiffuseReflection | EDeltaReflection
		| EDelta1DReflection | EGlossyReflection,
	/// Any transmission component (scattering into discrete, 1D, or 2D set of directions)
	ETransmission = EDiffuseTransmission | EDeltaTransmission
		| EDelta1DTransmission | EGlossyTransmission | ENull,
	/// Diffuse scattering into a 2D set of directions
	EDiffuse      = EDiffuseReflection | EDiffuseTransmission,
	/// Non-diffuse scattering into a 2D set of directions
	EGlossy       = EGlossyReflection | EGlossyTransmission,
	/// Scattering into a 2D set of directions
	ESmooth       = EDiffuse | EGlossy,
	/// Scattering into a discrete set of directions
	EDelta        = ENull | EDeltaReflection | EDeltaTransmission,
	/// Scattering into a 1D space of directions
	EDelta1D      = EDelta1DReflection | EDelta1DTransmission,
	/// Any kind of scattering
	EAll          = EDiffuse | EGlossy | EDelta | EDelta1D
};

struct PositionSamplingRecord
{
public:
	float3 p;
	float3 n;
	float pdf;
	EMeasure measure;
	float2 uv;
	///This is so unbelievably ugly it hurts my brain....
	const void* object;
public:
	CUDA_FUNC_IN PositionSamplingRecord() { }
	CUDA_FUNC_IN PositionSamplingRecord(const float3& _p, const float3& _n, const void* _obj, EMeasure m = EArea )
		: p(_p), n(_n), measure(m), object(_obj)
	{

	}
};

struct DirectionSamplingRecord
{
public:
	float3 d;
	float pdf;
	EMeasure measure;
public:
	CUDA_FUNC_IN DirectionSamplingRecord() { }
	CUDA_FUNC_IN DirectionSamplingRecord(const float3 &d, EMeasure m = ESolidAngle)
		: d(d), measure(m)
	{
	}
};

struct DirectSamplingRecord : public PositionSamplingRecord
{
	float3 ref;
	float3 refN;
	float3 d;
	float dist;

	CUDA_FUNC_IN DirectSamplingRecord()
	{
	}

	CUDA_FUNC_IN DirectSamplingRecord(const float3& _p, const float3& _n)
		: PositionSamplingRecord(_p, _n, 0), ref(_p), refN(_n)
	{
		refN = _n;
	}
};

struct PhaseFunctionSamplingRecord
{
	float3 wi;
	float3 wo;
	ETransportMode mode;

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const float3& _wo, ETransportMode m = ERadiance)
	{
		wo = _wo;
		mode = m;
	}

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const float3& _wo, const float3& _wi, ETransportMode m = ERadiance)
	{
		wo = _wo;
		wi = _wi;
		mode = m;
	}

	CUDA_FUNC_IN void reverse()
	{
		swapk(&wo, &wi);
		mode = (ETransportMode) (1-mode);
	}
};

struct BSDFSamplingRecord
{
	CudaRNG* rng;
	DifferentialGeometry& dg;
	/// Normalized incident direction in local coordinates
	float3 wi;
	/// Normalized outgoing direction in local coordinates
	float3 wo;
	/// Relative index of refraction in the sampled direction
	float eta;
	ETransportMode mode;
	unsigned int typeMask;
	unsigned int sampledType;
	CUDA_FUNC_IN BSDFSamplingRecord(DifferentialGeometry& dg) : dg(dg) {}
	CUDA_FUNC_IN void Clear(CudaRNG& _rng)
	{
		rng = &_rng;
		typeMask = ETypeCombinations::EAll;
		sampledType = 0;
		eta = 1.0f;
		mode = ERadiance;
	}
	CUDA_FUNC_IN float3 getOutgoing()
	{
		return normalize(dg.toWorld(wo));
	}
};