#pragma once

#include <MathTypes.h>
#include "e_KernelMapping.h"

struct CameraSample
{
    float imageX, imageY;
    float lensU, lensV;
    float time;
};

struct LightSample
{
	CUDA_FUNC_IN LightSample() { }
	CUDA_FUNC_IN LightSample(CudaRNG &rng)
	{
		uPos[0] = rng.randomFloat();
		uPos[1] = rng.randomFloat();
		uComponent = rng.randomFloat();
	}
	CUDA_FUNC_IN LightSample(float up0, float up1, float ucomp)
	{
		uPos[0] = up0; uPos[1] = up1;
		uComponent = ucomp;
	}
	float uPos[2], uComponent;
};

enum ETransportMode
{
	ERadiance = 0,
	EImportance = 1,
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

/// Convenient combinations of flags from \ref EBSDFType
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

struct BSDFSamplingRecord
{
	CudaRNG* rng;
	MapParameters map;
	/// Normalized incident direction in local coordinates
	float3 wi;
	/// Normalized outgoing direction in local coordinates
	float3 wo;
	/// Relative index of refraction in the sampled direction
	float eta;
	ETransportMode mode;
	unsigned int typeMask;
	int component;
	unsigned int sampledType;
	int sampledComponent;
	CUDA_FUNC_IN BSDFSamplingRecord():rng(0){}
	CUDA_FUNC_IN BSDFSamplingRecord(const MapParameters& mp, const Ray& r, CudaRNG& _rng, ETransportMode _mode = ERadiance)
		: map(mp), wi(-1.0 * r.direction), mode(_mode), eta(1.0f),
		  typeMask(ETypeCombinations::EAll), component(-1), sampledType(0), sampledComponent(-1),
		  rng(&_rng)
	{

	}
};