#pragma once

#include <MathTypes.h>

//Implementation and interface copied from Mitsuba.

namespace CudaTracerLib {

struct CudaRNG;
struct DifferentialGeometry;

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
	ENull = 0x00001,
	/// Ideally diffuse reflection
	EDiffuseReflection = 0x00002,
	/// Ideally diffuse transmission
	EDiffuseTransmission = 0x00004,
	/// Glossy reflection
	EGlossyReflection = 0x00008,
	/// Glossy transmission
	EGlossyTransmission = 0x00010,
	/// Reflection into a discrete set of directions
	EDeltaReflection = 0x00020,
	/// Transmission into a discrete set of directions
	EDeltaTransmission = 0x00040,
	/// Reflection into a 1D space of directions
	EDelta1DReflection = 0x00080,
	/// Transmission into a 1D space of directions
	EDelta1DTransmission = 0x00100,

	// =============================================================
	//!                   Other lobe attributes
	// =============================================================
	/// The lobe is not invariant to rotation around the normal
	EAnisotropic = 0x01000,
	/// The BSDF depends on the UV coordinates
	ESpatiallyVarying = 0x02000,
	/// Flags non-symmetry (e.g. transmission in dielectric materials)
	ENonSymmetric = 0x04000,
	/// Supports interactions on the front-facing side
	EFrontSide = 0x08000,
	/// Supports interactions on the back-facing side
	EBackSide = 0x10000,
	/// Uses extra random numbers from the supplied sampler instance
	EUsesSampler = 0x20000,
};

enum ETypeCombinations {
	/// Any reflection component (scattering into discrete, 1D, or 2D set of directions)
	EReflection = EDiffuseReflection | EDeltaReflection
	| EDelta1DReflection | EGlossyReflection,
	/// Any transmission component (scattering into discrete, 1D, or 2D set of directions)
	ETransmission = EDiffuseTransmission | EDeltaTransmission
	| EDelta1DTransmission | EGlossyTransmission | ENull,
	/// Diffuse scattering into a 2D set of directions
	EDiffuse = EDiffuseReflection | EDiffuseTransmission,
	/// Non-diffuse scattering into a 2D set of directions
	EGlossy = EGlossyReflection | EGlossyTransmission,
	/// Scattering into a 2D set of directions
	ESmooth = EDiffuse | EGlossy,
	/// Scattering into a discrete set of directions
	EDelta = ENull | EDeltaReflection | EDeltaTransmission,
	/// Scattering into a 1D space of directions
	EDelta1D = EDelta1DReflection | EDelta1DTransmission,
	/// Any kind of scattering
	EAll = EDiffuse | EGlossy | EDelta | EDelta1D
};

struct PositionSamplingRecord
{
public:
	Vec3f p;
	Vec3f n;
	float pdf;
	EMeasure measure;
	Vec2f uv;
	///This is so unbelievably ugly it hurts my brain....
	CUDA_ALIGN(16) const void* object;
public:
	CUDA_FUNC_IN PositionSamplingRecord() { }
	CUDA_FUNC_IN PositionSamplingRecord(const Vec3f& _p, const Vec3f& _n, const void* _obj, EMeasure m = EArea)
		: p(_p), n(_n), measure(m), object(_obj)
	{

	}
};

struct DirectionSamplingRecord
{
public:
	Vec3f d;
	float pdf;
	EMeasure measure;
public:
	CUDA_FUNC_IN DirectionSamplingRecord() { }
	CUDA_FUNC_IN DirectionSamplingRecord(const Vec3f &d, EMeasure m = ESolidAngle)
		: d(d), measure(m)
	{
	}
};

struct DirectSamplingRecord : public PositionSamplingRecord
{
	Vec3f ref;
	Vec3f refN;
	Vec3f d;
	float dist;

	CUDA_FUNC_IN DirectSamplingRecord()
	{
	}

	CUDA_FUNC_IN DirectSamplingRecord(const Vec3f& _p, const Vec3f& _n)
		: PositionSamplingRecord(_p, _n, 0), ref(_p), refN(_n)
	{
		refN = _n;
	}
};

struct PhaseFunctionSamplingRecord
{
	Vec3f wi;
	Vec3f wo;
	ETransportMode mode;

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const Vec3f& _wo, ETransportMode m = ERadiance)
	{
		wo = _wo;
		mode = m;
	}

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const Vec3f& _wo, const Vec3f& _wi, ETransportMode m = ERadiance)
	{
		wo = _wo;
		wi = _wi;
		mode = m;
	}

	CUDA_FUNC_IN void reverse()
	{
		swapk(wo, wi);
		mode = (ETransportMode)(1 - mode);
	}
};

struct BSDFSamplingRecord
{
	CudaRNG* rng;
	DifferentialGeometry& dg;
	/// Normalized incident direction in local coordinates
	Vec3f wi;
	/// Normalized outgoing direction in local coordinates
	Vec3f wo;
	/// Relative index of refraction in the sampled direction
	float eta;
	ETransportMode mode;
	unsigned int typeMask;
	unsigned int sampledType;
	CUDA_FUNC_IN BSDFSamplingRecord(DifferentialGeometry& dg) : dg(dg) {}
	CUDA_DEVICE CUDA_HOST Vec3f getOutgoing();
	CUDA_DEVICE CUDA_HOST BSDFSamplingRecord& operator=(const BSDFSamplingRecord& other);
};

}