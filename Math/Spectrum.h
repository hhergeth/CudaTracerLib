#pragma once

#include "Vector.h"

//The general design of the class is copied from PBRT, the conversion/integration routines from Mitsuba.

namespace CudaTracerLib {

#define SPECTRUM_SAMPLES 3

#define SPECTRUM_min_WAVELENGTH   360
#define SPECTRUM_max_WAVELENGTH   830
#define SPECTRUM_RANGE                \
	(SPECTRUM_max_WAVELENGTH-SPECTRUM_min_WAVELENGTH)

template <typename T, int N> struct TSpectrum
{
public:
	typedef T          Scalar;
	const static int dim = N;

	CUDA_FUNC_IN TSpectrum() { }

	CUDA_FUNC_IN TSpectrum(Scalar v) {
		for (int i = 0; i < N; i++)
			s[i] = v;
	}

	/// Copy a spectral power distribution
	CUDA_FUNC_IN TSpectrum(Scalar spec[N]) {
		for (int i = 0; i < N; i++)
			s[i] = spec[i];
	}

	/// Initialize with a TSpectrum data type based on a alternate representation
	template <typename AltScalar> CUDA_FUNC_IN TSpectrum(const TSpectrum<AltScalar, N> &v) {
		for (int i = 0; i < N; ++i)
			s[i] = (Scalar)v[i];
	}

	/// Add two spectral power distributions
	CUDA_FUNC_IN TSpectrum operator+(const TSpectrum &spec) const {
		TSpectrum value = *this;
		for (int i = 0; i < N; i++)
			value.s[i] += spec.s[i];
		return value;
	}

	/// Add a spectral power distribution to this instance
	CUDA_FUNC_IN TSpectrum& operator+=(const TSpectrum &spec) {
		for (int i = 0; i < N; i++)
			s[i] += spec.s[i];
		return *this;
	}

	/// Subtract a spectral power distribution
	CUDA_FUNC_IN TSpectrum operator-(const TSpectrum &spec) const {
		TSpectrum value = *this;
		for (int i = 0; i < N; i++)
			value.s[i] -= spec.s[i];
		return value;
	}

	/// Subtract a spectral power distribution from this instance
	CUDA_FUNC_IN TSpectrum& operator-=(const TSpectrum &spec) {
		for (int i = 0; i < N; i++)
			s[i] -= spec.s[i];
		return *this;
	}

	/// Multiply by a scalar
	CUDA_FUNC_IN TSpectrum operator*(Scalar f) const {
		TSpectrum value = *this;
		for (int i = 0; i < N; i++)
			value.s[i] *= f;
		return value;
	}

	/// Multiply by a scalar
	CUDA_FUNC_IN friend TSpectrum operator*(Scalar f, const TSpectrum &spec) {
		return spec * f;
	}

	/// Multiply by a scalar
	CUDA_FUNC_IN TSpectrum& operator*=(Scalar f) {
		for (int i = 0; i < N; i++)
			s[i] *= f;
		return *this;
	}

	/// Perform a component-wise multiplication by another spectrum
	CUDA_FUNC_IN TSpectrum operator*(const TSpectrum &spec) const {
		TSpectrum value = *this;
		for (int i = 0; i < N; i++)
			value.s[i] *= spec.s[i];
		return value;
	}

	/// Perform a component-wise multiplication by another spectrum
	CUDA_FUNC_IN TSpectrum& operator*=(const TSpectrum &spec) {
		for (int i = 0; i < N; i++)
			s[i] *= spec.s[i];
		return *this;
	}

	/// Perform a component-wise division by another spectrum
	CUDA_FUNC_IN TSpectrum& operator/=(const TSpectrum &spec) {
		for (int i = 0; i < N; i++)
			s[i] /= spec.s[i];
		return *this;
	}

	/// Perform a component-wise division by another spectrum
	CUDA_FUNC_IN TSpectrum operator/(const TSpectrum &spec) const {
		TSpectrum value = *this;
		for (int i = 0; i < N; i++)
			value.s[i] /= spec.s[i];
		return value;
	}

	CUDA_FUNC_IN TSpectrum operator/(Scalar f) const {
		TSpectrum value = *this;
		Scalar recip = 1.0f / f;
		for (int i = 0; i < N; i++)
			value.s[i] *= recip;
		return value;
	}

	/// Equality test
	CUDA_FUNC_IN bool operator==(const TSpectrum &spec) const {
		for (int i = 0; i < N; i++) {
			if (s[i] != spec.s[i])
				return false;
		}
		return true;
	}

	/// Inequality test
	CUDA_FUNC_IN bool operator!=(const TSpectrum &spec) const {
		return !operator==(spec);
	}

	/// Divide by a scalar
	CUDA_FUNC_IN friend TSpectrum operator/(Scalar f, TSpectrum &spec) {
		return TSpectrum(f) / spec;
	}

	/// Divide by a scalar
	CUDA_FUNC_IN TSpectrum& operator/=(Scalar f) {
		Scalar recip = 1.0f / f;
		for (int i = 0; i < N; i++)
			s[i] *= recip;
		return *this;
	}

	/// Check for NaNs
	CUDA_FUNC_IN bool isNaN() const {
		for (int i = 0; i < N; i++)
			if (isnan(s[i]))
				return true;
		return false;
	}

	/// Returns whether the spectrum only contains valid (non-NaN, nonnegative) samples
	CUDA_FUNC_IN bool isValid() const {
		for (int i = 0; i < N; i++)
			if (!isfinite(s[i]) || s[i] < 0.0f)
				return false;
		return true;
	}

	/// Multiply-accumulate operation, adds \a weight * \a spec
	CUDA_FUNC_IN void addWeighted(Scalar weight, const TSpectrum &spec) {
		for (int i = 0; i < N; i++)
			s[i] += weight * spec.s[i];
	}

	/// Return the average over all wavelengths
	CUDA_FUNC_IN Scalar average() const {
		Scalar result = 0.0f;
		for (int i = 0; i < N; i++)
			result += s[i];
		return result * (1.0f / N);
	}

	/// Component-wise square root
	CUDA_FUNC_IN TSpectrum sqrt() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::sqrt(s[i]);
		return value;
	}

	/// Component-wise abs
	CUDA_FUNC_IN TSpectrum abs() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::abs(s[i]);
		return value;
	}

	/// Component-wise saturate
	CUDA_FUNC_IN TSpectrum saturate() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::clamp01(s[i]);
		return value;
	}

	/// Component-wise square root
	CUDA_FUNC_IN TSpectrum safe_sqrt() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::safe_sqrt(s[i]);
		return value;
	}

	/// Component-wise exponentation
	CUDA_FUNC_IN TSpectrum exp() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::exp(s[i]);
		return value;
	}

	/// Component-wise power
	CUDA_FUNC_IN TSpectrum pow(Scalar f) const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = math::pow(s[i], f);
		return value;
	}

	/// math::clamp negative values
	CUDA_FUNC_IN void clampNegative() {
		for (int i = 0; i < N; i++)
			s[i] = CudaTracerLib::max((Scalar) 0.0f, s[i]);
	}

	/// Return the highest-valued spectral sample
	CUDA_FUNC_IN Scalar max() const {
		Scalar result = s[0];
		for (int i = 1; i < N; i++)
			result = CudaTracerLib::max(result, s[i]);
		return result;
	}

	/// Return the lowest-valued spectral sample
	CUDA_FUNC_IN Scalar min() const {
		Scalar result = s[0];
		for (int i = 1; i < N; i++)
			result = CudaTracerLib::min(result, s[i]);
		return result;
	}

	/// Negate
	CUDA_FUNC_IN TSpectrum operator-() const {
		TSpectrum value;
		for (int i = 0; i < N; i++)
			value.s[i] = -s[i];
		return value;
	}

	/// Indexing operator
	CUDA_FUNC_IN Scalar &operator[](int entry) {
		return s[entry];
	}

	/// Indexing operator
	CUDA_FUNC_IN Scalar operator[](int entry) const {
		return s[entry];
	}

	/// Check if this spectrum is zero at all wavelengths
	CUDA_FUNC_IN bool isZero() const {
		for (int i = 0; i < N; i++) {
			if (s[i] != 0.0f)
				return false;
		}
		return true;
	}

	template <class T2, int L2> friend std::ostream& operator<< (std::ostream & os, const TSpectrum<T2, L2>& rhs);
protected:
	Scalar s[N];
};

template <class T2, int L2> inline std::ostream& operator << (std::ostream & stream, const TSpectrum<T2, L2>& v)
{
	stream << "(";
	for (int i = 0; i < L2; i++)
	{
		if (i != 0)
			stream << ", ";
		stream << v.operator[](i);
	}
	stream << ")";
	return stream;
}

template<typename T, int N> CUDA_FUNC_IN TSpectrum<T, N> max(const TSpectrum<T, N>& a, const TSpectrum<T, N>& b)
{
	TSpectrum<T, N> r;
	for (int i = 0; i < N; i++)
		r[i] = max(a[i], b[i]);
	return r;
}

template<typename T, int N> CUDA_FUNC_IN TSpectrum<T, N> min(const TSpectrum<T, N>& a, const TSpectrum<T, N>& b)
{
	TSpectrum<T, N> r;
	for (int i = 0; i < N; i++)
		r[i] = min(a[i], b[i]);
	return r;
}

typedef uchar4 RGBCOL;
typedef uchar4 RGBE;

struct Spectrum : public TSpectrum<float, SPECTRUM_SAMPLES> {
public:
	typedef TSpectrum<float, SPECTRUM_SAMPLES> Parent;

	enum EConversionIntent {
		EReflectance,
		EIlluminant
	};

	CUDA_FUNC_IN Spectrum() { }

	CUDA_FUNC_IN Spectrum(float r, float g, float b)
	{
		fromLinearRGB(r, g, b);
	}

	/// Construct from a TSpectrum instance
	CUDA_FUNC_IN Spectrum(const Parent &s) : Parent(s) { }

	/// Initialize with a TSpectrum data type based on a alternate representation
	template <typename AltScalar> CUDA_FUNC_IN explicit Spectrum(const TSpectrum<AltScalar, SPECTRUM_SAMPLES> &v) {
		for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
			s[i] = (Scalar)v[i];
	}

	/// Create a new spectral power distribution with all samples set to the given value
	CUDA_FUNC_IN Spectrum(float v) {
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			s[i] = v;
	}

	/// Copy a spectral power distribution
	explicit CUDA_FUNC_IN Spectrum(float value[SPECTRUM_SAMPLES]) {
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			s[i] = value[i];
	}

	CUDA_FUNC_IN bool operator==(const Spectrum &val) const {
		for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
			if (s[i] != val.s[i])
				return false;
		}
		return true;
	}

	CUDA_FUNC_IN bool operator!=(const Spectrum &val) const {
		return !operator==(val);
	}

	/**
	 * \brief Evaluate the SPD for the given wavelength
	 * in nanometers.
	 */
	CUDA_FUNC_IN float eval(float lambda) const  {
#if SPECTRUM_SAMPLES == 3
		return 0.0f;
#else
		int index = Floor2Int((lambda - SPECTRUM_min_WAVELENGTH) *
			((float) SPECTRUM_SAMPLES / (float) SPECTRUM_RANGE));

		if (index < 0 || index >= SPECTRUM_SAMPLES)
			return 0.0f;
		else
			return s[index];
#endif
	}

	/// Return the luminance in candelas.
	CUDA_HOST CUDA_DEVICE float getLuminance() const;

	CUDA_HOST CUDA_DEVICE void toLinearRGB(float &r, float &g, float &b) const;

	CUDA_HOST CUDA_DEVICE void fromLinearRGB(float r, float g, float b, EConversionIntent intent = EReflectance);

	CUDA_HOST CUDA_DEVICE void toXYZ(float &x, float &y, float &z) const;

	CUDA_HOST CUDA_DEVICE void fromXYZ(float x, float y, float z, EConversionIntent intent = EReflectance);

	CUDA_HOST CUDA_DEVICE void toIPT(float &I, float &P, float &T) const;

	CUDA_HOST CUDA_DEVICE void fromIPT(float I, float P, float T, EConversionIntent intent = EReflectance);

	CUDA_HOST CUDA_DEVICE void toSRGB(float &r, float &g, float &b) const;

	CUDA_HOST CUDA_DEVICE void fromSRGB(float r, float g, float b);

	CUDA_HOST CUDA_DEVICE RGBE toRGBE() const;

	CUDA_HOST CUDA_DEVICE void fromRGBE(RGBE rgbe, EConversionIntent intent = EIlluminant);

	CUDA_HOST CUDA_DEVICE RGBCOL toRGBCOL() const;

	CUDA_HOST CUDA_DEVICE void fromRGBCOL(RGBCOL col);

	CUDA_HOST CUDA_DEVICE void toYxy(float &Y, float &x, float &y) const;

	CUDA_HOST CUDA_DEVICE void fromYxy(float Y, float x, float y, EConversionIntent intent = EReflectance);

	CUDA_HOST CUDA_DEVICE void fromHSL(float h, float s, float l);

	CUDA_HOST CUDA_DEVICE void toHSL(float& h, float& s, float& l) const;

	void fromContinuousSpectrum(const float* wls, const float* vals, unsigned int N);

	//samples a wavelength from the spectral power distribution of this spectrum
	CUDA_HOST CUDA_DEVICE float SampleWavelength(Spectrum& res, float& pdf, float sample) const;

	CUDA_HOST CUDA_DEVICE float PdfWavelength(float lambda) const;

	CUDA_HOST CUDA_DEVICE Spectrum FWavelength(float lambda) const;
};

static const int   CIE_samples = 471;

class SpectrumHelper
{
public:
	struct staticData
	{
		Spectrum CIE_X;
		Spectrum CIE_Y;
		Spectrum CIE_Z;
		float CIE_normalization;
		Spectrum rgbRefl2SpecWhite;
		Spectrum rgbRefl2SpecCyan;
		Spectrum rgbRefl2SpecMagenta;
		Spectrum rgbRefl2SpecYellow;
		Spectrum rgbRefl2SpecRed;
		Spectrum rgbRefl2SpecGreen;
		Spectrum rgbRefl2SpecBlue;
		Spectrum rgbIllum2SpecWhite;
		Spectrum rgbIllum2SpecCyan;
		Spectrum rgbIllum2SpecMagenta;
		Spectrum rgbIllum2SpecYellow;
		Spectrum rgbIllum2SpecRed;
		Spectrum rgbIllum2SpecGreen;
		Spectrum rgbIllum2SpecBlue;
		float m_wavelengths[SPECTRUM_SAMPLES + 1];

		float m_CIE_wavelengths[CIE_samples];
		float m_CIE_X_entries[CIE_samples];
		float m_CIE_Y_entries[CIE_samples];
		float m_CIE_Z_entries[CIE_samples];

		void init();
	};
	static void StaticInitialize();
	static void StaticDeinitialize();
	CUDA_FUNC_IN static staticData* getData();
};

class SpectrumConverter
{
public:
	static CUDA_FUNC_IN float y(const Vec3f& v)
	{
		const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
		return YWeight[0] * v.x + YWeight[1] * v.y + YWeight[2] * v.z;
	}

#define toInt(x) (int((float)math::pow(math::clamp01(x),1.0f/1.2f)*255.0f+0.5f))
	//#define toInt(x) (unsigned char(x * 255.0f))

	static CUDA_FUNC_IN RGBCOL Float4ToCOLORREF(const Vec4f& c)
	{
		return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), toInt(c.w));
	}

	static CUDA_FUNC_IN Vec4f COLORREFToFloat4(RGBCOL c)
	{
		return Vec4f((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, (float)c.w / 255.0f);
	}

	static CUDA_FUNC_IN RGBCOL Float3ToCOLORREF(const Vec3f& c)
	{
		return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), 255);
	}

	static CUDA_FUNC_IN Vec3f COLORREFToFloat3(RGBCOL c)
	{
		return Vec3f((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f);
	}
#undef toInt

	static CUDA_FUNC_IN RGBE Float3ToRGBE(const Vec3f& c)
	{
		float v = c.max();
		if (v < 1e-32)
			return make_uchar4(0, 0, 0, 0);
		else
		{
			int e;
			v = frexp(v, &e) * 256.0f / v;
			return make_uchar4((unsigned char)(c.x * v), (unsigned char)(c.y * v), (unsigned char)(c.z * v), e + 128);
		}
	}

	static CUDA_FUNC_IN Vec3f RGBEToFloat3(RGBE a)
	{
		float f = ldexp(1.0f, a.w - (int)(128 + 8));
		return Vec3f((float)a.x * f, (float)a.y * f, (float)a.z * f);
	}

	///this is not luminence! This is some strange msdn stuff, no idea
	static CUDA_FUNC_IN float Luma(const Vec3f& c)
	{
		return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
	}

	static CUDA_FUNC_IN Vec3f RGBToXYZ(const Vec3f& c)
	{
		Vec3f r;
		r.x = dot(Vec3f(0.5767309f, 0.1855540f, 0.1881852f), c);
		r.y = dot(Vec3f(0.2973769f, 0.6273491f, 0.0752741f), c);
		r.z = dot(Vec3f(0.0270343f, 0.0706872f, 0.9911085f), c);
		return r;
	}

	static CUDA_FUNC_IN Vec3f XYZToRGB(const Vec3f& c)
	{
		Vec3f r;
		r.x = dot(Vec3f(2.0413690f, -0.5649464f, -0.3446944f), c);
		r.y = dot(Vec3f(-0.9692660f, 1.8760108f, 0.0415560f), c);
		r.z = dot(Vec3f(0.0134474f, -0.1183897f, 1.0154096f), c);
		return r;
	}

	static CUDA_FUNC_IN Vec3f XYZToYxy(const Vec3f& c)
	{
		float s = c.x + c.y + c.z;
		return Vec3f(c.y, c.x / s, c.y / s);
	}

	static CUDA_FUNC_IN Vec3f YxyToXYZ(const Vec3f& c)
	{
		Vec3f r;
		r.x = c.x * c.y / c.z;
		r.y = c.x;
		r.z = c.x * (1.0f - c.y - c.z) / c.z;
		return r;
	}
};

}
