#pragma once

#include "Vector.h"
#include "MathFunc.h"

//The general design of the class is copied from PBRT, the conversion/integration routines from Mitsuba.

namespace CudaTracerLib {

#define SPECTRUM_SAMPLES 3

#define SPECTRUM_MIN_WAVELENGTH   360
#define SPECTRUM_MAX_WAVELENGTH   830
#define SPECTRUM_RANGE                \
	(SPECTRUM_MAX_WAVELENGTH-SPECTRUM_MIN_WAVELENGTH)

template <typename T, int N> struct TSpectrum
{
public:
	typedef T          Scalar;
	enum { DIMENSION = N };

	TSpectrum() = default;

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
	CUDA_FUNC_IN Scalar avg() const {
		return sum() * (1.0f / N);
	}

	/// Return the sum over all wavelengths
	CUDA_FUNC_IN Scalar sum() const {
		Scalar result = 0.0f;
		for (int i = 0; i < N; i++)
			result += s[i];
		return result;
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

	Spectrum() = default;

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
		int index = Floor2Int((lambda - SPECTRUM_MIN_WAVELENGTH) *
			((float) SPECTRUM_SAMPLES / (float) SPECTRUM_RANGE));

		if (index < 0 || index >= SPECTRUM_SAMPLES)
			return 0.0f;
		else
			return s[index];
#endif
	}

	/// Return the luminance in candelas.
	CTL_EXPORT CUDA_HOST CUDA_DEVICE float getLuminance() const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toLinearRGB(float &r, float &g, float &b) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromLinearRGB(float r, float g, float b, EConversionIntent intent = EReflectance);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toXYZ(float &x, float &y, float &z) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromXYZ(float x, float y, float z, EConversionIntent intent = EReflectance);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toIPT(float &I, float &P, float &T) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromIPT(float I, float P, float T, EConversionIntent intent = EReflectance);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toSRGB(float &r, float &g, float &b) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromSRGB(float r, float g, float b);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE RGBE toRGBE() const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromRGBE(RGBE rgbe, EConversionIntent intent = EIlluminant);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE RGBCOL toRGBCOL() const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromRGBCOL(RGBCOL col);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toYxy(float &Y, float &x, float &y) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromYxy(float Y, float x, float y, EConversionIntent intent = EReflectance);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void fromHSL(float h, float s, float l);

	CTL_EXPORT CUDA_HOST CUDA_DEVICE void toHSL(float& h, float& s, float& l) const;

	CTL_EXPORT void fromContinuousSpectrum(const float* wls, const float* vals, unsigned int N);

	//samples a wavelength from the spectral power distribution of this spectrum
	CTL_EXPORT CUDA_HOST CUDA_DEVICE float SampleWavelength(Spectrum& res, float& pdf, float sample) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE float PdfWavelength(float lambda) const;

	CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum FWavelength(float lambda) const;
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
	CTL_EXPORT static void StaticInitialize();
	CTL_EXPORT static void StaticDeinitialize();
	CUDA_FUNC_IN static staticData* getData();
};

class SpectrumConverter
{
    //https://people.freebsd.org/~jake/frexp.c
    CUDA_FUNC_IN static double frexp_self(double value, int* eptr)
    {
        struct ieee_double {
            uint32_t   dbl_fracl;
            uint32_t   dbl_frach : 20;
            uint32_t   dbl_exp : 11;
            uint32_t   dbl_sign : 1;
        };

        union {
            double v;
            struct ieee_double s;
        } u;

        if (value) {
            /*
            * Fractions in [0.5..1.0) have an exponent of 2^-1.
            * Leave Inf and NaN alone, however.
            * WHAT ABOUT DENORMS?
            */
            u.v = value;
            const uint32_t DBL_EXP_BIAS = 1023;
            const uint32_t DBL_EXP_INFNAN = 2047;
            if (u.s.dbl_exp != DBL_EXP_INFNAN) {
                *eptr = u.s.dbl_exp - (DBL_EXP_BIAS - 1);
                u.s.dbl_exp = DBL_EXP_BIAS - 1;
            }
            return (u.v);
        }
        else {
            *eptr = 0;
            return (0.0);
        }
    }
public:
	static CUDA_FUNC_IN float y(const Vec3f& v)
	{
		const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
		return YWeight[0] * v.x + YWeight[1] * v.y + YWeight[2] * v.z;
	}

	static CUDA_FUNC_IN RGBCOL Float3ToCOLORREF(const Vec3f& c)
	{
#define toInt(x) (unsigned char)(math::clamp01(x) * 255.0f)
        return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), 255);
#undef toInt
	}

	static CUDA_FUNC_IN Vec3f COLORREFToFloat3(RGBCOL col)
	{
        float r = float(col.x) / 255.0f, g = float(col.y) / 255.0f, b = float(col.z) / 255.0f;
		return Vec3f(r, g, b);
	}

	static CUDA_FUNC_IN RGBE Float3ToRGBE(const Vec3f& c)
	{
        /* Find the largest contribution */
        float max_ = CudaTracerLib::max(c.x, c.y, c.z);
        RGBE rgbe;
        if (max_ < 1e-32) {
            rgbe.x = rgbe.y = rgbe.z = rgbe.w = 0;
        }
        else {
            int e;
            /* Extract exponent and convert the fractional part into
            the [0..255] range. Afterwards, divide by max so that
            any color component multiplied by the result will be in [0,255] */
            //max_ = math::frexp(max_, &e) * 256.0f / max_;
            max_ = frexp_self((double)max_, &e) * 256.0f / max_;
            rgbe.x = (unsigned char)(c.x * max_);
            rgbe.y = (unsigned char)(c.y * max_);
            rgbe.z = (unsigned char)(c.z * max_);
            rgbe.w = (unsigned char)(e + 128); /* Exponent value in bias format */
        }
        return rgbe;
	}

	static CUDA_FUNC_IN Vec3f RGBEToFloat3(RGBE rgbe)
	{
        if (rgbe.w) {
            /* Calculate exponent/256 */
            float exp = ldexp(1.0f, int(rgbe.w) - (128 + 8));
            return Vec3f(rgbe.x*exp, rgbe.y*exp, rgbe.z*exp);
        }
        else return Vec3f(0.0f);
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
