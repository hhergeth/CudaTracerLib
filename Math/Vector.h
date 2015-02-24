#pragma once

#include "../Defines.h"
#include <iostream>
#pragma warning(push, 3)
#include <vector_functions.h> // float4, etc.
#pragma warning(pop)
#include "MathFunc.h"

template <class T, int L> class Vector;

template <class T, int L, class S> class VectorBase
{
public:
	CUDA_FUNC_IN                    VectorBase(void)                      {}

	CUDA_FUNC_IN    const T*        getPtr(void) const                { return ((S*)this)->getPtr(); }
	CUDA_FUNC_IN    T*              getPtr(void)                      { return ((S*)this)->getPtr(); }
	CUDA_FUNC_IN    const T&        get(int idx) const             { CT_ASSERT(idx >= 0 && idx < L); return getPtr()[idx]; }
	CUDA_FUNC_IN    T&              get(int idx)                   { CT_ASSERT(idx >= 0 && idx < L); return getPtr()[idx]; }
	CUDA_FUNC_IN    T               set(int idx, const T& a)       { T& slot = get(idx); T old = slot; slot = a; return old; }

	CUDA_FUNC_IN    void            set(const T& a)                { T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = a; }
	CUDA_FUNC_IN    void            set(const T* ptr)              { CT_ASSERT(ptr); T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = ptr[i]; }
	CUDA_FUNC_IN    void            setZero(void)                      { set((T)0); }

	CUDA_FUNC_IN    bool            isZero(void) const                { const T* tp = getPtr(); for (int i = 0; i < L; i++) if (tp[i] != (T)0) return false; return true; }
	CUDA_FUNC_IN    T               lenSqr(void) const                { const T* tp = getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += math::sqr(tp[i]); return r; }
	CUDA_FUNC_IN    T               length(void) const                { return sqrt(lenSqr()); }
	CUDA_FUNC_IN    S               normalized(T len = (T)1) const        { return operator*(len * math::rcp(length())); }
	CUDA_FUNC_IN    void            normalize(T len = (T)1)              { set(normalized(len)); }
	CUDA_FUNC_IN    T               min(void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = ::min(r, tp[i]); return r; }
	CUDA_FUNC_IN    T               max(void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = ::max(r, tp[i]); return r; }
	CUDA_FUNC_IN    T               sum(void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r += tp[i]; return r; }
	CUDA_FUNC_IN    S               abs(void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = math::abs(tp[i]); return r; }
	CUDA_FUNC_IN	S				sign() const					{ const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = math::sign(tp[i]); return r; }
	CUDA_FUNC_IN	S				floor() const					{ const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = math::floor(tp[i]); return r; }
	CUDA_FUNC_IN	S				ceil() const					{ const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = math::ceil(tp[i]); return r; }

	CUDA_FUNC_IN    Vector<T, L + 1> toHomogeneous(void) const              { const T* tp = getPtr(); Vector<T, L + 1> r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i]; rp[L] = (T)1; return r; }
	CUDA_FUNC_IN    Vector<T, L - 1> toCartesian(void) const                { const T* tp = getPtr(); Vector<T, L - 1> r; T* rp = r.getPtr(); T c = rcp(tp[L - 1]); for (int i = 0; i < L - 1; i++) rp[i] = tp[i] * c; return r; }

	CUDA_FUNC_IN    const T&        operator[]  (int idx) const             { return get(idx); }
	CUDA_FUNC_IN    T&              operator[]  (int idx)                   { return get(idx); }

	CUDA_FUNC_IN    S&              operator=   (const T& a)                { set(a); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator+=  (const T& a)                { set(operator+(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator-=  (const T& a)                { set(operator-(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator*=  (const T& a)                { set(operator*(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator/=  (const T& a)                { set(operator/(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator%=  (const T& a)                { set(operator%(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator&=  (const T& a)                { set(operator&(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator|=  (const T& a)                { set(operator|(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator^=  (const T& a)                { set(operator^(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator<<= (const T& a)                { set(operator<<(a)); return *(S*)this; }
	CUDA_FUNC_IN    S&              operator>>= (const T& a)                { set(operator>>(a)); return *(S*)this; }

	CUDA_FUNC_IN    S               operator+   (void) const                { return *this; }
	CUDA_FUNC_IN    S               operator-   (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = -tp[i]; return r; }
	CUDA_FUNC_IN    S               operator~   (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = ~tp[i]; return r; }

	CUDA_FUNC_IN    S               operator+   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] + a; return r; }
	CUDA_FUNC_IN    S               operator-   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] - a; return r; }
	CUDA_FUNC_IN    S               operator*   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] * a; return r; }
	CUDA_FUNC_IN    S               operator/   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] / a; return r; }
	CUDA_FUNC_IN    S               operator%   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] % a; return r; }
	CUDA_FUNC_IN    S               operator&   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] & a; return r; }
	CUDA_FUNC_IN    S               operator|   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] | a; return r; }
	CUDA_FUNC_IN    S               operator^   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] ^ a; return r; }
	CUDA_FUNC_IN    S               operator<<  (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] << a; return r; }
	CUDA_FUNC_IN    S               operator>>  (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] >> a; return r; }

	template <class V> CUDA_FUNC_IN void    set(const VectorBase<T, L, V>& v)          { set(v.getPtr()); }
	template <class V> CUDA_FUNC_IN T       dot(const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += tp[i] * vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       min(const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = ::min(tp[i], vp[i]); return r; }
	template <class V> CUDA_FUNC_IN S       max(const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = ::max(tp[i], vp[i]); return r; }
	template <class V, class W> CUDA_FUNC_IN S clamp(const VectorBase<T, L, V>& lo, const VectorBase<T, L, W>& hi) const { const T* tp = getPtr(); const T* lop = lo.getPtr(); const T* hip = hi.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = math::clamp(tp[i], lop[i], hip[i]); return r; }

	template <class V> CUDA_FUNC_IN S&      operator=   (const VectorBase<T, L, V>& v)          { set(v); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator+=  (const VectorBase<T, L, V>& v)          { set(operator+(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator-=  (const VectorBase<T, L, V>& v)          { set(operator-(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator*=  (const VectorBase<T, L, V>& v)          { set(operator*(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator/=  (const VectorBase<T, L, V>& v)          { set(operator/(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator%=  (const VectorBase<T, L, V>& v)          { set(operator%(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator&=  (const VectorBase<T, L, V>& v)          { set(operator&(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator|=  (const VectorBase<T, L, V>& v)          { set(operator|(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator^=  (const VectorBase<T, L, V>& v)          { set(operator^(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator<<= (const VectorBase<T, L, V>& v)          { set(operator<<(v)); return *(S*)this; }
	template <class V> CUDA_FUNC_IN S&      operator>>= (const VectorBase<T, L, V>& v)          { set(operator>>(v)); return *(S*)this; }

	template <class V> CUDA_FUNC_IN S       operator+   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] + vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator-   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] - vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator*   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] * vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator/   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] / vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator%   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] % vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator&   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] & vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator|   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] | vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator^   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] ^ vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator<<  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] << vp[i]; return r; }
	template <class V> CUDA_FUNC_IN S       operator>>  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] >> vp[i]; return r; }

	template <class V> CUDA_FUNC_IN bool    operator==  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); for (int i = 0; i < L; i++) if (tp[i] != vp[i]) return false; return true; }
	template <class V> CUDA_FUNC_IN bool    operator!=  (const VectorBase<T, L, V>& v) const    { return (!operator==(v)); }


	template <class T2, int L2, class S2> friend std::ostream& operator<< (std::ostream & os, const VectorBase<T2, L2, S2 >& rhs);
};

template <class T2, int L2, class S2> inline std::ostream& operator << (std::ostream & stream, const VectorBase<T2, L2, S2 >& v)
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

//------------------------------------------------------------------------

template <class T, int L> class Vector : public VectorBase<T, L, Vector<T, L> >
{
public:
	CUDA_FUNC_IN                    Vector(void)                      {  }
	CUDA_FUNC_IN                    Vector(T a)                       { set(a); }

	CUDA_FUNC_IN    const T*        getPtr(void) const                { return m_values; }
	CUDA_FUNC_IN    T*              getPtr(void)                      { return m_values; }
	static CUDA_FUNC_IN Vector      fromPtr(const T* ptr)              { Vector v; v.set(ptr); return v; }

	template <class V> CUDA_FUNC_IN Vector(const VectorBase<T, L, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vector& operator=(const VectorBase<T, L, V>& v) { set(v); return *this; }

private:
	T               m_values[L];
};

class Vec2i : public VectorBase<int, 2, Vec2i>, public int2
{
public:
	CUDA_FUNC_IN                    Vec2i(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec2i(int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec2i(int xx, int yy)            { x = xx; y = yy; }

	CUDA_FUNC_IN    const int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec2i       fromPtr(const int* ptr)            { return Vec2i(ptr[0], ptr[1]); }

	CUDA_FUNC_IN    Vec2i           perpendicular(void) const               { return Vec2i(-y, x); }

	template <class V> CUDA_FUNC_IN Vec2i(const VectorBase<int, 2, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec2i& operator=(const VectorBase<int, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3i : public VectorBase<int, 3, Vec3i>, public int3
{
public:
	CUDA_FUNC_IN                    Vec3i(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec3i(int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec3i(int xx, int yy, int zz)    { x = xx; y = yy; z = zz; }
	CUDA_FUNC_IN                    Vec3i(const Vec2i& xy, int zz)   { x = xy.x; y = xy.y; z = zz; }

	CUDA_FUNC_IN    const int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec3i       fromPtr(const int* ptr)            { return Vec3i(ptr[0], ptr[1], ptr[2]); }

	CUDA_FUNC_IN    Vec2i           getXY(void) const                { return Vec2i(x, y); }

	template <class V> CUDA_FUNC_IN Vec3i(const VectorBase<int, 3, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec3i& operator=(const VectorBase<int, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4i : public VectorBase<int, 4, Vec4i>, public int4
{
public:
	CUDA_FUNC_IN                    Vec4i(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec4i(int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec4i(int xx, int yy, int zz, int ww) { x = xx; y = yy; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4i(const Vec2i& xy, int zz, int ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4i(const Vec3i& xyz, int ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
	CUDA_FUNC_IN                    Vec4i(const Vec2i& xy, const Vec2i& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }

	CUDA_FUNC_IN    const int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec4i       fromPtr(const int* ptr)            { return Vec4i(ptr[0], ptr[1], ptr[2], ptr[3]); }

	CUDA_FUNC_IN    Vec2i           getXY(void) const                { return Vec2i(x, y); }
	CUDA_FUNC_IN    Vec3i           getXYZ(void) const                { return Vec3i(x, y, z); }
	CUDA_FUNC_IN    Vec3i           getXYW(void) const                { return Vec3i(x, y, w); }

	template <class V> CUDA_FUNC_IN Vec4i(const VectorBase<int, 4, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec4i& operator=(const VectorBase<int, 4, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec2u : public VectorBase<unsigned int, 2, Vec2u>, public uint2
{
public:
	CUDA_FUNC_IN                    Vec2u(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec2u(unsigned int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec2u(unsigned int xx, unsigned int yy)            { x = xx; y = yy; }

	CUDA_FUNC_IN    const unsigned int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    unsigned int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec2u       fromPtr(const int* ptr)            { return Vec2u(ptr[0], ptr[1]); }

	template <class V> CUDA_FUNC_IN Vec2u(const VectorBase<unsigned int, 2, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec2u& operator=(const VectorBase<unsigned int, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3u : public VectorBase<unsigned int, 3, Vec3u>, public uint3
{
public:
	CUDA_FUNC_IN                    Vec3u(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec3u(unsigned int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec3u(unsigned int xx, unsigned int yy, unsigned int zz)    { x = xx; y = yy; z = zz; }
	CUDA_FUNC_IN                    Vec3u(const Vec2u& xy, unsigned int zz)   { x = xy.x; y = xy.y; z = zz; }

	CUDA_FUNC_IN    const unsigned int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    unsigned int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec3u       fromPtr(const unsigned int* ptr)            { return Vec3u(ptr[0], ptr[1], ptr[2]); }

	CUDA_FUNC_IN    Vec2u           getXY(void) const                { return Vec2u(x, y); }

	template <class V> CUDA_FUNC_IN Vec3u(const VectorBase<unsigned int, 3, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec3u& operator=(const VectorBase<unsigned int, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4u : public VectorBase<unsigned int, 4, Vec4u>, public uint4
{
public:
	CUDA_FUNC_IN                    Vec4u(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec4u(unsigned int a)                     { set(a); }
	CUDA_FUNC_IN                    Vec4u(unsigned int xx, unsigned int yy, unsigned int zz, unsigned int ww) { x = xx; y = yy; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4u(const Vec2i& xy, unsigned int zz, unsigned int ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4u(const Vec3i& xyz, unsigned int ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
	CUDA_FUNC_IN                    Vec4u(const Vec2i& xy, const Vec2i& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }

	CUDA_FUNC_IN    const unsigned int*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    unsigned int*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec4u       fromPtr(const unsigned int* ptr)            { return Vec4u(ptr[0], ptr[1], ptr[2], ptr[3]); }

	CUDA_FUNC_IN    Vec2u           getXY(void) const                { return Vec2u(x, y); }
	CUDA_FUNC_IN    Vec3u           getXYZ(void) const                { return Vec3u(x, y, z); }
	CUDA_FUNC_IN    Vec3u           getXYW(void) const                { return Vec3u(x, y, w); }

	template <class V> CUDA_FUNC_IN Vec4u(const VectorBase<unsigned int, 4, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec4u& operator=(const VectorBase<unsigned int, 4, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec2f : public VectorBase<float, 2, Vec2f>, public float2
{
public:
	CUDA_FUNC_IN                    Vec2f(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec2f(float a)                     { set(a); }
	CUDA_FUNC_IN                    Vec2f(float xx, float yy)            { x = xx; y = yy; }
	CUDA_FUNC_IN                    Vec2f(const Vec2i& v)            { x = (float)v.x; y = (float)v.y; }

	CUDA_FUNC_IN    const float*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    float*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec2f       fromPtr(const float* ptr)            { return Vec2f(ptr[0], ptr[1]); }

	CUDA_FUNC_IN    operator Vec2i       (void) const                { return Vec2i((int)x, (int)y); }

	CUDA_FUNC_IN    Vec2f           perpendicular(void) const               { return Vec2f(-y, x); }
	CUDA_FUNC_IN    float             cross(const Vec2f& v) const      { return x * v.y - y * v.x; }

	template <class V> CUDA_FUNC_IN Vec2f(const VectorBase<float, 2, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec2f& operator=(const VectorBase<float, 2, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec3f : public VectorBase<float, 3, Vec3f>, public float3
{
public:
	CUDA_FUNC_IN                    Vec3f(void)                      {  }
	explicit CUDA_FUNC_IN           Vec3f(float a)                     { set(a); }
	CUDA_FUNC_IN                    Vec3f(float xx, float yy, float zz)    { x = xx; y = yy; z = zz; }
	CUDA_FUNC_IN                    Vec3f(const Vec2f& xy, float zz)   { x = xy.x; y = xy.y; z = zz; }
	CUDA_FUNC_IN                    Vec3f(const Vec3i& v)            { x = (float)v.x; y = (float)v.y; z = (float)v.z; }

	CUDA_FUNC_IN    const float*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    float*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec3f       fromPtr(const float* ptr)            { return Vec3f(ptr[0], ptr[1], ptr[2]); }

	CUDA_FUNC_IN    operator Vec3i       (void) const                { return Vec3i((int)x, (int)y, (int)z); }
	CUDA_FUNC_IN    Vec2f           getXY(void) const                { return Vec2f(x, y); }

	CUDA_FUNC_IN    Vec3f           cross(const Vec3f& v) const      { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

	template <class V> CUDA_FUNC_IN Vec3f(const VectorBase<float, 3, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec3f& operator=(const VectorBase<float, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

class Vec4f : public VectorBase<float, 4, Vec4f>, public float4
{
public:
	CUDA_FUNC_IN                    Vec4f(void)                      {  }
	explicit CUDA_FUNC_IN                    Vec4f(float a)                     { set(a); }
	CUDA_FUNC_IN                    Vec4f(float xx, float yy, float zz, float ww) { x = xx; y = yy; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4f(const Vec2f& xy, float zz, float ww) { x = xy.x; y = xy.y; z = zz; w = ww; }
	CUDA_FUNC_IN                    Vec4f(const Vec3f& xyz, float ww)  { x = xyz.x; y = xyz.y; z = xyz.z; w = ww; }
	CUDA_FUNC_IN                    Vec4f(const Vec2f& xy, const Vec2f& zw) { x = xy.x; y = xy.y; z = zw.x; w = zw.y; }
	CUDA_FUNC_IN                    Vec4f(const Vec4i& v)            { x = (float)v.x; y = (float)v.y; z = (float)v.z; w = (float)v.w; }

	CUDA_FUNC_IN    const float*      getPtr(void) const                { return &x; }
	CUDA_FUNC_IN    float*            getPtr(void)                      { return &x; }
	static CUDA_FUNC_IN Vec4f       fromPtr(const float* ptr)            { return Vec4f(ptr[0], ptr[1], ptr[2], ptr[3]); }

	CUDA_FUNC_IN    operator Vec4i       (void) const                { return Vec4i((int)x, (int)y, (int)z, (int)w); }
	CUDA_FUNC_IN    Vec2f           getXY(void) const                { return Vec2f(x, y); }
	CUDA_FUNC_IN    Vec3f           getXYZ(void) const                { return Vec3f(x, y, z); }
	CUDA_FUNC_IN    Vec3f           getXYW(void) const                { return Vec3f(x, y, w); }

	template <class V> CUDA_FUNC_IN Vec4f(const VectorBase<float, 4, V>& v) { set(v); }
	template <class V> CUDA_FUNC_IN Vec4f& operator=(const VectorBase<float, 4, V>& v) { set(v); return *this; }
};

template <class T, int L, class S> CUDA_FUNC_IN T lenSqr(const VectorBase<T, L, S>& v)                  { return v.lenSqr(); }
template <class T, int L, class S> CUDA_FUNC_IN T length(const VectorBase<T, L, S>& v)                  { return v.length(); }
template <class T, int L, class S> CUDA_FUNC_IN S normalize(const VectorBase<T, L, S>& v, T len = (T)1)    { return v.normalized(len); }
template <class T, int L, class S> CUDA_FUNC_IN T min(const VectorBase<T, L, S>& v)                  { return v.min(); }
template <class T, int L, class S> CUDA_FUNC_IN T max(const VectorBase<T, L, S>& v)                  { return v.max(); }
template <class T, int L, class S> CUDA_FUNC_IN T sum(const VectorBase<T, L, S>& v)                  { return v.sum(); }
template <class T, int L, class S> CUDA_FUNC_IN S abs(const VectorBase<T, L, S>& v)                  { return v.abs(); }

template <class T, int L, class S> CUDA_FUNC_IN S operator+     (const T& a, const VectorBase<T, L, S>& b)  { return b + a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator-     (const T& a, const VectorBase<T, L, S>& b)  { return -b + a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator*     (const T& a, const VectorBase<T, L, S>& b)  { return b * a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator/     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a / bp[i]; return r; }
template <class T, int L, class S> CUDA_FUNC_IN S operator%     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a % bp[i]; return r; }
template <class T, int L, class S> CUDA_FUNC_IN S operator&     (const T& a, const VectorBase<T, L, S>& b)  { return b & a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator|     (const T& a, const VectorBase<T, L, S>& b)  { return b | a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator^     (const T& a, const VectorBase<T, L, S>& b)  { return b ^ a; }
template <class T, int L, class S> CUDA_FUNC_IN S operator<<    (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a << bp[i]; return r; }
template <class T, int L, class S> CUDA_FUNC_IN S operator>>    (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a >> bp[i]; return r; }

template <class T, int L, class S, class V> CUDA_FUNC_IN T dot(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return a.dot(b); }
template <class T, int L, class S, class V> CUDA_FUNC_IN T absdot(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return abs(a.dot(b)); }
template <class T, int L, class S, class V> CUDA_FUNC_IN T distance(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return length(a - b); }
template <class T, int L, class S, class V> CUDA_FUNC_IN T distanceSquared(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return (a - b).lenSqr(); }

CUDA_FUNC_IN Vec2f  perpendicular(const Vec2f& v)                    { return v.perpendicular(); }
CUDA_FUNC_IN float    cross(const Vec2f& a, const Vec2f& b)    { return a.cross(b); }
CUDA_FUNC_IN Vec3f  cross(const Vec3f& a, const Vec3f& b)    { return a.cross(b); }

#define minmax(T) \
    CUDA_FUNC_IN T min(const T& a, const T& b)                          { return a.min(b); } \
    CUDA_FUNC_IN T min(T& a, T& b)                                      { return a.min(b); } \
    CUDA_FUNC_IN T max(const T& a, const T& b)                          { return a.max(b); } \
    CUDA_FUNC_IN T max(T& a, T& b)                                      { return a.max(b); } \
    CUDA_FUNC_IN T min(const T& a, const T& b, const T& c)              { return a.min(b).min(c); } \
    CUDA_FUNC_IN T min(T& a, T& b, T& c)                                { return a.min(b).min(c); } \
    CUDA_FUNC_IN T max(const T& a, const T& b, const T& c)              { return a.max(b).max(c); } \
    CUDA_FUNC_IN T max(T& a, T& b, T& c)                                { return a.max(b).max(c); } \
    CUDA_FUNC_IN T min(const T& a, const T& b, const T& c, const T& d)  { return a.min(b).min(c).min(d); } \
    CUDA_FUNC_IN T min(T& a, T& b, T& c, T& d)                          { return a.min(b).min(c).min(d); } \
    CUDA_FUNC_IN T max(const T& a, const T& b, const T& c, const T& d)  { return a.max(b).max(c).max(d); } \
    CUDA_FUNC_IN T max(T& a, T& b, T& c, T& d)                          { return a.max(b).max(c).max(d); } \
    CUDA_FUNC_IN T clamp(const T& v, const T& lo, const T& hi)          { return v.clamp(lo, hi); } \
    CUDA_FUNC_IN T clamp(T& v, T& lo, T& hi)                            { return v.clamp(lo, hi); } \
    CUDA_FUNC_IN T ceil(const T& v)										{ return v.ceil(); } \
    CUDA_FUNC_IN T ceil(T& v)											{ return v.ceil(); } \
    CUDA_FUNC_IN T floor(const T& v)									{ return v.floor(); } \
    CUDA_FUNC_IN T floor(T& v)											{ return v.floor(); } \
	CUDA_FUNC_IN T clamp01(const T& v)									{ return v.clamp(T(0), T(1)); } \
	CUDA_FUNC_IN T clamp01(T& v)										{ return v.clamp(T(0), T(1)); } \

minmax(Vec2i) minmax(Vec3i) minmax(Vec4i)
minmax(Vec2u) minmax(Vec3u) minmax(Vec4u)
minmax(Vec2f) minmax(Vec3f) minmax(Vec4f)

#define signABC(T) \
	CUDA_FUNC_IN T sign(const T& v)										{ return v.sign(); } \
	CUDA_FUNC_IN T sign(T& v)											{ return v.sign(); }

signABC(Vec2i) signABC(Vec3i) signABC(Vec4i)
signABC(Vec2f) signABC(Vec3f) signABC(Vec4f)

#undef minmax
#undef signABC