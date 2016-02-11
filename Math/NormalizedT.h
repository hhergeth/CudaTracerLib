#pragma once

#include <Defines.h>
#include <type_traits>
#include <iostream>

namespace CudaTracerLib {

class IVectorBase;
template<typename T, class Enable> struct NormalizedT : public T
{
	CUDA_FUNC_IN explicit NormalizedT(const T& v)
		: T(v)
	{

	}

	template<typename... V> CUDA_FUNC_IN explicit NormalizedT(V... vals)
		: T(vals...)
	{

	}
};

template<typename T> inline std::ostream& operator << (std::ostream & stream, const NormalizedT<T>& v)
{
	stream << "|" << v << "|";
	return stream;
}

template<typename VEC> struct NormalizedT<VEC, typename std::enable_if<std::is_base_of<IVectorBase, VEC>::value>::type> : public VEC
{
	typedef typename VEC::SCALAR_TYPE  T;
	typedef typename VEC::STORAGE_TYPE S;

	CUDA_FUNC_IN explicit NormalizedT(const VEC& v)
		: VEC(v)
	{
		/*int L = VEC::DIMENSION;
		float s = v.lenSqr();
		if (math::abs(s - 1) > 0.1f)
		{
			if (L == 2)
				printf("Trying to set unnormalized vector = {%f, %f}\n", v[0], v[1]);
			else if (L == 3)
				printf("Trying to set unnormalized vector = {%f, %f, %f}\n", v[0], v[1], v[2]);
			else printf("Invalid Vec\n");
		}*/
	}

	template<typename... V> CUDA_FUNC_IN explicit NormalizedT(V... vals)
		: VEC(vals...)
	{

	}

	T lenSqr(void) const = delete;
	T length(void) const = delete;
	S normalized() const = delete;

	S& operator+=  (const T& a) = delete;
	S& operator-=  (const T& a) = delete;
	S& operator*=  (const T& a) = delete;
	S& operator/=  (const T& a) = delete;
	S& operator%=  (const T& a) = delete;
	S& operator&=  (const T& a) = delete;
	S& operator|=  (const T& a) = delete;
	S& operator^=  (const T& a) = delete;
	S& operator<<= (const T& a) = delete;
	S& operator>>= (const T& a) = delete;

	template <class V, int L> S& operator+=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator-=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator*=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator/=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator%=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator&=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator|=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator^=  (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator<<= (const VectorBase<T, L, V>& v) = delete;
	template <class V, int L> S& operator>>= (const VectorBase<T, L, V>& v) = delete;
};

template<typename VEC> NormalizedT<VEC> CUDA_FUNC_IN operator - (const NormalizedT<VEC>& in)
{
	return NormalizedT<VEC>(-(VEC)in);
}

template<typename VEC> CUDA_FUNC_IN NormalizedT<VEC> normalize(const NormalizedT<VEC>& v)
{
	static_assert(sizeof(VEC) == 0, "normalize is not necessary for normalized vector!");
	return v;
}

template<typename VEC> CUDA_FUNC_IN typename VEC::SCALAR_TYPE length(const NormalizedT<VEC>& v)
{
	static_assert(sizeof(VEC) == 0, "length of normalized vector := 1!");
	return (typename VEC::SCALAR_TYPE)1;
}

template<typename VEC> CUDA_FUNC_IN typename VEC::SCALAR_TYPE lenSqr(const NormalizedT<VEC>& v)
{
	static_assert(sizeof(VEC) == 0, "lenSqr of normalized vector := 1!");
	return (typename VEC::SCALAR_TYPE)1;
}

}
