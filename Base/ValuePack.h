#pragma once
#include <Defines.h>
#include <tuple>
namespace CudaTracerLib {

//small variadic tuple style class for use with cuda
template<typename... ARGS> struct ValuePack {};

template<typename ARG, typename... ARGS> struct ValuePack<ARG, ARGS...> : ValuePack<ARGS...>
{
	ARG val;
	ValuePack<ARGS...> rest;

	CUDA_FUNC_IN ValuePack()
	{

	}

	CUDA_FUNC_IN ValuePack(const ARG& v, const ARGS&... vals)
		: ValuePack<ARGS...>(vals...), val(v)
	{

	}

	CUDA_FUNC_IN ValuePack(const ARG& v, const ValuePack<ARGS...>& rest)
		: ValuePack<ARGS...>(rest), val(v)
	{

	}

	CUDA_FUNC_IN float Sum() const
	{
		return (float)val + rest.Sum();
	}

	CUDA_FUNC_IN ValuePack<ARG, ARGS...> operator+(const ValuePack<ARG, ARGS...>& rhs) const
	{
		return ValuePack<ARG, ARGS...>(val + rhs.val, rest + rhs.rest);
	}

	CUDA_FUNC_IN ValuePack<ARG, ARGS...> operator-(const ValuePack<ARG, ARGS...>& rhs) const
	{
		return ValuePack<ARG, ARGS...>(val - rhs.val, rest - rhs.rest);
	}

	CUDA_FUNC_IN ValuePack<ARG, ARGS...> operator*(const ValuePack<ARG, ARGS...>& rhs) const
	{
		return ValuePack<ARG, ARGS...>(val * rhs.val, rest * rhs.rest);
	}

	CUDA_FUNC_IN ValuePack<ARG, ARGS...> operator*(float f) const
	{
		return ValuePack<ARG, ARGS...>((ARG)(val * f), rest * f);
	}

	CUDA_FUNC_IN static ValuePack<ARG, ARGS...> Zero()
	{
		return ValuePack<ARG, ARGS...>((ARG)0, ValuePack<ARGS...>::Zero());
	}
};

template<> struct ValuePack<>
{
	CUDA_FUNC_IN ValuePack()
	{

	}

	CUDA_FUNC_IN ValuePack<> operator+(const ValuePack<>& rhs) const
	{
		return ValuePack<>();
	}

	CUDA_FUNC_IN ValuePack<> operator-(const ValuePack<>& rhs) const
	{
		return ValuePack<>();
	}

	CUDA_FUNC_IN ValuePack<> operator*(const ValuePack<>& rhs) const
	{
		return ValuePack<>();
	}

	CUDA_FUNC_IN ValuePack<> operator*(float f) const
	{
		return ValuePack<>();
	}

	CUDA_FUNC_IN float Sum() const
	{
		return 0.0f;
	}

	CUDA_FUNC_IN static ValuePack<> Zero()
	{
		return ValuePack<>();
	}
};

namespace __value_pack_detail__	{

	template<int N, typename ARG, typename... ARGS> struct get_helper
	{
		typedef typename get_helper<N - 1, ARGS...>::RET_TYPE RET_TYPE;

		CUDA_FUNC_IN static RET_TYPE& get(ValuePack<ARG, ARGS...>& pack)
		{
			return get_helper<N - 1, ARGS...>::get(pack.rest);
		}

		CUDA_FUNC_IN static const RET_TYPE& get(const ValuePack<ARG, ARGS...>& pack)
		{
			return get_helper<N - 1, ARGS...>::get(pack.rest);
		}
	};

	template<typename ARG, typename... ARGS> struct get_helper<0, ARG, ARGS...>
	{
		typedef ARG RET_TYPE;

		CUDA_FUNC_IN static ARG& get(ValuePack<ARG, ARGS...>& pack)
		{
			return pack.val;
		}

		CUDA_FUNC_IN static const ARG& get(const ValuePack<ARG, ARGS...>& pack)
		{
			return pack.val;
		}
	};

	template<int I, typename T, typename... TYPES> struct Extractor
	{
		CUDA_FUNC_IN static typename __value_pack_detail__::get_helper<I - 1, TYPES...>::RET_TYPE get(T val, TYPES... vals)
		{
			return Extractor<I - 1, TYPES...>::get(vals...);
		}
	};

	template<typename T, typename... TYPES> struct Extractor<0, T, TYPES...>
	{
		CUDA_FUNC_IN static T get(T val, TYPES... vals)
		{
			return val;
		}
	};

}

template<int N, typename... ARGS> CUDA_FUNC_IN static typename __value_pack_detail__::get_helper<N, ARGS...>::RET_TYPE& get(ValuePack<ARGS...>& pack)
{
	return __value_pack_detail__::get_helper<N, ARGS...>::get(pack);
}

template<int N, typename... ARGS> CUDA_FUNC_IN static const typename __value_pack_detail__::get_helper<N, ARGS...>::RET_TYPE& get(const ValuePack<ARGS...>& pack)
{
	return __value_pack_detail__::get_helper<N, ARGS...>::get(pack);
}

template<int N, typename T> static CUDA_FUNC_IN T extract_val(std::initializer_list<T> l)
{
	return *(l.begin() + N);
}

//needed for template parameter lists as macro arguments
#define PP_COMMA ,

#define DCL_TUPLE_REF_STRUCT(TEMP, NAME, TUPLE_TYPE) \
	TEMP struct NAME : TUPLE_TYPE \
	{ \
		typedef TUPLE_TYPE PACK_TYPE; \
		NAME() \
		{ \
		} \
		NAME(const TUPLE_TYPE& r) \
			: TUPLE_TYPE(r) \
		{ \
		}

#define DCL_TUPLE_REF_MEMBER(NAME, ID) \
	auto& NAME() {return get<ID>(*this); }

}