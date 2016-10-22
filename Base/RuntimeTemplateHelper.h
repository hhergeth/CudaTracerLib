#pragma once
#include <Defines.h>
#include <tuple>

namespace CudaTracerLib {

namespace __iterateTypes__detail__
{
template<typename T, typename... Types> struct it
{
	template<typename X, typename F> static void f(const X* obj, F& clb)
	{
		it<T>::f(obj, clb);
		it<Types...>::f(obj, clb);
	}
};

template<typename T> struct it<T>
{
	template<typename X, typename F> static void f(const X* obj, F& clb)
	{
		auto* obj_t = dynamic_cast<const T*>(obj);
		if (obj_t)
			clb((const T*)obj_t);
	}
};
}

template<typename... Types, typename X, typename F> void iterateTypes(const X* obj, F& clb)
{
	__iterateTypes__detail__::it<Types...>::f(obj, clb);
}

}