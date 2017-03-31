#pragma once
#include <type_traits>
#include <memory>
#include <string.h>
#include "Platform.h"

namespace CudaTracerLib {

namespace CTVirtualHelper
{
	//http://stackoverflow.com/a/107657/1715849
	CUDA_FUNC_IN constexpr unsigned int uint32HashName(const char* name)
	{
		return name[0] != '\0' ? name[0] + uint32HashName(name + 1) * 101 : 0;
	}

	template<typename T, typename... REST> struct Unifier
	{
		enum { result = DMAX2(sizeof(T), Unifier<REST...>::result) };
	};

	template<typename T> struct Unifier<T>
	{
		enum { result = sizeof(T) };
	};

	//http://stackoverflow.com/questions/2118541/check-if-c0x-parameter-pack-contains-a-type
	template < typename Tp, typename... List >
	struct contains : std::true_type {};

	template < typename Tp, typename Head, typename... Rest >
	struct contains<Tp, Head, Rest...>
		: std::conditional< std::is_same<Tp, Head>::value,
		std::true_type,
		contains<Tp, Rest...>
		>::type{};

	template < typename Tp >
	struct contains<Tp> : std::false_type{};

	template<class _Ty> CUDA_FUNC_IN
		_Ty&& forward(typename std::remove_reference<_Ty>::type& _Arg)
	{	// forward an lvalue
		return (static_cast<_Ty&&>(_Arg));
	}

	template<class _Ty> CUDA_FUNC_IN
		_Ty&& forward(typename std::remove_reference<_Ty>::type&& _Arg)
	{	// forward anything
		return (static_cast<_Ty&&>(_Arg));
	}


	template<class... Types> class Typer
	{
	public:
		CUDA_FUNC_IN Typer()
		{
		}
	};

	template<typename T, typename... ARGS> CUDA_FUNC_IN static bool Contains_Type()
	{
		return contains< T, ARGS... >::value;
	}

	template<typename T, typename... ARGS> CUDA_FUNC_IN static void Check_Type()
	{
		static_assert(contains< T, ARGS... >::value, "Type not in type list!");
	}
}

#define TYPE_FUNC(id) \
	CUDA_FUNC_IN static constexpr unsigned int TYPE() \
	{ \
		return id; \
	}

struct BaseType
{
	virtual ~BaseType()
	{

	}
	virtual void Update()
	{
	}
};

#define CALLER(FUNC_NAME) \
	private: \
	struct FUNC_NAME##_Helper \
	{ \
		template<typename R, typename T, typename... Args, typename AGG> CUDA_FUNC_IN static R FUNC_NAME##__INTERNAL__CALLER(const AGG* obj, CTVirtualHelper::Typer<Args...> typ, Args&&... args) \
		{ \
			if (obj->Is<T>()) \
				return obj->As<T>()->FUNC_NAME(args...); \
			return R(); \
		} \
		template<typename R, typename T, typename T2, typename... REST, typename... Args, typename AGG> CUDA_FUNC_IN static R FUNC_NAME##__INTERNAL__CALLER(const AGG* obj, CTVirtualHelper::Typer<Args...> typ, Args&&... args) \
		{ \
			if (obj->Is<T>()) \
				return obj->As<T>()->FUNC_NAME(args...); \
			return FUNC_NAME##__INTERNAL__CALLER<R, T2, REST...>(obj, CTVirtualHelper::Typer<Args...>(), CTVirtualHelper::forward<Args>(args)...); \
		} \
		template<typename R, class BaseType, class... Types, typename... Args> CUDA_FUNC_IN static R Caller(const CudaVirtualAggregate<BaseType, Types...>* obj, Args&&... args) \
		{ \
			return FUNC_NAME##__INTERNAL__CALLER<R, Types...>(obj, CTVirtualHelper::Typer<Args...>(), CTVirtualHelper::forward<Args>(args)...); \
		} \
	}; \
	public:

template<typename BaseType, typename... Types> struct CudaVirtualAggregate
{
	enum { DATA_SIZE = CTVirtualHelper::Unifier<Types...>::result };

	static_assert(DATA_SIZE > 0, "CudaVirtualAggregate::Data too  small.");
	static_assert(DATA_SIZE < 4096, "CudaVirtualAggregate::Data too large.");
protected:
	unsigned int type;
	CUDA_ALIGN(16) unsigned char Data[DATA_SIZE];

	template<typename CLASS> void SetVtable()
	{
		if (type == CLASS::TYPE())
		{
			CLASS obj;
			uintptr_t* vftable = (uintptr_t*)&obj;
			uintptr_t* vftable_tar = (uintptr_t*)Data;
			*vftable_tar = *vftable;
		}
	}
	template<typename CLASS, typename CLASS2, typename... REST> void SetVtable()
	{
		SetVtable<CLASS>();//do the work
		SetVtable<CLASS2, REST...>();
	}
public:

	//cannot use ISCUDA due to nvcc believing a non default constructor exists
#ifndef __CUDACC__
	CUDA_FUNC_IN CudaVirtualAggregate()
	{
		Platform::SetMemory(this, sizeof(*this));
	}
#endif

	template<typename SpecializedType> CUDA_FUNC_IN SpecializedType* As() const
	{
		CTVirtualHelper::Check_Type<SpecializedType, BaseType, Types...>();
		return (SpecializedType*)Data;
	}

	template<typename SpecializedType> CUDA_FUNC_IN void SetData(const SpecializedType& val)
	{
		CTVirtualHelper::Check_Type<SpecializedType, Types...>();
		memcpy(Data, &val, sizeof(SpecializedType));
		type = SpecializedType::TYPE();
	}

	template<typename SpecializedType> CUDA_FUNC_IN bool Is() const
	{
		CTVirtualHelper::Check_Type<SpecializedType, Types...>();
		return type == SpecializedType::TYPE();
	}

	CUDA_FUNC_IN BaseType* As() const
	{
		return As<BaseType>();
	}

	template<typename SpecializedType>  CUDA_FUNC_IN bool IsBase() const
	{
		return CTVirtualHelper::Contains_Type<SpecializedType, Types...>();
	}

	CUDA_FUNC_IN unsigned int getTypeToken() const
	{
		return type;
	}

	CUDA_FUNC_IN void setTypeToken(unsigned int t)
	{
		type = t;
	}

	void SetVtable()
	{
		SetVtable<Types...>();
	}
};

template<typename AGGREGATE_TYPE, typename SPECIALIZED_TYPE> AGGREGATE_TYPE CreateAggregate(const SPECIALIZED_TYPE& val)
{
	AGGREGATE_TYPE agg;
	agg.SetData(val);//the compiler will check that only applicable types are accepted
	return agg;
}

}
