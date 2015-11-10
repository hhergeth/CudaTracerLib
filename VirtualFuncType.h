#pragma once
#include <type_traits>
#include <memory>

#define TYPE_FUNC(id) \
	static CUDA_FUNC_IN unsigned int TYPE() \
		{ \
		return id; \
		}

struct e_BaseType
{
	virtual void Update()
	{
	}
};

namespace CTVirtualHelper
{
	template<typename T, typename... REST> struct Unifier
	{
		enum { result = Dmax2(sizeof(T), Unifier<REST...>::result) };
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

#define CALLER(FUNC_NAME) \
	private: \
	template<typename R, typename T, typename... Args> CUDA_FUNC_IN R FUNC_NAME##ABC(CTVirtualHelper::Typer<Args...> typ, Args&&... args) const \
	{ \
		if (this->Is<T>()) \
			return this->As<T>()->FUNC_NAME(args...); \
		return R(); \
	} \
	template<typename R, typename T, typename T2, typename... REST, typename... Args> CUDA_FUNC_IN R FUNC_NAME##ABC(CTVirtualHelper::Typer<Args...> typ, Args&&... args) const \
	{ \
		if (this->Is<T>()) \
			return this->As<T>()->FUNC_NAME(args...); \
		return FUNC_NAME##ABC<R, T2, REST...>(CTVirtualHelper::Typer<Args...>(), CTVirtualHelper::forward<Args>(args)...); \
	} \
	template<typename R, template <class, class...> class A, class BaseType, class... Types, typename... Args> CUDA_FUNC_IN R FUNC_NAME##_Caller(const A<BaseType, Types...>& obj, Args&&... args) const \
	{ \
		return FUNC_NAME##ABC<R, Types...>(CTVirtualHelper::Typer<Args...>(), CTVirtualHelper::forward<Args>(args)...); \
	} \
	public:

template<typename BaseType, typename... Types> struct CudaVirtualAggregate
{
	enum { DATA_SIZE = CTVirtualHelper::Unifier<Types...>::result };

	static_assert(DATA_SIZE > 0, "CudaVirtualAggregate::Data too  small.");
	static_assert(DATA_SIZE < 2048, "CudaVirtualAggregate::Data too large.");
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
#ifndef __CUDACC__
	CUDA_FUNC_IN CudaVirtualAggregate()
	{
		memset(this, 0, sizeof(*this));
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