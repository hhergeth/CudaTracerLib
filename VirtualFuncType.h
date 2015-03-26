#pragma once

#include "Defines.h"
#include <array>

namespace CTVirtualHelper
{
	// STRUCT _Tuple_alloc_t
	struct _Tuple_alloc_t
	{	// tag type to disambiguate added allocator argument
	};

	const _Tuple_alloc_t _Tuple_alloc = _Tuple_alloc_t();

	// TEMPLATE CLASS _Tuple_val
	template<class _Ty>
	struct _Tuple_val
	{	// stores each value in a tuple
		CUDA_FUNC_IN _Tuple_val()
			: _Val()
		{	// default construct
		}

		template<class _Other>
		CUDA_FUNC_IN _Tuple_val(_Other&& _Arg)
			: _Val(_STD forward<_Other>(_Arg))
		{	// construct with argument
		}

		template<class _Other>
		CUDA_FUNC_IN _Tuple_val& operator=(_Other&& _Right)
		{	// assign
			_Val = _STD forward<_Other>(_Right);
			return (*this);
		}

		template<class _Alloc,
		class... _Other>
			CUDA_FUNC_IN _Tuple_val(const _Alloc&,
			typename std::enable_if<!std::uses_allocator<_Ty, _Alloc>::value,
			_Tuple_alloc_t>::type, _Other&&... _Arg)
			: _Val(_STD forward<_Other>(_Arg)...)
		{	// construct with optional arguments, no allocator
		}

		template<class _Alloc,
		class... _Other>
			CUDA_FUNC_IN _Tuple_val(const _Alloc& _Al,
			typename std::enable_if<std::uses_allocator<_Ty, _Alloc>::value
			&& std::is_constructible<_Ty,
			std::allocator_arg_t, _Alloc>::value,
			_Tuple_alloc_t>::type, _Other&&... _Arg)
			: _Val(std::allocator_arg, _Al, _STD forward<_Other>(_Arg)...)
		{	// construct with optional arguments, leading allocator
		}

		template<class _Alloc,
		class... _Other>
			CUDA_FUNC_IN _Tuple_val(const _Alloc& _Al,
			typename std::enable_if<std::uses_allocator<_Ty, _Alloc>::value
			&& !std::is_constructible<_Ty,
			std::allocator_arg_t, _Alloc>::value,
			_Tuple_alloc_t>::type, _Other&&... _Arg)
			: _Val(_STD forward<_Other>(_Arg)..., _Al)
		{	// construct with optional arguments, trailing allocator
		}


		_Ty _Val;
	};

	// CLASS tuple
	template<class... _Types>
	class tuple;

	template<>
	class tuple<>
	{	// empty tuple
	public:
		typedef tuple<> _Myt;

		CUDA_FUNC_IN tuple()
		{	// default construct
		}

		template<class _Alloc>
		CUDA_FUNC_IN tuple(std::allocator_arg_t, const _Alloc&) _NOEXCEPT
		{	// default construct, allocator
		}

		CUDA_FUNC_IN tuple(const tuple&) _NOEXCEPT
		{	// copy construct
		}

		template<class _Alloc>
		CUDA_FUNC_IN tuple(std::allocator_arg_t, const _Alloc&, const tuple&) _NOEXCEPT
		{	// copy construct, allocator
		}

		CUDA_FUNC_IN void swap(_Myt&) _NOEXCEPT
		{	// swap elements
		}

		CUDA_FUNC_IN bool _Equals(const _Myt&) const _NOEXCEPT
		{	// test if *this == _Right
			return (true);
		}

			CUDA_FUNC_IN bool _Less(const _Myt&) const _NOEXCEPT
		{	// test if *this < _Right
			return (false);
		}
	};

	// STRUCT _Tuple_enable
	template<class _Src,
	class _Dest>
	struct _Tuple_enable
	{	// default has no type definition
	};

	template<>
	struct _Tuple_enable<tuple<>, tuple<> >
	{	// empty tuples match
		typedef void ** type;
	};

	template<class _Src0,
	class... _Types1,
	class _Dest0,
	class... _Types2>
	struct _Tuple_enable<tuple<_Src0, _Types1...>,
		tuple<_Dest0, _Types2...> >
		: std::_If<std::is_convertible<_Src0, _Dest0>::value,
		_Tuple_enable<tuple<_Types1...>, tuple<_Types2...> >,
		_Tuple_enable<int, int>
		>::type
	{	// tests if all tuple element pairs are implicitly convertible
	};

	template<class _This,
	class... _Rest>
	class tuple<_This, _Rest...>
		: private tuple<_Rest...>
	{	// recursive tuple definition
	public:
		typedef _This _This_type;
		typedef tuple<_This, _Rest...> _Myt;
		typedef tuple<_Rest...> _Mybase;
		static const size_t _Mysize = 1 + sizeof...(_Rest);

		CUDA_FUNC_IN tuple()
			: _Mybase(),
			_Myfirst()
		{	// construct default
		}

		template<class _This2,
		class... _Rest2,
		class = typename _Tuple_enable<
			tuple<_This2, _Rest2...>, _Myt>::type>
			CUDA_FUNC_IN explicit tuple(_This2&& _This_arg, _Rest2&&... _Rest_arg)
			: _Mybase(_STD forward<_Rest2>(_Rest_arg)...),
			_Myfirst(_STD forward<_This2>(_This_arg))
		{	// construct from one or more moved elements
		}

		CUDA_FUNC_IN _Myt& operator=(const _Myt& _Right)
		{	// assign
			_Myfirst._Val = _Right._Myfirst._Val;
			(_Mybase&)*this = _Right._Get_rest();
			return (*this);
		}

		_Tuple_val<_This> _Myfirst;	// the stored element
	};

	// CLASS tuple_element
	template<size_t _Index,
	class _Tuple>
	struct tuple_element;

	template<class _This,
	class... _Rest>
	struct tuple_element<0, tuple<_This, _Rest...> >
	{	// select first element
		typedef _This type;
		typedef typename std::add_lvalue_reference<const _This>::type _Ctype;
		typedef typename std::add_lvalue_reference<_This>::type _Rtype;
		typedef typename std::add_rvalue_reference<_This>::type _RRtype;
		typedef tuple<_This, _Rest...> _Ttype;
	};

	template<size_t _Index,
	class _This,
	class... _Rest>
	struct tuple_element<_Index, tuple<_This, _Rest...> >
		: public tuple_element<_Index - 1, tuple<_Rest...> >
	{	// recursive tuple_element definition
	};


	template<size_t _Index,
	class _Tuple>
	struct tuple_element<_Index, const _Tuple>
		: public tuple_element<_Index, _Tuple>
	{	// tuple_element for const
		typedef tuple_element<_Index, _Tuple> _Mybase;
		typedef typename std::add_const<typename _Mybase::type>::type type;
	};

	template<size_t _Index,
	class _Tuple>
	struct tuple_element<_Index, volatile _Tuple>
		: public tuple_element<_Index, _Tuple>
	{	// tuple element for volatile
		typedef tuple_element<_Index, _Tuple> _Mybase;
		typedef typename std::add_volatile<typename _Mybase::type>::type type;
	};

	template<size_t _Index,
	class _Tuple>
	struct tuple_element<_Index, const volatile _Tuple>
		: public tuple_element<_Index, _Tuple>
	{	// tuple_element for const volatile
		typedef tuple_element<_Index, _Tuple> _Mybase;
		typedef typename std::add_cv<typename _Mybase::type>::type type;
	};

	template <int... Is> struct index {};
	template <int N, int... Is>	struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};
	template <int... Is> struct gen_seq<0, Is...> : index<Is...>{};
	template<typename T, typename... REST> struct Unifier
	{
		enum { result = Dmax2(sizeof(T), Unifier<REST...>::result) };
	};

	template<typename T> struct Unifier<T>
	{
		enum { result = sizeof(T) };
	};

	template<size_t _Index,
	class... _Types> CUDA_FUNC_IN
		typename CTVirtualHelper::tuple_element<_Index, CTVirtualHelper::tuple<_Types...> >::_Rtype
		get(CTVirtualHelper::tuple<_Types...>& _Tuple)
	{	// get reference to _Index element of tuple
		typedef typename CTVirtualHelper::tuple_element<_Index, CTVirtualHelper::tuple<_Types...> >::_Ttype
			_Ttype;
		return (((_Ttype&)_Tuple)._Myfirst._Val);
	}

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

	template<class... _Types> CUDA_FUNC_IN
		CTVirtualHelper::tuple<typename std::_Unrefwrap<_Types>::type...>
		make_tuple(_Types&&... _Args)
	{	// make tuple from elements
		typedef CTVirtualHelper::tuple<typename std::_Unrefwrap<_Types>::type...> _Ttype;
		return (_Ttype(forward<_Types>(_Args)...));
	}
}

struct CudaVirtualClass
{
public:
	virtual void Update()
	{

	}
};

template<int ID> struct CudaVirtualClassHelper
{
	CUDA_FUNC_IN static int TYPEID()
	{
		return ID;
	}
};

template<typename BaseType, typename... Types> struct CudaVirtualAggregate
{
private:
	unsigned int type;
	unsigned char data[CTVirtualHelper::Unifier<Types...>::result];
	template<typename Functor, typename CLASS> CUDA_FUNC_IN void CallInternal(Functor& f) const
	{
		if (type == CLASS::TYPEID())
			f(As<CLASS>());
	}
	template<typename Functor, typename CLASS, typename CLASS2, typename... REST> CUDA_FUNC_IN void CallInternal(Functor& f) const
	{
		if (type == CLASS::TYPEID())
			f(As<CLASS>());
		else CallInternal<Functor, CLASS2, REST...>(f);
	}

	template<typename T, typename CLASS> CUDA_FUNC_IN bool isDerived() const
	{
		return T::Type() == CLASS::TYPEID();
	}
	template<typename T, typename CLASS, typename CLASS2, typename... REST> CUDA_FUNC_IN bool isDerived() const
	{
		if (T::Type() == CLASS::TYPEID())
			return true;
		else return isDerived<T, CLASS2, REST...>();
	}

	template<typename CLASS> void SetVtable()
	{
		if (type == CLASS::TYPEID())
		{
			CLASS obj;
			uintptr_t* vftable = (uintptr_t*)&obj;
			uintptr_t* vftable_tar = (uintptr_t*)data;
			*vftable_tar = *vftable;
		}
	}
	template<typename CLASS, typename CLASS2, typename... REST> void SetVtable()
	{
		SetVtable<CLASS>();//do the work
		SetVtable<CLASS2, REST...>();
	}
protected:
	template<typename Functor> CUDA_FUNC_IN void Call(Functor& f) const
	{
		CallInternal<Functor, Types...>(f);
	}
public:
	template<typename SpecializedType> CUDA_FUNC_IN SpecializedType* As() const
	{
		return (SpecializedType*)data;
	}

	template<typename SpecializedType> CUDA_FUNC_IN void SetData(const SpecializedType& val)
	{
		memcpy(data, &val, sizeof(SpecializedType));
		type = SpecializedType::TYPEID();
	}

	template<typename SpecializedType> CUDA_FUNC_IN bool Is() const
	{
		return type == SpecializedType::TYPEID();
	}

	CUDA_FUNC_IN BaseType* As() const
	{
		return As<BaseType>();
	}

	template<typename SpecializedType>  CUDA_FUNC_IN bool IsBase() const
	{
		return isDerived<SpecializedType, Types...>();
	}

	void SetVtable()
	{
		SetVtable<Types...>();
	}
};

#define VIRTUAL_CALLER_RET(FUNC_NAME, TYPE_NAME) \
	template<typename ReturnType, typename... ARGUMENTS> struct TYPE_NAME \
	{ \
	private: \
		template <typename T, typename... Args, int... Is> CUDA_FUNC_IN \
			void func(T* obj, CTVirtualHelper::tuple<Args...>& tup, CTVirtualHelper::index<Is...>) \
		{ \
			ret = obj->FUNC_NAME(CTVirtualHelper::get<Is>(tup)...); \
		} \
	public: \
		CTVirtualHelper::tuple<ARGUMENTS...> args; \
		ReturnType ret; \
		CUDA_FUNC_IN TYPE_NAME(const ARGUMENTS&... args) \
			: args(CTVirtualHelper::make_tuple(args...)) \
		{ \
		} \
		template<typename T> CUDA_FUNC_IN void operator()(T* obj) \
		{ \
			func(obj, args, CTVirtualHelper::gen_seq<sizeof...(ARGUMENTS)>{}); \
		} \
	};