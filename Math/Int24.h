#pragma once

#include <Defines.h>
#include <iostream>

namespace CudaTracerLib {

const int INT24_MAX = 8388607;

//3 byte signed integer datatype
//http://stackoverflow.com/questions/2682725/int24-24-bit-integral-datatype
struct Int24
{
protected:
	unsigned char m_Internal[3];
public:
	CUDA_FUNC_IN Int24()
	{
	}

	CUDA_FUNC_IN Int24(const int val)
	{
		*this = val;
	}

	CUDA_FUNC_IN Int24(const Int24& val)
	{
		*this = val;
	}

	CUDA_FUNC_IN operator int() const
	{
		if (m_Internal[2] & 0x80) // Is this a negative?  Then we need to siingn extend.
		{
			return (0xff << 24) | (m_Internal[2] << 16) | (m_Internal[1] << 8) | (m_Internal[0] << 0);
		}
		else
		{
			return (m_Internal[2] << 16) | (m_Internal[1] << 8) | (m_Internal[0] << 0);
		}
	}

	CUDA_FUNC_IN operator float() const
	{
		return (float)this->operator int();
	}

	CUDA_FUNC_IN Int24& operator =(const Int24& input)
	{
		m_Internal[0] = input.m_Internal[0];
		m_Internal[1] = input.m_Internal[1];
		m_Internal[2] = input.m_Internal[2];

		return *this;
	}

	CUDA_FUNC_IN Int24& operator =(const int input)
	{
		m_Internal[0] = ((unsigned char*)&input)[0];
		m_Internal[1] = ((unsigned char*)&input)[1];
		m_Internal[2] = ((unsigned char*)&input)[2];

		return *this;
	}

	/***********************************************/

	CUDA_FUNC_IN Int24 operator +(const Int24& val) const
	{
		return Int24((int)*this + (int)val);
	}

	CUDA_FUNC_IN Int24 operator -(const Int24& val) const
	{
		return Int24((int)*this - (int)val);
	}

	CUDA_FUNC_IN Int24 operator *(const Int24& val) const
	{
		return Int24((int)*this * (int)val);
	}

	CUDA_FUNC_IN Int24 operator /(const Int24& val) const
	{
		return Int24((int)*this / (int)val);
	}

	/***********************************************/

	CUDA_FUNC_IN Int24 operator +(const int val) const
	{
		return Int24((int)*this + val);
	}

	CUDA_FUNC_IN Int24 operator -(const int val) const
	{
		return Int24((int)*this - val);
	}

	CUDA_FUNC_IN Int24 operator *(const int val) const
	{
		return Int24((int)*this * val);
	}

	CUDA_FUNC_IN Int24 operator /(const int val) const
	{
		return Int24((int)*this / val);
	}

	/***********************************************/
	/***********************************************/


	CUDA_FUNC_IN Int24& operator +=(const Int24& val)
	{
		*this = *this + val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator -=(const Int24& val)
	{
		*this = *this - val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator *=(const Int24& val)
	{
		*this = *this * val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator /=(const Int24& val)
	{
		*this = *this / val;
		return *this;
	}

	/***********************************************/

	CUDA_FUNC_IN Int24& operator +=(const int val)
	{
		*this = *this + val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator -=(const int val)
	{
		*this = *this - val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator *=(const int val)
	{
		*this = *this * val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator /=(const int val)
	{
		*this = *this / val;
		return *this;
	}

	/***********************************************/
	/***********************************************/

	CUDA_FUNC_IN Int24 operator >>(const int val) const
	{
		return Int24((int)*this >> val);
	}

	CUDA_FUNC_IN Int24 operator <<(const int val) const
	{
		return Int24((int)*this << val);
	}

	/***********************************************/

	CUDA_FUNC_IN Int24& operator >>=(const int val)
	{
		*this = *this >> val;
		return *this;
	}

	CUDA_FUNC_IN Int24& operator <<=(const int val)
	{
		*this = *this << val;
		return *this;
	}

	/***********************************************/
	/***********************************************/

	CUDA_FUNC_IN operator bool() const
	{
		return (int)*this != 0;
	}

	CUDA_FUNC_IN bool operator !() const
	{
		return !((int)*this);
	}

	CUDA_FUNC_IN Int24 operator -()
	{
		return Int24(-(int)*this);
	}

	/***********************************************/
	/***********************************************/

	CUDA_FUNC_IN bool operator ==(const Int24& val) const
	{
		return (int)*this == (int)val;
	}

	CUDA_FUNC_IN bool operator !=(const Int24& val) const
	{
		return (int)*this != (int)val;
	}

	CUDA_FUNC_IN bool operator >=(const Int24& val) const
	{
		return (int)*this >= (int)val;
	}

	CUDA_FUNC_IN bool operator <=(const Int24& val) const
	{
		return (int)*this <= (int)val;
	}

	CUDA_FUNC_IN bool operator >(const Int24& val) const
	{
		return (int)*this > (int)val;
	}

	CUDA_FUNC_IN bool operator <(const Int24& val) const
	{
		return (int)*this < (int)val;
	}

	/***********************************************/

	CUDA_FUNC_IN bool operator ==(const int val) const
	{
		return (int)*this == val;
	}

	CUDA_FUNC_IN bool operator !=(const int val) const
	{
		return (int)*this != val;
	}

	CUDA_FUNC_IN  bool operator >=(const int val) const
	{
		return (int)*this >= val;
	}

	CUDA_FUNC_IN bool operator <=(const int val) const
	{
		return (int)*this <= val;
	}

	CUDA_FUNC_IN bool operator >(const int val) const
	{
		return ((int)*this) > val;
	}

	CUDA_FUNC_IN bool operator <(const int val) const
	{
		return (int)*this < val;
	}

	/***********************************************/
	/***********************************************/

	friend std::ostream& operator<< (std::ostream & os, const Int24& rhs)
	{
		os << rhs.operator int();
		return os;
	}
};

}