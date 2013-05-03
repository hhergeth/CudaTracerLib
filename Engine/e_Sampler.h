#pragma once

#include "e_Texture.h"

typedef char e_String[256];

template<typename T> struct e_Sampler
{
public:
	unsigned int m_uType;
	e_String m_sPath;
	T m_sValue;
	e_KernelTexture* m_pTex;
public:
	e_Sampler(const T& v)
	{
		m_uType = 0;
		m_sValue = v;
		memset(m_sPath, 0, sizeof(m_sPath));
	}

	e_Sampler(const T* v)
	{
		m_uType = 0;
		m_sValue = *v;
		memset(m_sPath, 0, sizeof(m_sPath));
	}

	e_Sampler(char* path, bool THIS_IS_REALLY_THE_RIGHT_CONSTRUCTOR_NO_FLOAT_0_SHIT)
	{
		m_uType = 1;
		memset(m_sPath, 0, sizeof(m_sPath));
		memcpy(m_sPath, path, strlen(path)); 
	}

	template<typename L> void LoadTextures(L callback)
	{
		if(m_uType)
			m_pTex = callback(m_sPath).getDevice();
	}

	CUDA_FUNC_IN T Sample(const float2& uv) const
	{
		return m_uType ? m_pTex->Sample<T>(uv) : m_sValue;
	}

	CUDA_FUNC_IN bool HasTexture() const
	{
		return m_uType;
	}
};