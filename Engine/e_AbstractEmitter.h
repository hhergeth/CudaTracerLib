#pragma once

#include <MathTypes.h>
#include "../VirtualFuncType.h"

enum EEmitterType {
	/// Emission profile contains a Dirac delta term with respect to direction
	EDeltaDirection = 0x01,

	/// Emission profile contains a Dirac delta term with respect to position
	EDeltaPosition = 0x02,

	/// Is the emitter associated with a surface in the scene?
	EOnSurface = 0x04,

	/// Is this an environment emitter, such as a HDRI map?
	EEnvironmentEmitter = 0x010,
};

struct e_AbstractEmitter : public e_BaseType
{
	unsigned int m_Type;

	e_AbstractEmitter(unsigned int type)
		: m_Type(type)
	{

	}

	CUDA_FUNC_IN bool IsDegenerate() const
	{
		return (m_Type & (EDeltaPosition | EDeltaDirection)) != 0;
	}

	CUDA_FUNC_IN bool IsOnSurface() const
	{
		return (m_Type & EOnSurface) != 0;
	}
};