#pragma once

#include "e_FileTexture.h"
#include "e_DifferentialGeometry.h"
#include "e_Buffer.h"
#include "../Base/FixedString.h"

struct e_KernelTextureMapping2D
{
	e_KernelTextureMapping2D(float su = 1, float sv = 1, float du = 0, float dv = 0, int setId = 0)
	{
		this->su = su;
		this->sv = sv;
		this->du = du;
		this->dv = dv;
		this->setId = setId;
	}
	CUDA_FUNC_IN Vec2f Map(const DifferentialGeometry& map) const
	{
		float u = su * map.uv[setId].x + du;
		float v = sv * map.uv[setId].y + dv;
		return Vec2f(u, v);
	}
	float su, sv, du, dv;
	int setId;
};

struct e_KernelTextureBase : public e_BaseType, public e_BaseTypeHelper<5784916>
{
};

struct e_KernelBilerpTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<1>
{
	e_KernelBilerpTexture(){}
	e_KernelBilerpTexture(const e_KernelTextureMapping2D& m, const Spectrum &t00, const Spectrum &t01, const Spectrum &t10, const Spectrum &t11)
	{
		mapping = m;
		v00 = t00;
		v01 = t01;
		v10 = t10;
		v11 = t11;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		Vec2f uv = mapping.Map(map);
		return (1-uv.x)*(1-uv.y) * v00 + (1-uv.x)*(  uv.y) * v01 +
               (  uv.x)*(1-uv.y) * v10 + (  uv.x)*(  uv.y) * v11;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return (v00+v01+v10+v11) * 0.25f;
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	e_KernelTextureMapping2D mapping;
	Spectrum v00, v01, v10, v11;
};

struct e_KernelConstantTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<2>
{
	e_KernelConstantTexture(){}
	e_KernelConstantTexture(const Spectrum& v)
		: val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		return val;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	Spectrum val;
};

struct e_KernelCheckerboardTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<3>
{
	e_KernelCheckerboardTexture(){}
	e_KernelCheckerboardTexture(const Spectrum& u, const Spectrum& v, const e_KernelTextureMapping2D& m)
		: val0(u), val1(v), mapping(m)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		Vec2f uv = mapping.Map(map);
		int x = 2*math::modulo((int) (uv.x * 2), 2) - 1,
			y = 2*math::modulo((int) (uv.y * 2), 2) - 1;

		if (x*y == 1)
			return val0;
		else
			return val1;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return (val0 + val1) / 2.0f;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	Spectrum val0, val1;
	e_KernelTextureMapping2D mapping;
};

struct e_KernelImageTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<4>
{
	e_KernelImageTexture(){}
	e_KernelImageTexture(const e_KernelTextureMapping2D& m, const char* _file)
		: mapping(m), file(_file)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		Vec2f uv = mapping.Map(its);
		if (its.hasUVPartials)
			return tex->eval(uv, Vec2f(its.dudx * mapping.su, its.dvdx * mapping.sv),
								 Vec2f(its.dudy * mapping.su, its.dvdy * mapping.sv));
		return tex->Sample(uv);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		if(tex.operator*())
			return tex->Sample(Vec2f(0), 1);
		else return Spectrum(0.0f);
	}
	template<typename L> void LoadTextures(L callback)
	{
		texRef = callback(file, true);
		tex = texRef.AsVar();
	}
	e_BufferReference<e_MIPMap, e_KernelMIPMap> texRef;
	e_Variable<e_KernelMIPMap> tex;
	e_KernelTextureMapping2D mapping;
	FixedString<64> file;
};

struct e_KernelUVTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<5>
{
	e_KernelUVTexture(){}
	e_KernelUVTexture(const e_KernelTextureMapping2D& m)
		: mapping(m)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		float2 uv = mapping.Map(map);
		return Spectrum(math::frac(uv.x), math::frac(uv.y), 0);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.5f);
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	e_KernelTextureMapping2D mapping;
};

struct e_KernelWireframeTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<6>
{
	e_KernelWireframeTexture(float lineWidth = 0.1f, const Spectrum& interior = Spectrum(0.5f), const Spectrum& edge = Spectrum(0.0f))
		: width(lineWidth), interiorColor(interior), edgeColor(edge)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		if (map.bary.x < width || map.bary.y < width || map.bary.x + map.bary.y > 1.0f - width)
			return edgeColor;
		else return interiorColor;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.1f);
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	Spectrum interiorColor, edgeColor;
	float width;
};

struct e_KernelExtraDataTexture : public e_KernelTextureBase, public e_DerivedTypeHelper<7>
{
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		return Spectrum(float(map.extraData) / 255.0f);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.0f);
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
};

#define TEX_SIZE Dmax7(sizeof(e_KernelBilerpTexture), sizeof(e_KernelConstantTexture), sizeof(e_KernelImageTexture), \
						sizeof(e_KernelUVTexture), sizeof(e_KernelCheckerboardTexture), sizeof(e_KernelWireframeTexture), sizeof(e_KernelExtraDataTexture))

struct e_KernelTexture : public e_AggregateBaseType<e_KernelTextureBase, TEX_SIZE>
{
public:
	CUDA_FUNC_IN e_KernelTexture()
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry & dg) const
	{
		CALL_FUNC7(e_KernelBilerpTexture, e_KernelConstantTexture, e_KernelCheckerboardTexture, e_KernelImageTexture, e_KernelUVTexture, e_KernelWireframeTexture, e_KernelExtraDataTexture, Evaluate(dg))
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		CALL_FUNC7(e_KernelBilerpTexture, e_KernelConstantTexture, e_KernelCheckerboardTexture, e_KernelImageTexture, e_KernelUVTexture, e_KernelWireframeTexture, e_KernelExtraDataTexture, Average())
		return Spectrum(0.0f);
	}
	template<typename L> void LoadTextures(L callback)
	{
		CALL_FUNC7(e_KernelBilerpTexture, e_KernelConstantTexture, e_KernelCheckerboardTexture, e_KernelImageTexture, e_KernelUVTexture, e_KernelWireframeTexture, e_KernelExtraDataTexture, LoadTextures(callback))
	}
};

template<typename U> static e_KernelTexture CreateTexture(const U& val)
{
	e_KernelTexture r;
	r.SetData(val);
	return r;
}

static e_KernelTexture CreateTexture(const char* p)
{
	e_KernelImageTexture f(e_KernelTextureMapping2D(), p);
	return CreateTexture(f);
}

static e_KernelTexture CreateTexture(const Spectrum& col)
{
	e_KernelConstantTexture f(col);
	return CreateTexture(f);
}

static e_KernelTexture CreateTexture(const char* p, const Spectrum& col)
{
	if(p && *p)
		return CreateTexture(p);
	else return CreateTexture(col);
}