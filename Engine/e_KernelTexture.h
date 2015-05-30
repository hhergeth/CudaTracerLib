#pragma once

#include "e_FileTexture_device.h"
#include "e_DifferentialGeometry.h"
#include "../Base/FixedString.h"

struct e_TextureMapping2D
{
	e_TextureMapping2D(float su = 1, float sv = 1, float du = 0, float dv = 0, int setId = 0)
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

struct e_TextureBase : public e_BaseType//, public e_BaseTypeHelper<5784916>
{
};

struct e_BilerpTexture : public e_TextureBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	e_BilerpTexture(){}
	e_BilerpTexture(const e_TextureMapping2D& m, const Spectrum &t00, const Spectrum &t01, const Spectrum &t10, const Spectrum &t11)
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
	e_TextureMapping2D mapping;
	Spectrum v00, v01, v10, v11;
};

struct e_ConstantTexture : public e_TextureBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	e_ConstantTexture(){}
	e_ConstantTexture(const Spectrum& v)
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
	Spectrum val;
};

struct e_CheckerboardTexture : public e_TextureBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	e_CheckerboardTexture(){}
	e_CheckerboardTexture(const Spectrum& u, const Spectrum& v, const e_TextureMapping2D& m)
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
	Spectrum val0, val1;
	e_TextureMapping2D mapping;
};

struct e_ImageTexture : public e_TextureBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	e_ImageTexture(){}
	e_ImageTexture(const e_TextureMapping2D& m, const std::string& _file)
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
	template<typename L> void LoadTextures(L& callback)
	{
		tex = callback(file, tex);
	}
	template<typename L> void UnloadTexture(L& callback)
	{
		callback(file, tex);
	}
	e_Variable<e_KernelMIPMap> tex;
	e_TextureMapping2D mapping;
	FixedString<64> file;
};

struct e_UVTexture : public e_TextureBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	e_UVTexture(){}
	e_UVTexture(const e_TextureMapping2D& m)
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
	e_TextureMapping2D mapping;
};

struct e_WireframeTexture : public e_TextureBase//, public e_DerivedTypeHelper<6>
{
	TYPE_FUNC(6)
	e_WireframeTexture(float lineWidth = 0.1f, const Spectrum& interior = Spectrum(0.5f), const Spectrum& edge = Spectrum(0.0f))
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
	Spectrum interiorColor, edgeColor;
	float width;
};

struct e_ExtraDataTexture : public e_TextureBase//, public e_DerivedTypeHelper<7>
{
	TYPE_FUNC(7)
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& map) const
	{
		return Spectrum(float(map.extraData) / 255.0f);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.0f);
	}
};

struct e_Texture : public CudaVirtualAggregate<e_TextureBase, e_BilerpTexture, e_ConstantTexture, e_ImageTexture, e_UVTexture, e_CheckerboardTexture, e_WireframeTexture, e_ExtraDataTexture>
{
public:
	CUDA_FUNC_IN e_Texture()
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry & dg) const
	{
		CALL_FUNC7(e_BilerpTexture, e_ConstantTexture, e_ImageTexture, e_UVTexture, e_CheckerboardTexture, e_WireframeTexture, e_ExtraDataTexture, Evaluate(dg))
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		CALL_FUNC7(e_BilerpTexture, e_ConstantTexture, e_ImageTexture, e_UVTexture, e_CheckerboardTexture, e_WireframeTexture, e_ExtraDataTexture, Average())
		return Spectrum(0.0f);
	}
};

static e_Texture CreateTexture(const Spectrum& col)
{
	e_ConstantTexture f(col);
	e_Texture r;
	r.SetData(f);
	return r;
}

static e_Texture CreateTexture(const std::string& p)
{
	e_ImageTexture f(e_TextureMapping2D(), p);
	e_Texture r;
	r.SetData(f);
	return r;
}