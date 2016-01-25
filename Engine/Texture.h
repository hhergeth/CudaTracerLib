#pragma once

#include <Math/float4x4.h>
#include <Math/Spectrum.h>
#include "MIPMap_device.h"
#include "DifferentialGeometry.h"
#include <Base/FixedString.h>
#include <VirtualFuncType.h>

//Implementation and interface designed after PBRT.

namespace CudaTracerLib {

//An affine map from R^2 -> R^2
struct TextureMapping2D
{
	TextureMapping2D(float su = 1, float sv = 1, float du = 0, float dv = 0, int setId = 0)
	{
		m11 = su; m12 = 0;  m13 = du;
		m21 = 0;  m22 = sv; m23 = dv;
		this->setId = setId;
	}
	TextureMapping2D(const float4x4& mat, int setId = 0)
	{
		m11 = mat(0, 0); m12 = mat(0, 1); m13 = mat(0, 3);
		m21 = mat(1, 0); m22 = mat(1, 1); m23 = mat(1, 3);
		this->setId = setId;
	}
	CUDA_FUNC_IN Vec2f Map(const DifferentialGeometry& map) const
	{
		float x = map.uv[setId].x, y = map.uv[setId].y;
		return TransformPoint(Vec2f(x, y));
	}
	CUDA_FUNC_IN Vec2f TransformPoint(const Vec2f& p) const
	{
		return TransformDirection(p) + Vec2f(m13, m23);
	}
	CUDA_FUNC_IN Vec2f TransformDirection(const Vec2f& d) const
	{
		return Vec2f(m11 * d.x + m12 * d.y, m21 * d.x + m22 * d.y);
	}
	CUDA_FUNC_IN Vec2f TransformPointInverse(const Vec2f& p) const
	{
		return TransformDirectionInverse(p - Vec2f(m13, m23));
	}
	CUDA_FUNC_IN Vec2f TransformDirectionInverse(const Vec2f& d) const
	{
		float q = d.y - m21 / m11 * d.x, p = m22 - m21 / m11 * m12;
		float y = q / p, x = (d.x - m12 * y) / m11;
		return Vec2f(x, y);
	}
	CUDA_FUNC_IN void differentiate(const DifferentialGeometry& map,
									float& dsdx, float& dsdy, float& dtdx, float& dtdy) const
	{
		dsdx = m11 * map.dudx + m12 * map.dvdx;
		dsdy = m11 * map.dudy + m12 * map.dvdy;
		dtdx = m21 * map.dudx + m22 * map.dvdx;
		dtdy = m21 * map.dudy + m22 * map.dvdy;
	}
	CUDA_FUNC_IN float4x4 ToFloat4x4() const
	{
		float4x4 r = float4x4::Identity();
		r(0, 0) = m11; r(0, 1) = m12; r(0, 3) = m13;
		r(1, 0) = m21; r(1, 1) = m22; r(1, 3) = m23;
		return r;
	}
public:
	float m11, m12, m13;
	float m21, m22, m23;
	int setId;
};

struct TextureBase : public BaseType//, public BaseTypeHelper<5784916>
{
};

struct BilerpTexture : public TextureBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	BilerpTexture(){}
	BilerpTexture(const TextureMapping2D& m, const Spectrum &t00, const Spectrum &t01, const Spectrum &t10, const Spectrum &t11)
	{
		mapping = m;
		v00 = t00;
		v01 = t01;
		v10 = t10;
		v11 = t11;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const Vec2f& _uv) const
	{
		Vec2f uv = mapping.TransformPoint(_uv);
		return (1 - uv.x)*(1 - uv.y) * v00 + (1 - uv.x)*(uv.y) * v01 +
			   (uv.x)*(1 - uv.y) * v10 + (uv.x)*(uv.y) * v11;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return Evaluate(its.uv[mapping.setId]);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return (v00 + v01 + v10 + v11) * 0.25f;
	}
	TextureMapping2D mapping;
	Spectrum v00, v01, v10, v11;
};

struct ConstantTexture : public TextureBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	ConstantTexture(){}
	ConstantTexture(const Spectrum& v)
		: val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return val;
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return val;
	}
	Spectrum val;
};

struct CheckerboardTexture : public TextureBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	CheckerboardTexture(){}
	CheckerboardTexture(const Spectrum& u, const Spectrum& v, const TextureMapping2D& m)
		: val0(u), val1(v), mapping(m)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const Vec2f& _uv) const
	{
		Vec2f uv = mapping.TransformPoint(_uv);
		int x = 2 * math::modulo((int)(uv.x * 2), 2) - 1,
			y = 2 * math::modulo((int)(uv.y * 2), 2) - 1;

		if (x*y == 1)
			return val0;
		else
			return val1;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return Evaluate(its.uv[mapping.setId]);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return (val0 + val1) / 2.0f;
	}
	Spectrum val0, val1;
	TextureMapping2D mapping;
};

struct ImageTexture : public TextureBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
		ImageTexture() :tex(0, 0){}
	ImageTexture(const TextureMapping2D& m, const std::string& _file, const Spectrum& scale)
		: mapping(m), file(_file), tex(0, 0), m_scale(scale)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const Vec2f& _uv) const
	{
		Vec2f uv = mapping.TransformPoint(_uv);
		return tex->Sample(uv) * m_scale;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		if (its.hasUVPartials)
		{
			Vec2f uv = mapping.Map(its);
			float dsdx, dsdy,
				  dtdx, dtdy;
			mapping.differentiate(its, dsdx, dsdy, dtdx, dtdy);
			return tex->eval(uv, Vec2f(dsdx, dtdx), Vec2f(dsdy, dtdy)) * m_scale;
		}
		else return Evaluate(its.uv[mapping.setId]);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		if (tex.operator*())
			return tex->Sample(Vec2f(0), 1) * m_scale;
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
	e_Variable<KernelMIPMap> tex;
	TextureMapping2D mapping;
	Spectrum m_scale;
	FixedString<128> file;
};

struct UVTexture : public TextureBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	UVTexture(){}
	UVTexture(const TextureMapping2D& m)
		: mapping(m)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const Vec2f& _uv) const
	{
		Vec2f uv = mapping.TransformPoint(_uv);
		return Spectrum(math::frac(uv.x), math::frac(uv.y), 0);
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return Evaluate(its.uv[mapping.setId]);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return Spectrum(0.5f);
	}
	TextureMapping2D mapping;
};

struct WireframeTexture : public TextureBase//, public e_DerivedTypeHelper<6>
{
	TYPE_FUNC(6)
	WireframeTexture(float lineWidth = 0.1f, const Spectrum& interior = Spectrum(0.5f), const Spectrum& edge = Spectrum(0.0f))
		: width(lineWidth), interiorColor(interior), edgeColor(edge)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const Vec2f& bary) const
	{
		if (bary.x < width || bary.y < width || bary.x + bary.y > 1.0f - width)
			return edgeColor;
		else return interiorColor;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return Evaluate(its.bary);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return Spectrum(0.1f);
	}
	Spectrum interiorColor, edgeColor;
	float width;
};

struct ExtraDataTexture : public TextureBase//, public e_DerivedTypeHelper<7>
{
	TYPE_FUNC(7)
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry& its) const
	{
		return Spectrum(float(its.extraData) / 255.0f);
	}
	CUDA_FUNC_IN Spectrum Average() const
	{
		return Spectrum(0.0f);
	}
};

struct Texture : public CudaVirtualAggregate<TextureBase, BilerpTexture, ConstantTexture, ImageTexture, UVTexture, CheckerboardTexture, WireframeTexture, ExtraDataTexture>
{
public:
	CALLER(Evaluate)
	CUDA_FUNC_IN Spectrum Evaluate(const DifferentialGeometry & its) const
	{
		return Evaluate_Helper::Caller<Spectrum>(this, its);
	}

	CALLER(Average)
	CUDA_FUNC_IN Spectrum Average() const
	{
		return Average_Helper::Caller<Spectrum>(this);
	}
};

static Texture CreateTexture(const Spectrum& col)
{
	ConstantTexture f(col);
	Texture r;
	r.SetData(f);
	return r;
}

static Texture CreateTexture(const std::string& p)
{
	ImageTexture f(TextureMapping2D(), p, Spectrum(1.0f));
	Texture r;
	r.SetData(f);
	return r;
}

}