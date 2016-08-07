#include "Material.h"
#include "TriangleData.h"

namespace CudaTracerLib {

void initbssrdf(VolumeRegion& reg)
{
	const float a = 1e10f;
	PhaseFunction func;
	func.SetData(IsotropicPhaseFunction());
	reg.SetData(HomogeneousVolumeDensity(func, float4x4::Scale(Vec3f(2*a)) % float4x4::Translate(Vec3f(-0.5f)), Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f)));
	reg.As()->Update();
}

Material::Material()
{
	parallaxMinSamples = 10;
	parallaxMaxSamples = 50;
	enableParallaxOcclusion = false;
	Name = "NoNameMaterial";
	HeightScale = 1.0f;
	NodeLightIndex = UINT_MAX;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	NormalMap.used = HeightMap.used = 0;
	AlphaMap.state = AlphaBlendState::Disabled;
	initbssrdf(bssrdf);
}

Material::Material(const std::string& name)
{
	parallaxMinSamples = 10;
	parallaxMaxSamples = 50;
	enableParallaxOcclusion = false;
	Name = name;
	HeightScale = 1.0f;
	NodeLightIndex = UINT_MAX;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	NormalMap.used = HeightMap.used = 0;
	AlphaMap.state = AlphaBlendState::Disabled;
	initbssrdf(bssrdf);
}

CUDA_FUNC_IN void parallaxOcclusion(Vec2f& texCoord, KernelMIPMap* tex, const Vec3f& vViewTS, float HeightScale, int MinSamples, int MaxSamples)
{
	const Vec2f vParallaxDirection = normalize(vViewTS.getXY());
	float fLength = length(vViewTS);
	float fParallaxLength = sqrt(fLength * fLength - vViewTS.z * vViewTS.z) / vViewTS.z;
	const Vec2f vParallaxOffsetTS = vParallaxDirection * fParallaxLength * HeightScale;

	int nNumSteps = (int)math::lerp(MaxSamples, MinSamples, Frame::cosTheta(normalize(vViewTS)));
	float CurrHeight = 0.0f;
	float StepSize = 1.0f / (float)nNumSteps;
	float PrevHeight = 1.0f;
	int    StepIndex = 0;
	Vec2f TexOffsetPerStep = StepSize * vParallaxOffsetTS;
	Vec2f TexCurrentOffset = texCoord;
	float  CurrentBound = 1.0f;
	float  ParallaxAmount = 0.0f;

	Vec2f pt1 = Vec2f(0);
	Vec2f pt2 = Vec2f(0);

	Vec2f texOffset2 = Vec2f(0);

	while (StepIndex < nNumSteps)
	{
		TexCurrentOffset -= TexOffsetPerStep;
		CurrHeight = tex->Sample(TexCurrentOffset).average();
		CurrentBound -= StepSize;
		if (CurrHeight > CurrentBound)
		{
			pt1 = Vec2f(CurrentBound, CurrHeight);
			pt2 = Vec2f(CurrentBound + StepSize, PrevHeight);

			texOffset2 = TexCurrentOffset - TexOffsetPerStep;

			StepIndex = nNumSteps + 1;
			PrevHeight = CurrHeight;
		}
		else
		{
			StepIndex++;
			PrevHeight = CurrHeight;
		}
	}
	float Delta2 = pt2.x - pt2.y;
	float Delta1 = pt1.x - pt1.y;
	float Denominator = Delta2 - Delta1;
	ParallaxAmount = Denominator != 0 ? (pt1.x * Delta2 - pt2.x * Delta1) / Denominator : 0;
	Vec2f ParallaxOffset = vParallaxOffsetTS * (1 - ParallaxAmount);
	texCoord -= ParallaxOffset;
}

bool Material::SampleNormalMap(DifferentialGeometry& dg, const Vec3f& wi) const
{
	if (NormalMap.used)
	{
		Vec3f n;
		NormalMap.tex.Evaluate(dg).toLinearRGB(n.x, n.y, n.z);
		auto nWorld = dg.sys.toWorld(n - Vec3f(0.5f)).normalized();
		dg.sys.n = nWorld;
		dg.sys.t = normalize(cross(nWorld, dg.sys.s));
		dg.sys.s = normalize(cross(nWorld, dg.sys.t));
		return true;
	}
	else if (HeightMap.used && HeightMap.tex.Is<ImageTexture>())
	{
		TextureMapping2D& map = HeightMap.tex.As<ImageTexture>()->mapping;
		Vec2f uv = map.Map(dg);
		if (enableParallaxOcclusion)
		{
			parallaxOcclusion(uv, HeightMap.tex.As<ImageTexture>()->tex.operator->(), dg.sys.toLocal(-wi), HeightScale, parallaxMinSamples, parallaxMaxSamples);
			dg.uv[map.setId] = map.TransformPointInverse(uv);
		}

		Spectrum grad[2];
		HeightMap.tex.As<ImageTexture>()->tex->evalGradient(uv, grad);
		float dDispDu = grad[0].getLuminance();
		float dDispDv = grad[1].getLuminance();
		Vec3f dpdu = dg.dpdu + dg.sys.n * (
			dDispDu - dot(dg.sys.n, dg.dpdu));
		Vec3f dpdv = dg.dpdv + dg.sys.n * (
			dDispDv - dot(dg.sys.n, dg.dpdv));

		dg.sys.n = normalize(cross(dpdu, dpdv));
		dg.sys.s = normalize(dpdu - dg.sys.n * dot(dg.sys.n, dpdu));
		dg.sys.t = cross(dg.sys.n, dg.sys.s).normalized();

		if (dot(dg.sys.n, dg.n) < 0)
			dg.sys.n = -dg.sys.n;

		return true;
	}
	else return false;
}

CUDA_FUNC_IN Spectrum sample_fast(const Texture& tex, const Vec2f& bary, const Vec2f& uv)
{
	Spectrum val;
	if (tex.Is<ConstantTexture>())
		val = tex.As<ConstantTexture>()->val.average();
	else if (tex.Is<WireframeTexture>())
		val = tex.As<WireframeTexture>()->Evaluate(bary).average();
	else
	{
		if (tex.Is<ImageTexture>())
			val = tex.As<ImageTexture>()->tex->SampleAlpha(tex.As<ImageTexture>()->mapping.TransformPoint(uv));
		else if (tex.Is<BilerpTexture>())
			val = tex.As<BilerpTexture>()->Evaluate(uv).average();
		else if (tex.Is<CheckerboardTexture>())
			val = tex.As<CheckerboardTexture>()->Evaluate(uv).average();
		else if (tex.Is<UVTexture>())
			val = tex.As<UVTexture>()->Evaluate(uv).average();
	}
	return val;
}

bool Material::AlphaTest(const Vec2f& bary, const Vec2f& uv) const
{
	if (AlphaMap.used())
	{
		auto* refl_tex = bsdf.As()->getTexture(0);
		auto* refl_img = refl_tex ? refl_tex->As<ImageTexture>() : 0;
		auto refl_uv = refl_img ? refl_img->mapping.TransformPoint(uv) : Vec2f(0.0f);
		auto* alpha_img = AlphaMap.tex.As<ImageTexture>();

		if (((AlphaMap.state == AlphaBlendState::AlphaMap_Alpha && alpha_img != 0) ||
			 (AlphaMap.state == AlphaBlendState::ReflectanceMap_Alpha && refl_img != 0)))
		{
			float alpha = FLT_MAX;
			if (AlphaMap.state == AlphaBlendState::AlphaMap_Alpha)
				alpha = alpha_img->tex->SampleAlpha(alpha_img->mapping.TransformPoint(uv));
			else alpha = refl_img->tex->SampleAlpha(refl_uv);
			return alpha >= AlphaMap.test_val_scalar;
		}
		else
		{
			Spectrum val = AlphaMap.state & 4 ? sample_fast(*refl_tex, bary, uv) : sample_fast(AlphaMap.tex, bary, uv);
			if ((AlphaMap.state & 3) == 1)
				return val.getLuminance() >= AlphaMap.test_val_scalar;
			else if ((AlphaMap.state & 3) == 3)
				return (val - AlphaMap.test_val_color).abs().max() <= AlphaMap.test_val_scalar;
			else return true;
		}
	}
	else return true;
}

bool Material::GetBSSRDF(const DifferentialGeometry& uv, const VolumeRegion** res) const
{
	if (usedBssrdf && res)
		*res = &bssrdf;
	return !!usedBssrdf;
}

}