#include "Material.h"

namespace CudaTracerLib {

void initbssrdf(VolumeRegion& reg)
{
	const float a = 1e10f;
	PhaseFunction func;
	func.SetData(IsotropicPhaseFunction());
	reg.SetData(HomogeneousVolumeDensity(func, (float4x4::Translate(Vec3f(0.5f)) % float4x4::Scale(Vec3f(0.5f / a))).inverse(), Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f)));
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
	m_fAlphaThreshold = 1.0f;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	AlphaMap.used = NormalMap.used = HeightMap.used = 0;
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
	m_fAlphaThreshold = 1.0f;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	AlphaMap.used = NormalMap.used = HeightMap.used = 0;
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
	float StepSize = 1.0 / (float)nNumSteps;
	float PrevHeight = 1.0;
	int    StepIndex = 0;
	Vec2f TexOffsetPerStep = StepSize * vParallaxOffsetTS;
	Vec2f TexCurrentOffset = texCoord;
	float  CurrentBound = 1.0;
	float  ParallaxAmount = 0.0;

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
		Vec3f nWorld = dg.toWorld(n - Vec3f(0.5f));
		dg.sys.n = normalize(nWorld);
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
			parallaxOcclusion(uv, HeightMap.tex.As<ImageTexture>()->tex.operator->(), dg.toLocal(-wi), HeightScale, parallaxMinSamples, parallaxMaxSamples);
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
		dg.sys.s = normalize(dpdu - dg.sys.n
			* dot(dg.sys.n, dpdu));
		dg.sys.t = cross(dg.sys.n, dg.sys.s);

		if (dot(dg.sys.n, dg.n) < 0)
			dg.sys.n *= -1;

		return true;
	}
	else return false;
}

float Material::SampleAlphaMap(const DifferentialGeometry& uv) const
{
	if (AlphaMap.used)
	{//return 1;
		if (AlphaMap.tex.Is<ImageTexture>())
		{
			Vec2f uv2 = AlphaMap.tex.As<ImageTexture>()->mapping.Map(uv);
			return AlphaMap.tex.As<ImageTexture>()->tex->SampleAlpha(uv2) != 1 ? 0 : 1;
		}
		Spectrum s = AlphaMap.tex.Evaluate(uv);
		if (s.isZero())
			return 0.0f;
		else return 1.0f;
	}
	else return 1.0f;
}

bool Material::GetBSSRDF(const DifferentialGeometry& uv, const VolumeRegion** res) const
{
	if (usedBssrdf)
		*res = &bssrdf;
	return !!usedBssrdf;
}

}