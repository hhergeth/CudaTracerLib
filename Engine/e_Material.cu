#include "e_Material.h"

void initbssrdf(e_VolumeRegion& reg)
{
	const float a = 1e10f;
	e_PhaseFunction func;
	func.SetData(e_IsotropicPhaseFunction());
	reg.SetData(e_HomogeneousVolumeDensity(func, (float4x4::Translate(Vec3f(0.5f)) % float4x4::Scale(Vec3f(0.5f / a))).inverse(), Spectrum(0.0f), Spectrum(0.0f), Spectrum(0.0f)));
	reg.As()->Update();
}

e_KernelMaterial::e_KernelMaterial()
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

e_KernelMaterial::e_KernelMaterial(const std::string& name)
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

CUDA_FUNC_IN Vec2f parallaxOcclusion(const Vec2f& texCoord, e_KernelMIPMap* tex, const Vec3f& vViewTS, float HeightScale, int MinSamples, int MaxSamples)
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
	return texCoord - ParallaxOffset;
}

bool e_KernelMaterial::SampleNormalMap(DifferentialGeometry& dg, const Vec3f& wi) const
{
	if(NormalMap.used)
	{
		Vec3f n;
		NormalMap.tex.Evaluate(dg).toLinearRGB(n.x, n.y, n.z);
		Vec3f nWorld = dg.toWorld(n - Vec3f(0.5f));
		dg.sys.n = normalize(nWorld);
		dg.sys.t = normalize(cross(nWorld, dg.sys.s));
		dg.sys.s = normalize(cross(nWorld, dg.sys.t));
		return true;
	}
	else if (HeightMap.used && HeightMap.tex.Is<e_ImageTexture>())
	{
		e_TextureMapping2D& map = HeightMap.tex.As<e_ImageTexture>()->mapping;
		Vec2f uv = map.Map(dg);
		if (enableParallaxOcclusion)
		{
			uv = parallaxOcclusion(uv, HeightMap.tex.As<e_ImageTexture>()->tex.operator->(), dg.toLocal(-wi), HeightScale, parallaxMinSamples, parallaxMaxSamples);
			dg.uv[0] = (uv - Vec2f(map.du, map.dv)) / Vec2f(map.su, map.sv);
		}

		Spectrum grad[2];
		HeightMap.tex.As<e_ImageTexture>()->tex->evalGradient(uv, grad);
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

float e_KernelMaterial::SampleAlphaMap(const DifferentialGeometry& uv) const
{
	if(AlphaMap.used)
	{//return 1;
		if (AlphaMap.tex.Is<e_ImageTexture>())
		{
			Vec2f uv2 = AlphaMap.tex.As<e_ImageTexture>()->mapping.Map(uv);
			return AlphaMap.tex.As<e_ImageTexture>()->tex->SampleAlpha(uv2) != 1 ? 0 : 1;
		}
		Spectrum s = AlphaMap.tex.Evaluate(uv);
		if(s.isZero())
			return 0.0f;
		else return 1.0f;
	}
	else return 1.0f;
}

bool e_KernelMaterial::GetBSSRDF(const DifferentialGeometry& uv, const e_VolumeRegion** res) const
{
	if (usedBssrdf)
		*res = &bssrdf;
	return !!usedBssrdf;
}
