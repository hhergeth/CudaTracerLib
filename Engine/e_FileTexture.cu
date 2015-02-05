#include "StdAfx.h"
#include "e_FileTexture.h"
#include "e_ErrorHandler.h"
#include "e_FileTextureHelper.h"

/// Integer floor function (single precision)
template <typename Scalar> CUDA_FUNC_IN int floorToInt(Scalar value) { return (int)floor(value); }

/// Integer ceil function (single precision)
template <typename Scalar> CUDA_FUNC_IN int ceilToInt(Scalar value) { return (int)ceil(value); }

CUDA_FUNC_IN float hypot2(float a, float b)
{
	return math::sqrt(a * a + b * b);
}

Spectrum e_KernelMIPMap::Texel(unsigned int level, const Vec2f& a_UV) const
{
	Vec2f l;
	if (!WrapCoordinates(a_UV, Vec2f(m_uWidth >> level, m_uHeight >> level), m_uWrapMode, &l))
		return Spectrum(0.0f);
	else
	{
		unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y;
		void* data;
#ifdef ISCUDA
		data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
		data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
		Spectrum s;
		if(m_uType == vtRGBE)
			s.fromRGBE(*(RGBE*)data);
		else s.fromRGBCOL(*(RGBCOL*)data);
		return s;
	}
}

Spectrum e_KernelMIPMap::triangle(unsigned int level, const Vec2f& a_UV) const
{
	level = math::clamp(level, 0u, m_uLevels-1);
	Vec2f s = Vec2f(m_uWidth >> level, m_uHeight >> level), is = Vec2f(1) / s;
	Vec2f l = a_UV * s;// - make_float2(0.5f)
	float ds = math::frac(l.x), dt = math::frac(l.y);
	return (1.f-ds) * (1.f-dt) * Texel(level, a_UV) +
		(1.f - ds) * dt       * Texel(level, a_UV + Vec2f(0, is.y)) +
		ds       * (1.f - dt) * Texel(level, a_UV + Vec2f(is.x, 0)) +
		ds       * dt       * Texel(level, a_UV + Vec2f(is.x, is.y));
}

Spectrum e_KernelMIPMap::evalEWA(unsigned int level, const Vec2f &uv, float A, float B, float C) const
{
	if (level >= m_uLevels)
		return Texel(m_uLevels - 1, Vec2f(0));

	Vec2f size = Vec2f(m_uWidth >> level, m_uHeight >> level);
	float u = uv.x * size.x - 0.5f;
	float v = uv.y * size.y - 0.5f;

	/* Do the same to the ellipse coefficients */
	Vec2f ratio = size / m_fDim;
	A /= ratio.x * ratio.x;
	B /= ratio.x * ratio.y;
	C /= ratio.y * ratio.y;

	float invDet = 1.0f / (-B*B + 4.0f*A*C),
		deltaU = 2.0f * math::sqrt(C * invDet),
		deltaV = 2.0f * math::sqrt(A * invDet);
	int u0 = ceilToInt(u - deltaU), u1 = floorToInt(u + deltaU);
	int v0 = ceilToInt(v - deltaV), v1 = floorToInt(v + deltaV);

	float As = A * MTS_MIPMAP_LUT_SIZE,
		  Bs = B * MTS_MIPMAP_LUT_SIZE,
		  Cs = C * MTS_MIPMAP_LUT_SIZE;

	Spectrum result(0.0f);
	float denominator = 0.0f;
	float ddq = 2 * As, uu0 = u0 - u;
	int nSamples = 0;

	for (int vt = v0; vt <= v1; ++vt)
	{
		const float vv = vt - v;

		float q = As*uu0*uu0 + (Bs*uu0 + Cs*vv)*vv;
		float dq = As*(2 * uu0 + 1) + Bs*vv;

		for (int ut = u0; ut <= u1; ++ut)
		{
			if (q < MTS_MIPMAP_LUT_SIZE)
			{
				unsigned int qi = (unsigned int)q;
				if (qi < MTS_MIPMAP_LUT_SIZE)
				{
					const float weight = m_weightLut[(int)q];
					result += Texel(level, Vec2f(ut, vt) / size) * weight;
					denominator += weight;
					++nSamples;
				}
			}
			q += dq;
			dq += ddq;
		}
	}

	if (denominator == 0)
		return triangle(level, uv);
	return result / denominator;
}

Spectrum e_KernelMIPMap::Sample(const Vec2f& uv) const
{
	if (m_uFilterMode == TEXTURE_Point)
		return Texel(0, uv);
	else return triangle(0, uv);
}

float e_KernelMIPMap::SampleAlpha(const Vec2f& uv) const
{
	Vec2f l;
	if (!WrapCoordinates(uv, Vec2f(m_uWidth, m_uHeight), m_uWrapMode, &l))
		return 0.0f;
	unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y, level = 0;
	void* data;
#ifdef ISCUDA
			data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
			data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
	if(m_uType == vtRGBE)
		return 1.0f;
	else return float(((RGBCOL*)data)->w) / 255.0f;
}

Spectrum e_KernelMIPMap::Sample(const Vec2f& a_UV, float width) const
{
	float level = m_uLevels - 1 + math::log2(max((float)width, 1e-8f));
	if (level < 0)
		return triangle(0, a_UV);
	else if (level >= m_uLevels - 1)
		return Texel(m_uLevels - 1, a_UV);
	else
	{
		int iLevel = math::Floor2Int(level);
		float delta = level - iLevel;
		return (1.f-delta) * triangle(iLevel, a_UV) + delta * triangle(iLevel+1, a_UV);
	}
}

Spectrum e_KernelMIPMap::Sample(float width, int x, int y) const
{
	float l = m_uLevels - 1 + math::log2(max((float)width, 1e-8f));
	int level = (int)math::clamp(l, 0.0f, float(m_uLevels - 1));
	void* data;
#ifdef ISCUDA
		data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
		data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
	Spectrum s;
	if(m_uType == vtRGBE)
		s.fromRGBE(*(RGBE*)data);
	else s.fromRGBCOL(*(RGBCOL*)data);
	return s;	
}

void e_KernelMIPMap::evalGradient(const Vec2f& uv, Spectrum* gradient) const
{
	const int level = 0;

	float u = uv.x * m_fDim.x - 0.5f, v = uv.y * m_fDim.y - 0.5f;

	int xPos = math::Float2Int(u), yPos = math::Float2Int(v);
	float dx = u - xPos, dy = v - yPos;

	const Spectrum p00 = Texel(level, Vec2f(xPos, yPos) / m_fDim);
	const Spectrum p10 = Texel(level, Vec2f(xPos + 1, yPos) / m_fDim);
	const Spectrum p01 = Texel(level, Vec2f(xPos, yPos + 1) / m_fDim);
	const Spectrum p11 = Texel(level, Vec2f(xPos + 1, yPos + 1) / m_fDim);
	Spectrum tmp = p01 + p10 - p11;

	gradient[0] = (p10 + p00*(dy - 1) - tmp*dy) * m_fDim.x;
	gradient[1] = (p01 + p00*(dx - 1) - tmp*dx) * m_fDim.y;
}

Spectrum e_KernelMIPMap::eval(const Vec2f& uv, const Vec2f& d0, const Vec2f& d1) const
{
	if (m_uFilterMode == TEXTURE_Point)
		return Texel(0, uv);
	else if (m_uFilterMode == TEXTURE_Bilinear)
		return triangle(0, uv);

	/* Convert into texel coordinates */
	float du0 = d0.x * m_fDim.x, dv0 = d0.y * m_fDim.y,
		  du1 = d1.x * m_fDim.x, dv1 = d1.y * m_fDim.y;

	/* Turn the texture-space Jacobian into the coefficients of an
	implicitly defined ellipse. */
	float A = dv0*dv0 + dv1*dv1,
		B = -2.0f * (du0*dv0 + du1*dv1),
		C = du0*du0 + du1*du1,
		F = A*C - B*B*0.25f;

	float root = hypot2(A - C, B),
		Aprime = 0.5f * (A + C - root),
		Cprime = 0.5f * (A + C + root),
		majorRadius = Aprime != 0 ? math::sqrt(F / Aprime) : 0,
		minorRadius = Cprime != 0 ? math::sqrt(F / Cprime) : 0;

	if (!(minorRadius > 0) || !(majorRadius > 0) || F < 0)
	{
		float level = log2f(max(majorRadius, 1e-4f));
		int ilevel = math::Floor2Int(level);
		if (ilevel < 0)
			return triangle(0, uv);
		else
		{
			float a = level - ilevel;
			return triangle(ilevel, uv) * (1.0f - a)
				 + triangle(ilevel + 1, uv) * a;
		}
	}
	else
	{
		const float m_maxAnisotropy = 16;
		if (minorRadius * m_maxAnisotropy < majorRadius)
		{
			minorRadius = majorRadius / m_maxAnisotropy;
			float theta = 0.5f * std::atan(B / (A - C)), sinTheta, cosTheta;
			sincos(theta, &sinTheta, &cosTheta);
			float a2 = majorRadius*majorRadius,
				b2 = minorRadius*minorRadius,
				sinTheta2 = sinTheta*sinTheta,
				cosTheta2 = cosTheta*cosTheta,
				sin2Theta = 2 * sinTheta*cosTheta;

			A = a2*cosTheta2 + b2*sinTheta2;
			B = (a2 - b2) * sin2Theta;
			C = a2*sinTheta2 + b2*cosTheta2;
			F = a2*b2;
		}
		/* Switch to normalized coefficients */
		float scale = 1.0f / F;
		A *= scale; B *= scale; C *= scale;
		/* Determine a suitable MIP map level, such that the filter
		covers a reasonable amount of pixels */
		float level = max(0.0f, log2f(minorRadius));
		int ilevel = (int)level;
		float a = level - ilevel;

		/* Switch to bilinear interpolation, be wary of round-off errors */
		if (majorRadius < 1 || !(A > 0 && C > 0))
			return triangle(ilevel, uv);
		else
			return evalEWA(ilevel, uv, A, B, C) * (1.0f - a) +
				   evalEWA(ilevel + 1, uv, A, B, C) * a;
	}
}

struct MapPoint
{
	CUDA_FUNC_IN Vec2f cubizePoint4(Vec3f& position, int& face)
	{
		Vec3f q = position.abs();
		if(q.x > q.y && q.x > q.z)
			face = 0;
		else if(q.y > q.z)
			face = 1;
		else face = 2;
		int f = face;
		float* val = (float*)&position;
		face = 2 * face + (val[face] > 0 ? 0 : 1);

		int2 uvIdxs[3] = {make_int2(2, 1), make_int2(0, 2), make_int2(0, 1)};
		float sc = val[uvIdxs[f].x], tc = val[uvIdxs[f].y], w = abs(val[f]);
		float sign1 = (face == 0 || face == 5) ? -1 : 1, sign2 = face == 2 ? 1 : -1;
		return (Vec2f(sc * sign1, tc * sign2) / w + Vec2f(1)) / 2.0f;
	}

	CUDA_FUNC_IN Vec3f operator()(float w, float h, unsigned int& x, unsigned int y, imgData* maps)
	{
		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos((1.0f - x / w) * 2 * PI, &sinPhi, &cosPhi);
		sincos((1.0f - y / h) * PI, &sinTheta, &cosTheta);
		Vec3f d = Vec3f(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		int face;
		Vec2f uv = cubizePoint4(d, face);
		if(face == 2 || face == 3)
			x = (x + int(w) / 4) % int(w);
		Spectrum s = maps[face].Load(int(uv.x * (maps[face].w - 1)), int((1.0f - uv.y) * (maps[face].h - 1)));
		float r, g, b;
		s.toLinearRGB(r, g, b);
		return Vec3f(r, g, b);
	}
};

CUDA_CONST imgData mapsCuda[6];
__global__ void generateSkydome(unsigned int w, unsigned int h, Vec3f* Target)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int xp = x;
		Vec3f c = MapPoint()(w, h, xp, y, mapsCuda);
		Target[y * w + xp] = c;
	}
}

void e_MIPMap::CreateSphericalSkydomeTexture(const char* front, const char* back, const char* left, const char* right, const char* top, const char* bottom, const char* outFile)
{
	imgData maps[6];
	parseImage(front, maps + 5);
	parseImage(back, maps + 4);
	parseImage(left, maps + 1);
	parseImage(right, maps + 0);
	parseImage(top, maps + 2);
	parseImage(bottom, maps + 3);
	MapPoint M;
	unsigned int w = maps[0].w * 2, h = maps[0].h;
	FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBF, w, h, 32);
	Vec3f* B = (Vec3f*)FreeImage_GetBits(bitmap);
	const bool useCuda = true;
	if(useCuda)
	{
		imgData mapsC[6];
		for(int i = 0; i < 6; i++)
		{
			mapsC[i] = maps[i];
			CUDA_MALLOC(&mapsC[i].data, 4 * maps[i].w * maps[i].h);
			cudaMemcpy(mapsC[i].data, maps[i].data, 4 * maps[i].w * maps[i].h, cudaMemcpyHostToDevice); 
		}
		cudaMemcpyToSymbol(mapsCuda, &mapsC[0], sizeof(mapsCuda));
		void* T;
		CUDA_MALLOC(&T, sizeof(Vec3f) * w * h);
		generateSkydome << <dim3((w + 31) / 32, (h + 31) / 32, 1), dim3(32, 32, 1) >> >(w, h, (Vec3f*)T);
		cudaDeviceSynchronize();
		cudaMemcpy(B, T, sizeof(Vec3f) * w * h, cudaMemcpyDeviceToHost);
		CUDA_FREE(T);
		for(int i = 0; i < 6; i++)
			CUDA_FREE(mapsC[i].data);
	}
	else
	{
		for(unsigned int x = 0; x < w; x++)
			for(unsigned int y = 0; y < h; y++)
			{
				unsigned int xp = x;
				Vec3f c = M(w, h, xp, y, maps);
				B[y * w + xp] = c;
			}
	}
	bool b = FreeImage_Save(FIF_EXR, bitmap, outFile);
	FreeImage_Unload(bitmap);
	for(int i = 0; i < 6; i++)
		free(maps[i].data);
}