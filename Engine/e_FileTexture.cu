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
	return sqrtf(a * a + b * b);
}

Spectrum e_KernelMIPMap::Texel(unsigned int level, const float2& a_UV) const
{
	float2 l;
	if(!WrapCoordinates(a_UV, make_float2(m_uWidth >> level, m_uHeight >> level), m_uWrapMode, &l))
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

Spectrum e_KernelMIPMap::triangle(unsigned int level, const float2& a_UV) const
{
	level = clamp(level, 0u, m_uLevels-1);
	float2 s = make_float2(m_uWidth >> level, m_uHeight >> level), is = make_float2(1) / s;
	float2 l = a_UV * s;// - make_float2(0.5f)
	float ds = frac(l.x), dt = frac(l.y);
	return (1.f-ds) * (1.f-dt) * Texel(level, a_UV) +
			(1.f-ds) * dt       * Texel(level, a_UV + make_float2(0, is.y)) +
			ds       * (1.f-dt) * Texel(level, a_UV + make_float2(is.x, 0)) +
			ds       * dt       * Texel(level, a_UV + make_float2(is.x, is.y));
}

Spectrum e_KernelMIPMap::evalEWA(unsigned int level, const float2 &uv, float A, float B, float C) const
{
	if (level >= m_uLevels)
		return Texel(m_uLevels - 1, make_float2(0));

	float2 size = make_float2(m_uWidth >> level, m_uHeight >> level);
	float u = uv.x * size.x - 0.5f;
	float v = uv.y * size.y - 0.5f;

	/* Do the same to the ellipse coefficients */
	float2 ratio = size / m_fDim;
	A /= ratio.x * ratio.x;
	B /= ratio.x * ratio.y;
	C /= ratio.y * ratio.y;

	float invDet = 1.0f / (-B*B + 4.0f*A*C),
		deltaU = 2.0f * sqrtf(C * invDet),
		deltaV = 2.0f * sqrtf(A * invDet);
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
					result += Texel(level, make_float2(ut, vt) / size) * weight;
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

Spectrum e_KernelMIPMap::Sample(const float2& uv) const
{
	if (m_uFilterMode == TEXTURE_Point)
		return Texel(0, uv);
	else return triangle(0, uv);
}

float e_KernelMIPMap::SampleAlpha(const float2& uv) const
{
	float2 l;
	if(!WrapCoordinates(uv, make_float2(m_uWidth, m_uHeight), m_uWrapMode, &l))
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

Spectrum e_KernelMIPMap::Sample(const float2& a_UV, float width) const
{
	float level = m_uLevels - 1 + Log2(MAX((float)width, 1e-8f));
	if (level < 0)
		return triangle(0, a_UV);
	else if (level >= m_uLevels - 1)
		return Texel(m_uLevels - 1, a_UV);
	else
	{
		int iLevel = Floor2Int(level);
		float delta = level - iLevel;
		return (1.f-delta) * triangle(iLevel, a_UV) + delta * triangle(iLevel+1, a_UV);
	}
}

Spectrum e_KernelMIPMap::Sample(float width, int x, int y) const
{
	float l = m_uLevels - 1 + Log2(MAX((float)width, 1e-8f));
	int level = (int)clamp(l, 0.0f, float(m_uLevels - 1));
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

void e_KernelMIPMap::evalGradient(const float2& uv, Spectrum* gradient) const
{
	const int level = 0;

	float u = uv.x * m_fDim.x - 0.5f, v = uv.y * m_fDim.y - 0.5f;

	int xPos = Float2Int(u), yPos = Float2Int(v);
	float dx = u - xPos, dy = v - yPos;

	const Spectrum p00 = Texel(level, make_float2(xPos, yPos) / m_fDim);
	const Spectrum p10 = Texel(level, make_float2(xPos + 1, yPos) / m_fDim);
	const Spectrum p01 = Texel(level, make_float2(xPos, yPos + 1) / m_fDim);
	const Spectrum p11 = Texel(level, make_float2(xPos + 1, yPos + 1) / m_fDim);
	Spectrum tmp = p01 + p10 - p11;

	gradient[0] = (p10 + p00*(dy - 1) - tmp*dy) * m_fDim.x;
	gradient[1] = (p01 + p00*(dx - 1) - tmp*dx) * m_fDim.y;
}

Spectrum e_KernelMIPMap::eval(const float2& uv, const float2& d0, const float2& d1) const
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
		majorRadius = Aprime != 0 ? sqrtf(F / Aprime) : 0,
		minorRadius = Cprime != 0 ? sqrtf(F / Cprime) : 0;

	if (!(minorRadius > 0) || !(majorRadius > 0) || F < 0)
	{
		float level = log2f(MAX(majorRadius, 1e-4f));
		int ilevel = Floor2Int(level);
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
		float level = MAX(0.0f, log2f(minorRadius));
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
	CUDA_FUNC_IN float2 cubizePoint4(float3& position, int& face)
	{
		float3 q = fabsf(position);
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
		return (make_float2(sc * sign1, tc * sign2) / w + make_float2(1)) / 2.0f;
	}

	CUDA_FUNC_IN float3 operator()(float w, float h, unsigned int& x, unsigned int y, imgData* maps)
	{
		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos((1.0f - x / w) * 2 * PI, &sinPhi, &cosPhi);
		sincos((1.0f - y / h) * PI, &sinTheta, &cosTheta);
		float3 d = make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		int face;
		float2 uv = cubizePoint4(d, face);
		if(face == 2 || face == 3)
			x = (x + int(w) / 4) % int(w);
		Spectrum s = maps[face].Load(int(uv.x * (maps[face].w - 1)), int((1.0f - uv.y) * (maps[face].h - 1)));
		float r, g, b;
		s.toLinearRGB(r, g, b);
		return make_float3(r, g, b);
	}
};

CUDA_CONST imgData mapsCuda[6];
__global__ void generateSkydome(unsigned int w, unsigned int h, float3* Target)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x < w && y < h)
	{
		unsigned int xp = x;
		float3 c = MapPoint()(w, h, xp, y, mapsCuda);
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
	float3* B = (float3*)FreeImage_GetBits(bitmap);
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
		CUDA_MALLOC(&T, sizeof(float3) * w * h);
		generateSkydome<<<dim3((w+31)/32,(h+31)/32,1), dim3(32, 32, 1)>>>(w,h,(float3*)T);
		cudaDeviceSynchronize();
		cudaMemcpy(B, T, sizeof(float3) * w * h, cudaMemcpyDeviceToHost);
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
				float3 c = M(w, h, xp, y, maps);
				B[y * w + xp] = c;
			}
	}
	bool b = FreeImage_Save(FIF_EXR, bitmap, outFile);
	FreeImage_Unload(bitmap);
	for(int i = 0; i < 6; i++)
		free(maps[i].data);
}