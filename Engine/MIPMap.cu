#include "StdAfx.h"
#include "MIPMap.h"
#include "MIPMapHelper.h"
#include <Base/FileStream.h>
#include <CudaMemoryManager.h>
#define FREEIMAGE_LIB
#include <FreeImage.h>

//MipMap evaluation copied from Mitsuba.

namespace CudaTracerLib {

template <typename Scalar> CUDA_FUNC_IN int floorToInt(Scalar value) { return (int)math::floor(value); }
template <typename Scalar> CUDA_FUNC_IN int ceilToInt(Scalar value) { return (int)math::ceil(value); }

CUDA_FUNC_IN float hypot2(float a, float b)
{
	return math::sqrt(a * a + b * b);
}

Spectrum KernelMIPMap::Texel(unsigned int level, const Vec2f& a_UV) const
{
	Vec2f l;
	if (!WrapCoordinates(a_UV, Vec2f((float)(m_uWidth >> level), (float)(m_uHeight >> level)), m_uWrapMode, &l))
		return Spectrum(0.0f);
	else
	{
		unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y;
		void* data;
		int i = max(0, ((int)m_uWidth >> (int)level));
#ifdef ISCUDA
		data = m_pDeviceData + (m_sOffsets[level] + y * i + x);
#else
		data = m_pHostData + (m_sOffsets[level] + y * i + x);
#endif
		Spectrum s;
		if (m_uType == vtRGBE)
			s.fromRGBE(*(RGBE*)data);
		else s.fromRGBCOL(*(RGBCOL*)data);
		return s;
	}
}

Spectrum KernelMIPMap::triangle(unsigned int level, const Vec2f& a_UV) const
{
	level = math::clamp(level, 0u, m_uLevels - 1);
	Vec2f s = Vec2f((float)(m_uWidth >> level), (float)(m_uHeight >> level)), is = Vec2f(1.0f) / s;
	Vec2f l = a_UV * s;// - make_float2(0.5f)
	float ds = math::frac(l.x), dt = math::frac(l.y);
	return (1.f - ds) * (1.f - dt) * Texel(level, a_UV) +
		(1.f - ds) * dt       * Texel(level, a_UV + Vec2f(0, is.y)) +
		ds       * (1.f - dt) * Texel(level, a_UV + Vec2f(is.x, 0)) +
		ds       * dt       * Texel(level, a_UV + Vec2f(is.x, is.y));
}

Spectrum KernelMIPMap::evalEWA(unsigned int level, const Vec2f &uv, float A, float B, float C) const
{
	if (level >= m_uLevels)
		return Texel(m_uLevels - 1, Vec2f(0));

	Vec2f size = Vec2f((float)(m_uWidth >> level), (float)(m_uHeight >> level));
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
					result += Texel(level, Vec2f((float)ut, (float)vt) / size) * weight;
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

Spectrum KernelMIPMap::Sample(const Vec2f& uv) const
{
	if (m_uFilterMode == TEXTURE_Point)
		return Texel(0, uv);
	else return triangle(0, uv);
}

float KernelMIPMap::SampleAlpha(const Vec2f& uv) const
{
	Vec2f l;
	if (!WrapCoordinates(uv, Vec2f((float)m_uWidth, (float)m_uHeight), m_uWrapMode, &l))
		return 0.0f;
	unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y, level = 0;
	void* data;
#ifdef ISCUDA
	data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
	data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
	if (m_uType == vtRGBE)
		return 1.0f;
	else return float(((RGBCOL*)data)->w) / 255.0f;
}

Spectrum KernelMIPMap::Sample(const Vec2f& a_UV, float width) const
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
		return (1.f - delta) * triangle(iLevel, a_UV) + delta * triangle(iLevel + 1, a_UV);
	}
}

Spectrum KernelMIPMap::Sample(float width, int x, int y) const
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
	if (m_uType == vtRGBE)
		s.fromRGBE(*(RGBE*)data);
	else s.fromRGBCOL(*(RGBCOL*)data);
	return s;
}

void KernelMIPMap::evalGradient(const Vec2f& uv, Spectrum* gradient) const
{
	const int level = 0;

	float u = uv.x * m_fDim.x - 0.5f, v = uv.y * m_fDim.y - 0.5f;

	int xPos = math::Float2Int(u), yPos = math::Float2Int(v);
	float dx = u - xPos, dy = v - yPos;

	const Spectrum p00 = Texel(level, Vec2f((float)xPos, (float)yPos) / m_fDim);
	const Spectrum p10 = Texel(level, Vec2f((float)xPos + 1, (float)yPos) / m_fDim);
	const Spectrum p01 = Texel(level, Vec2f((float)xPos, (float)yPos + 1) / m_fDim);
	const Spectrum p11 = Texel(level, Vec2f((float)xPos + 1, (float)yPos + 1) / m_fDim);
	Spectrum tmp = p01 + p10 - p11;

	gradient[0] = (p10 + p00*(dy - 1) - tmp*dy) * m_fDim.x;
	gradient[1] = (p01 + p00*(dx - 1) - tmp*dx) * m_fDim.y;
}

Spectrum KernelMIPMap::eval(const Vec2f& uv, const Vec2f& d0, const Vec2f& d1) const
{
	/* Convert into texel coordinates */
	float du0 = d0.x * m_fDim.x, dv0 = d0.y * m_fDim.y,
		  du1 = d1.x * m_fDim.x, dv1 = d1.y * m_fDim.y,
		  du = (du0 + du1) / 2.0f, dv = (dv0 + dv1) / 2.0f;

	if (m_uFilterMode == TEXTURE_Point)
		return Texel(0, uv);
	else if (m_uFilterMode == TEXTURE_Bilinear)
		return triangle(0, uv);
	else if (m_uFilterMode == TEXTURE_Trilinear)
	{
		float levela = math::log2(m_fDim.x / math::abs(du)),
			  levelb = math::log2(m_fDim.y / math::abs(dv)),
			  level = m_uLevels - math::clamp((levela + levelb) / 2.0f, 1.0f, (float)m_uLevels);
		int iLevel = math::Floor2Int(level), iLevel2 = math::clamp(iLevel + 1, 0, (int)m_uLevels - 1);
		float p = level - iLevel;
		Spectrum texelA = triangle(iLevel, uv), texelB = triangle(iLevel2, uv);
		return p * texelA + (1 - p) * texelB;
	}


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
		if (q.x > q.y && q.x > q.z)
			face = 0;
		else if (q.y > q.z)
			face = 1;
		else face = 2;
		int f = face;
		float* val = (float*)&position;
		face = 2 * face + (val[face] > 0 ? 0 : 1);

		int2 uvIdxs[3] = { make_int2(2, 1), make_int2(0, 2), make_int2(0, 1) };
		float sc = val[uvIdxs[f].x], tc = val[uvIdxs[f].y], w = math::abs(val[f]);
		float sign1 = (face == 0 || face == 5) ? -1.0f : 1.0f, sign2 = face == 2 ? 1.0f : -1.0f;
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
		if (face == 2 || face == 3)
			x = (x + int(w) / 4) % int(w);
		Spectrum s = maps[face].Load(int(uv.x * (maps[face].w() - 1)), int((1.0f - uv.y) * (maps[face].h() - 1)));
		float r, g, b;
		s.toLinearRGB(r, g, b);
		return Vec3f(r, g, b);
	}
};

CUDA_CONST imgData mapsCuda[6];
__global__ void generateSkydome(unsigned int w, unsigned int h, Vec3f* Target)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < w && y < h)
	{
		unsigned int xp = x;
		Vec3f c = MapPoint()(w, h, xp, y, mapsCuda);
		Target[y * w + xp] = c;
	}
}

void MIPMap::CreateSphericalSkydomeTexture(const std::string& front, const std::string& back, const std::string& left, const std::string& right, const std::string& top, const std::string& bottom, const std::string& outFile)
{
	imgData maps[6];
	parseImage(front, maps[5]);
	parseImage(back, maps[4]);
	parseImage(left, maps[1]);
	parseImage(right, maps[0]);
	parseImage(top, maps[2]);
	parseImage(bottom, maps[3]);
	MapPoint M;
	unsigned int w = maps[0].w() * 2, h = maps[0].h();
	FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBF, w, h, 32);
	Vec3f* B = (Vec3f*)FreeImage_GetBits(bitmap);
	const bool useCuda = true;
	if (useCuda)
	{
		imgData mapsC[6];
		for (int i = 0; i < 6; i++)
		{
			mapsC[i] = maps[i];
			void* deviceData;
			CUDA_MALLOC(&deviceData, 4 * maps[i].w() * maps[i].h());
			mapsC[i].d(deviceData);
			cudaMemcpy(mapsC[i].d(), maps[i].d(), 4 * maps[i].w() * maps[i].h(), cudaMemcpyHostToDevice);
		}
		ThrowCudaErrors(cudaMemcpyToSymbol(mapsCuda, &mapsC[0], sizeof(mapsCuda)));
		void* T;
		CUDA_MALLOC(&T, sizeof(Vec3f) * w * h);
		generateSkydome << <dim3(w / 32 + 1, h / 32 + 1, 1), dim3(32, 32, 1) >> >(w, h, (Vec3f*)T);
		ThrowCudaErrors(cudaDeviceSynchronize());
		ThrowCudaErrors(cudaMemcpy(B, T, sizeof(Vec3f) * w * h, cudaMemcpyDeviceToHost));
		CUDA_FREE(T);
		for (int i = 0; i < 6; i++)
			CUDA_FREE(mapsC[i].d());
	}
	else
	{
		for (unsigned int x = 0; x < w; x++)
			for (unsigned int y = 0; y < h; y++)
			{
				unsigned int xp = x;
				Vec3f c = M((float)w, (float)h, xp, y, maps);
				B[y * w + xp] = c;
			}
	}
	FREE_IMAGE_FORMAT ff = FreeImage_GetFIFFromFilename(outFile.c_str());
	if (!FreeImage_Save(ff, bitmap, outFile.c_str()))
		throw std::runtime_error(std::string(__FUNCTION__) + " :: FreeImage_Save");
	FreeImage_Unload(bitmap);
	for (int i = 0; i < 6; i++)
		maps[i].Free();
}

CUDA_FUNC_IN float sample(imgData& img, const Vec3f& p)
{
	Vec2f q = clamp01(p.getXY()) * Vec2f((float)img.w(), (float)img.h());
	return img.Load((int)q.x, (int)q.y).avg();
}
template<int SEARCH_STEPS> CUDA_FUNC_IN bool text_cords_next_intersection(imgData& img, const Vec3f& pos, const Vec3f& dir, float& height, Vec2f& kw)
{
	Vec3f vec = dir;
	Vec3f step_fwd = vec / SEARCH_STEPS;
	Vec3f ray_pos = pos + step_fwd;
	for (int i = 1; i < SEARCH_STEPS; i++)
	{
		height = sample(img, ray_pos);
		if (height <= ray_pos.z)
			ray_pos += step_fwd;
		else
		{
			kw = ray_pos.getXY();
			return true;
		}
	}
	return false;
}
template<int SEARCH_STEPS, int LOOKUP_WIDTH> __global__ void generateRelaxedConeMap(imgData img, float* coneData)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < img.w() && y < img.h())
	{
		float src_texel_depth = img.Load(x, y).avg();
		float* radius_cone = coneData + (y * img.w() + x);
		const float MAX_CONE_RATIO = ((float)LOOKUP_WIDTH / (float)max(img.w(), img.h())) / (1 - src_texel_depth);
		*radius_cone = MAX_CONE_RATIO;
		Vec3f src = Vec3f(float(x) / img.w(), float(y) / img.h(), src_texel_depth);
		int xmin = math::clamp(x - LOOKUP_WIDTH, 0, (int)img.w() - 1), xmax = math::clamp(x + LOOKUP_WIDTH, 0, (int)img.w() - 1);
		int ymin = math::clamp(y - LOOKUP_WIDTH, 0, (int)img.h() - 1), ymax = math::clamp(y + LOOKUP_WIDTH, 0, (int)img.h() - 1);
		for (int ti = xmin; ti <= xmax; ti++)
			for (int tj = ymin; tj <= ymax; tj++)
			{
				float tj_depth = img.Load(ti, tj).avg();
				if ((ti == x && tj == y) || tj_depth <= src_texel_depth)
					continue;
				Vec3f dst = Vec3f(float(ti) / img.w(), float(tj) / img.h(), tj_depth);
				float d;
				Vec2f kw;
				if (!text_cords_next_intersection<SEARCH_STEPS>(img, src, dst - src, d, kw))
					continue;
				float cone_ratio = d <= src_texel_depth ? 1.0f : length(src.getXY() - kw) / (d - src_texel_depth);
				if (*radius_cone > cone_ratio)
					*radius_cone = cone_ratio;
			}
	}
}

template<int SEARCH_STEPS> __global__ void generateRelaxedConeMap(imgData img, float* coneData, Vec3f Offset)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < img.w() && y < img.h())
	{
		Vec3f src = Vec3f(float(x) / img.w(), float(y) / img.h(), 0);
		Vec3f dst = src + Offset;
		dst.z = sample(img, dst);
		Vec3f vec = dst - src;
		vec /= vec.z;
		vec *= 1 - dst.z;
		Vec3f step_fwd = vec / SEARCH_STEPS;
		Vec3f ray_pos = dst + step_fwd;
		for (int i = 1; i < SEARCH_STEPS; i++)
		{
			float current_depth = sample(img, ray_pos);
			if (current_depth <= ray_pos.z)
				ray_pos += step_fwd;
		}
		float src_texel_depth = img.Load(x, y).avg();
		float cone_ratio = (ray_pos.z >= src_texel_depth) ? 1.0 : length(ray_pos.getXY() - src.getXY()) / (src_texel_depth - ray_pos.z);
		float best_ratio = coneData[y * img.w() + x];
		if (cone_ratio > best_ratio)
			cone_ratio = best_ratio;
		coneData[y * img.w() + x] = cone_ratio;
	}
}

void MIPMap::CreateRelaxedConeMap(const std::string& a_InputFile, FileOutputStream& a_Out)
{	imgData data;
	if (!parseImage(a_InputFile, data))
		throw std::runtime_error("Impossible to load texture file!");
	RGBCOL* hostData = (RGBCOL*)data.d(), *deviceData;
	CUDA_MALLOC(&deviceData, data.w() * data.h() * 4);
	data.d(deviceData);
	ThrowCudaErrors(cudaMemcpy(deviceData, hostData, data.w() * data.h() * 4, cudaMemcpyHostToDevice));
	float* deviceDepthData;
	CUDA_MALLOC(&deviceDepthData, data.w() * data.h() * 4);
	generateRelaxedConeMap<32, 64> << < dim3(data.w() / 16 + 1, data.h() / 16 + 1), dim3(16, 16) >> >(data, deviceDepthData);

	CUDA_FREE(deviceData);
	data.d(hostData);

	float* hostConeData = new float[data.w() * data.h()];
	ThrowCudaErrors(cudaMemcpy(hostConeData, deviceDepthData, data.w() * data.h() * 4, cudaMemcpyDeviceToHost));

	data.d(hostConeData); //do NOT free the actual image data, this will be done later
	data.RescaleToPowerOf2();
	hostConeData = (float*)data.d();

	FREE_IMAGE_FORMAT ff = FIF_PNG;
	FIBITMAP* bitmap = FreeImage_Allocate(data.w(), data.h(), 24, 0x000000ff, 0x0000ff00, 0x00ff0000);
	BYTE* A = FreeImage_GetBits(bitmap);
	unsigned int pitch = FreeImage_GetPitch(bitmap);
	int off = 0;
	for ( int y = 0; y < data.h(); y++)
	{
		for ( int x = 0; x < data.w(); x++)
		{
			float d = hostConeData[y * data.w() + x];
			unsigned char col = (unsigned char)(d * 255.0f);
			A[off + x * 3 + 0] = col;
			A[off + x * 3 + 1] = col;
			A[off + x * 3 + 2] = col;
		}
		off += pitch;
	}
	if (!FreeImage_Save(ff, bitmap, "../Data/conemap.png"))
		throw std::runtime_error(__FUNCTION__);
	FreeImage_Unload(bitmap);

	a_Out << data.w();
	a_Out << data.h();
	a_Out << (unsigned int)4;
	a_Out << (int)data.t();
	a_Out << (int)TEXTURE_REPEAT;
	a_Out << (int)TEXTURE_Anisotropic;
	a_Out << (unsigned int)1;
	a_Out << data.w() * data.h() * 8;

	float oldw = (float)data.w(), oldh = (float)data.h();
	float2* imgData = new float2[data.w() * data.h()];
	for ( int i = 0; i < data.w(); i++)
		for (int j = 0; j < data.h(); j++)
		{
			float xs = float(i) / data.w() * oldw, ys = float(j) / data.h() * oldh;
			Spectrum s;
			s.fromRGBCOL(hostData[(int)ys * (int)oldw + (int)xs]);
			float d = s.avg();
			float c = hostConeData[j * data.w() + i];
			imgData[j * data.w() + i] = make_float2(d, c);
		}
	a_Out.Write(imgData, data.w() * data.h() * sizeof(float2));
	delete[] imgData;

	unsigned int m_sOffsets[MAX_MIPS];
	a_Out.Write(m_sOffsets, sizeof(m_sOffsets));
	for (int i = 0; i < MTS_MIPMAP_LUT_SIZE; ++i)
	{
		float r2 = (float)i / (float)(MTS_MIPMAP_LUT_SIZE - 1);
		float val = math::exp(-2.0f * r2) - math::exp(-2.0f);
		a_Out << val;
	}

	data.Free();
	free(hostData);
}

bool parseImage(const std::string& a_InputFile, imgData& data)
{
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(a_InputFile.c_str(), 0);
	if (fif == FIF_UNKNOWN)
	{
		fif = FreeImage_GetFIFFromFilename(a_InputFile.c_str());
	}
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		FIBITMAP *dib = FreeImage_Load(fif, a_InputFile.c_str(), 0);
		if (!dib)
			return false;
		unsigned int w = FreeImage_GetWidth(dib);
		unsigned int h = FreeImage_GetHeight(dib);
		unsigned int scan_width = FreeImage_GetPitch(dib);
		unsigned int pitch = FreeImage_GetPitch(dib);
		FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
		Texture_DataType type = Texture_DataType::vtRGBCOL;
		if (((imageType == FIT_RGBAF) && (bpp == 128)) || ((imageType == FIT_RGBF) && (bpp == 96)))
			type = Texture_DataType::vtRGBE;
		data.Allocate(w, h, type);
		const auto* palette = FreeImage_GetPalette(dib);
		for (unsigned int y = 0; y < h; ++y)
		{
			for (unsigned int x = 0; x < w; ++x)
			{
				FIRGBAF *pixel = (FIRGBAF *)(bits + bpp / 8 * x);
				BYTE* pixel2 = bits + bpp / 8 * x;
				BYTE pixel3;
				if (type == Texture_DataType::vtRGBE)
					data.SetRGBE(SpectrumConverter::Float3ToRGBE(Vec3f(pixel->red, pixel->green, pixel->blue)), x, y);
				else if (bpp <= 8)
				{
					if (!FreeImage_GetPixelIndex(dib, x, y, &pixel3) || !palette)
						throw std::runtime_error("Error while reading image!");
					auto color = palette[pixel3];
					data.SetRGBCOL(make_uchar4(color.rgbRed, color.rgbGreen, color.rgbBlue, 255), x, y);
				}
				else data.SetRGBCOL(make_uchar4(pixel2[FI_RGBA_RED], pixel2[FI_RGBA_GREEN], pixel2[FI_RGBA_BLUE], bpp == 32 ? pixel2[FI_RGBA_ALPHA] : 255), x, y);
			}
			bits += pitch;
		}
		FreeImage_Unload(dib);
		return true;
	}
	else
	{
		return false;
	}
}

}
