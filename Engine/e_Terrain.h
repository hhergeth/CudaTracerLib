#include "..\Math\vector.h"
#include <limits.h>
#include "e_TerrainHeader.h"
#include <vector>
#include "e_Buffer.h"
#include "..\Base\FileStream.h"

// ---------
// | 0 | 1 | 
// ---------
// | 2 | 3 | 
// ---------

CUDA_FUNC_IN unsigned int calcBoxSize(unsigned int d, unsigned int t)
{
	int q = pow2(t - d);
	return q;
}

CUDA_FUNC_IN unsigned int calcIndex(unsigned int x, unsigned int y, unsigned int d, unsigned int t)
{
	if(!d) return 0;
	unsigned int s = calcBoxSize(d, t); //thats the size of a subdivison ie. x + q = next subdivision
	unsigned int r = sum_pow4(d - 1);//thats the sum off all previous levels
	unsigned int x2 = x / s, y2 = y / s;//we know that x and y must be multiples of this
	return r + x2 + y2 * pow2(d);
}

CUDA_FUNC_IN unsigned int calcIndex_Direct(unsigned int x, unsigned int y, unsigned int d)
{
	if(!d) return 0;
	unsigned int r = sum_pow4(d - 1);//thats the sum off all previous levels
	return r + x + y * pow2(d);
}

struct e_TerrainData_Inner
{
	half2 blocks[4];
	void SetEmpty()
	{
		for(int i = 0; i < 4; i++)
			blocks[i] = make_float2(0,0);
	}
	float2 calcVerticalRange()
	{
		float2 r = make_float2(FLT_MAX, -FLT_MAX);
		for(int i = 0; i < 4; i++)
		{
			float2 q = blocks[i].ToFloat2();
			r.x = MIN(r.x, q.x);
			r.y = MAX(r.y, q.y);
		}
		return r;
	}
};

struct e_TerrainData_Leaf
{
	
	union
	{
		int4 data;
		struct
		{
			float H0;
			signed char		   off01, off02;
			signed char off10, off11, off12;
			signed char off20, off21, off22;
		};
		struct
		{
			float M;
			half offlu, offru;
			half offld, offrd;
			unsigned char offmu, offmd, offlm, offrm;
		};
	};

	CUDA_HOST bool hasValidHeight()
	{
		return H0 != -FLT_MAX;
	}

	CUDA_HOST float2 getRange()
	{/*
		int4 l = data;
		signed char o01 = l.y & 255, o02 = (l.y >> 8) & 255,
						o10 = (l.y >> 16) & 255, o11 = l.y >> 24, o12 = l.z & 255,
						o20 = (l.z >> 8) & 255, o21 = (l.z >> 16) & 255, o22 = l.z >> 24;
		float H00 = H0,						 H01 = H00 + (float)o01 * 8.0f, H02 = H01 + (float)o02 * 8.0f,
			  H10 = H00 + (float)o10 * 8.0f, H11 = H10 + (float)o11 * 8.0f, H12 = H11 + (float)o12 * 8.0f,
			  H20 = H10 + (float)o20 * 8.0f, H21 = H20 + (float)o21 * 8.0f, H22 = H21 + (float)o22 * 8.0f;
		float mi = MIN(MIN(H00, H01, H02), MIN(H10, H11, H12), MIN(H20, H21, H22)), ma = MAX(MAX(H00, H01, H02), MAX(H10, H11, H12), MAX(H20, H21, H22));
		float2 rr = make_float2(mi, ma);
		return rr;*/
		int4 l = data;
		unsigned short o0 = l.y & 0x0000ffff, o1 = l.y >> 16, o2 = l.z & 0x0000ffff, o3 = l.z >> 16;
		unsigned char s0 = l.w & 255, s1 = (l.w >> 8) & 255, s2 = (l.w >> 16) & 255, s3 = (l.w >> 24) & 255;
		float H11 = M, H00 = H11 + half(o0), H02 = H11 + half(o1), H20 = H11 + half(o2), H22 = H11 + half(o3),
			  H01 = lerp(H00, H02, float(s0) / 255.0f), H21 = lerp(H20, H22, float(s1) / 255.0f),
			  H10 = lerp(H00, H20, float(s2) / 255.0f), H12 = lerp(H02, H22, float(s3) / 255.0f);
		float mi = MIN(MIN(H00, H01, H02), MIN(H10, H11, H12), MIN(H20, H21, H22)), ma = MAX(MAX(H00, H01, H02), MAX(H10, H11, H12), MAX(H20, H21, H22));
		return make_float2(mi, ma);
	}

	CUDA_HOST float3 calcNormal(float dx, float dy)
	{
		float f00 = H0, f01 = f00 + 8.0f * (float)off01, f02 = f01 + 8.0f * (float)off02;
		float f10 = f00 + 8.0f * (float)off10, f11 = f10 + 8.0f * (float)off11, f12 = f11 + 8.0f * (float)off12;
		float f20 = f10 + 8.0f * (float)off20, f21 = f20 + 8.0f * (float)off21, f22 = f21 + 8.0f * (float)off22;
		
		float3 n = -cross(make_float3(dx * 2.0f, f02 - f00, 0), make_float3(0, f20 - f00, dy * 2.0f));
		return normalize(normalize(n));
	}

	CUDA_HOST void setData(float a0, float a1, float a2,
						   float b0, float b1, float b2,
						   float c0, float c1, float c2)
	{/*
#define C(B, V) signed char(clamp((V - B) / 8.0f, -128.0f, 127.0f))
		H0 = a0;
		off01 = C(a0, a1);
		off02 = C(a1, a2);

		off10 = C(a0, b0);
		off11 = C(b0, b1);
		off12 = C(b1, b2);

		off20 = C(b0, c0);
		off21 = C(c0, c1);
		off22 = C(c1, c2);
#undef C*/
		M = b1;
		offlu = half(a0 - b1);
		offru = half(a2 - b1);
		offld = half(c0 - b1);
		offrd = half(c2 - b1);
#define INV(des, start, end) unsigned char(clamp((des - start) / (end - start) * 255.0f, 0.0f, 255.0f))
		offmu = INV(a1, a0, a2);
		offmd = INV(c1, c0, c2);
		offlm = INV(b0, a0, c0);
		offrm = INV(b2, a2, c2);
	}
	
/*
union
	{
		int4 data;
		struct
		{
			float H0, H1, H2, H3;
		};
	};

	CUDA_HOST bool hasValidHeight()
	{
		return H0 != -FLT_MAX;
	}

	CUDA_HOST float2 getRange()
	{
		return make_float2(MIN(H0, H1, H2, H3), MAX(H0, H1, H2, H3));
	}

	CUDA_HOST float3 calcNormal(float dx, float dy)
	{
		return make_float3(1);
	}

	CUDA_HOST void setData(float a0, float a1, float a2, float b0, float b1, float b2, float c0, float c1, float c2)
	{
		H0 = a0;
		H1 = a2;
		H2 = c0;
		H3 = c2;
	}
	*/
};

template<unsigned int W> struct e_TerrainDataTransporter
{
	float4 data[W * W];
	std::vector<int2> changes;
	int2 off;
public:
	int2 m_sOffset;
private:
	float4* acc(int x, int y)
	{
		x += off.x;
		y += off.y;
		int w = W - 1;
		x = clamp(x, 0, w);
		y = clamp(y, 0, w);
		return data + y * W + x;
	}
public:
	e_TerrainDataTransporter(int2 _off)
	{
		off = make_int2(0, 0);
		m_sOffset = _off;
	}
	template<typename CB> static void Create(CB C, e_TerrainDataTransporter<W>* Q)
	{
		for(int y = 0; y < W; y++)
			for(int x = 0; x < W; x++)
				Q[0](x, y, C(x, y));
	}
	float4 operator()(unsigned int _x, unsigned int _y)
	{
		return *acc(_x, _y);
	}
	void operator()(unsigned int _x, unsigned int _y, float4 f)
	{
		*acc(_x, _y) = f;
	}
	void pushOffset(int2 o)
	{
		changes.push_back(o);
		off += o;
	}
	void popOffset()
	{
		int2 o = changes[changes.size() - 1];
		changes.pop_back();
		off -= o;
	}
};

class e_Terrain
{
	e_Stream<e_TerrainData_Inner>* m_pStream;
	e_Stream<CACHE_LEVEL_TYPE>* m_pCacheStream;
	float2 m_sMin;
	float2 m_sMax;
	float2 m_sSpan;
	unsigned int m_uDepth;
public:
	e_Terrain(int a_Depth, float2 m, float2 M)
	{
		m_uDepth = MAX(1, a_Depth - 1);
		unsigned int N = sum_pow4(m_uDepth);
		m_pStream = new e_Stream<e_TerrainData_Inner>(N);
		m_pCacheStream = new e_Stream<CACHE_LEVEL_TYPE>(pow4(CACHE_LEVEL));
		m_sMin = m;
		m_sMax = M;
		m_sSpan = make_float2(FLT_MAX, -FLT_MAX);/*
		int4 z0 = make_int4(0,0,0,0);
		unsigned int mi = calcIndex(0,0,m_uDepth - 1, m_uDepth);
		int4* nend = (int4*)m_pStream->operator()(mi).operator e_TerrainData_Inner *();
		for(int4* n = (int4*)m_pStream->operator()().operator e_TerrainData_Inner *(); n < nend; n++)
			*n = z0;
		nend = (int4*)m_pStream->operator()(N - 1).operator e_TerrainData_Inner *() + 1;
		float f0 = -FLT_MAX;
		int f2 = *(int*)&f0;
		z0 = make_int4(f2,0,0,0);
		for(int4* n = (int4*)m_pStream->operator()(mi).operator e_TerrainData_Inner *(); n < nend; n++)
			*n = z0;*/
	}
	~e_Terrain()
	{
		delete m_pStream;
		delete m_pCacheStream;
	}
	void UpdateInvalidated()
	{
		m_pStream->UpdateInvalidated();
		m_pCacheStream->UpdateInvalidated();
	}
	e_KernelTerrainData getKernelData(bool devicePointer = true)
	{
		e_KernelTerrainData q;
		q.m_pNodes = m_pStream->getKernelData(devicePointer).Data;
		q.m_pCacheData = m_pCacheStream->getKernelData(devicePointer).Data;
		q.m_sMax = make_float3(m_sMax.x, m_sSpan.y, m_sMax.y);
		q.m_sMin = make_float3(m_sMin.x, m_sSpan.x, m_sMin.y);
		q.m_uDepth = m_uDepth;
		return q;
	}
	e_TerrainData_Leaf* getValAt(unsigned int x, unsigned int y);
	void updateFromTriangles();
	unsigned int getBufferSize()
	{
		return m_pStream->getSizeInBytes() + m_pCacheStream->getSizeInBytes();
	}
	unsigned int getSideSize()
	{
		return pow2(m_uDepth);
	}
	void printLevelMap(unsigned int lvl, uchar3* data);
	void Serialize(OutputStream& a_Out)
	{
		a_Out << m_sMin;
		a_Out << m_sMax;
		a_Out << m_sSpan;
		a_Out << m_uDepth;
		updateFromTriangles();
		m_pStream->Serialize(a_Out);
		m_pCacheStream->Serialize(a_Out);
	}
	e_Terrain(InputStream& a_In)
	{
		a_In >> m_sMin;
		a_In >> m_sMax;
		a_In >> m_sSpan;
		a_In >> m_uDepth;
		m_pStream = new e_Stream<e_TerrainData_Inner>(a_In);
		m_pCacheStream = new e_Stream<CACHE_LEVEL_TYPE>(a_In);
	}
	void Move(float x, float z)
	{
		m_sMin += make_float2(x, z);
		m_sMax += make_float2(x, z);
	}
};