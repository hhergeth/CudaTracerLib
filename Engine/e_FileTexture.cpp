#include "StdAfx.h"
#include "e_FileTexture.h"
#include "e_ErrorHandler.h"
#include "e_FileTextureHelper.h"

e_MIPMap::e_MIPMap(InputStream& a_In)
{
	a_In >> m_uWidth;
	a_In >> m_uHeight;
	a_In >> m_uBpp;
	a_In.operator>>(*(int*)&m_uType);
	a_In.operator>>(*(int*)&m_uWrapMode);
	a_In.operator>>(*(int*)&m_uFilterMode);
	a_In >> m_uLevels;
	a_In >> m_uSize;
	if(CUDA_MALLOC(&m_pDeviceData, m_uSize))
	{
		cudaError_t r = cudaGetLastError();
		std::cout << cudaGetErrorString(r) << "\n";
		BAD_CUDA_ALLOC(m_uSize)
	}
	m_pHostData = (unsigned int*)malloc(m_uSize);
	a_In.Read(m_pHostData, m_uSize);
	if(cudaMemcpy(m_pDeviceData, m_pHostData, m_uSize, cudaMemcpyHostToDevice))
		BAD_HOST_DEVICE_COPY(m_pDeviceData, m_uSize)
	a_In.Read(m_sOffsets, sizeof(m_sOffsets));
	a_In.Read(m_weightLut, sizeof(m_weightLut));
}

struct sampleHelper
{
	imgData* data;
	void* source, *dest;
	int w;

	sampleHelper(imgData* d, void* tmp, int i, int _w)
	{
		w = _w;
		data = d;
		source = i % 2 == 1 ? d->data : tmp;
		dest = i % 2 == 1 ? tmp : d->data;
	}

	Vec4f operator()(int x, int y)
	{
		if(data->type == vtRGBE)
			return Vec4f(SpectrumConverter::RGBEToFloat3(((RGBE*)source)[y * w + x]), 1);
		else return SpectrumConverter::COLORREFToFloat4(((RGBCOL*)source)[y * w + x]);
	}
};

void resize(imgData* d)
{
	int w = math::RoundUpPow2(d->w), h = math::RoundUpPow2(d->h);
	int* data = (int*)malloc(w * h * 4);
	for(int x = 0; x < w; x++)
		for(int y = 0; y < h; y++)
		{
			float x2 = float(d->w) * float(x) / float(w), y2 = float(d->h) * float(y) / float(h);
			data[y * w + x] = ((int*)d->data)[int(y2) * d->w + int(x2)];
		}
	free(d->data);
	d->data = data;
	d->w = w;
	d->h = h;
}

void e_MIPMap::CompileToBinary(const char* a_InputFile, OutputStream& a_Out, bool a_MipMap)
{
	imgData data;
	if(!parseImage(a_InputFile, &data))
		throw 1;
	if(popc(data.w) != 1 || popc(data.h) != 1)
		resize(&data);

	unsigned int nLevels = 1 + math::Log2Int(max(float(data.w), float(data.h)));
	//if(!a_MipMap)
	//	nLevels = 1;
	unsigned int size = 0;
	for(unsigned int i = 0, j = data.w, k = data.h; i < nLevels; i++, j =  j >> 1, k = k >> 1)
		size += j * k * 4;
	void* buf = malloc(data.w * data.h * 4);//it will be smaller

	a_Out << data.w;
	a_Out << data.h;
	a_Out << unsigned int(4);
	a_Out << (int)data.type;
	a_Out << (int)TEXTURE_REPEAT;
	a_Out << (int)TEXTURE_Anisotropic;
	a_Out << nLevels;
	a_Out << size;
	a_Out.Write(data.data, data.w * data.h * sizeof(RGBCOL));

	unsigned int m_sOffsets[max_MIPS];
	m_sOffsets[0] = 0;
	unsigned int off = data.w * data.h;
	for(unsigned int i = 1, j = data.w / 2, k = data.h / 2; i < nLevels; i++, j >>= 1, k >>= 1)
	{
		sampleHelper H(&data, buf, i, j * 2);//last width
		for(unsigned int t = 0; t < k; t++)
			for(unsigned int s = 0; s < j; s++)
			{
				void* tar = (RGBE*)H.dest + t * j + s;
				Vec4f v = 0.25f * (H(2*s, 2*t) + H(2*s+1, 2*t) + H(2*s, 2*t+1) + H(2*s+1, 2*t+1));
				if(data.type == vtRGBE)
					*(RGBE*)tar = SpectrumConverter::Float3ToRGBE(v.getXYZ());
				else *(RGBCOL*)tar = SpectrumConverter::Float4ToCOLORREF(v);
			}
		m_sOffsets[i] = off;
		off += j * k;
		a_Out.Write(H.dest, j * k * sizeof(RGBCOL));
	}
	a_Out.Write(m_sOffsets, sizeof(m_sOffsets));
	for (int i = 0; i<MTS_MIPMAP_LUT_SIZE; ++i)
	{
		float r2 = (float)i / (float)(MTS_MIPMAP_LUT_SIZE - 1);
		float val = expf(-2.0f * r2) - expf(-2.0f);
		a_Out << val;
	}
	free(data.data);
	free(buf);
}

e_KernelMIPMap e_MIPMap::getKernelData()
{
	e_KernelMIPMap r;
	r.m_pDeviceData = m_pDeviceData;
	r.m_pHostData = m_pHostData;
	r.m_uType = m_uType;
	r.m_uWrapMode = m_uWrapMode;
	r.m_uFilterMode = m_uFilterMode;
	r.m_uWidth = m_uWidth;
	r.m_uHeight = m_uHeight;
	r.m_fDim = Vec2f(m_uWidth - 1, m_uHeight - 1);
	r.m_uLevels = m_uLevels;
	memcpy(r.m_sOffsets, m_sOffsets, sizeof(m_sOffsets));
	memcpy(r.m_weightLut, m_weightLut, sizeof(m_weightLut));
	return r;
}