#pragma once
#include "MIPMap.h"
#include <Math/Spectrum.h>

namespace CudaTracerLib {

//helper class for image loading
//C++ destructors can not be used due to passing to cuda kernels
struct imgData//used in global CUDA variables
{
private:
	void* data;
	unsigned int W, H;
	Texture_DataType type;
public:
	CUDA_FUNC_IN int w() const { return W; }
	CUDA_FUNC_IN int h() const { return H; }
	CUDA_FUNC_IN Texture_DataType t() const { return type; }
	CUDA_FUNC_IN void* d() const { return data; }
	void d(void* _data) { data = _data; }

	void SetInfo(int _w, int _h, Texture_DataType t)
	{
		type = t;
		W = _w;
		H = _h;
	}

	void Allocate(int _w, int _h, Texture_DataType t)
	{
		SetInfo(_w, _h, t);
		data = malloc(_w * _h * 4);
	}

	CUDA_FUNC_IN Spectrum Load(int x, int y) const
	{
		Spectrum s;
		if (type == vtRGBE)
			s.fromRGBE(((RGBE*)data)[y * W + x]);
		else s.fromRGBCOL(((RGBCOL*)data)[y * W + x]);
		return s;
	}

	void Free()
	{
		free(data);
	}

	void RescaleToPowerOf2()
	{
		int w = math::RoundUpPow2(W), h = math::RoundUpPow2(H);
		if (w == W && h == H)
			return;
		int* data = (int*)malloc(w * h * 4);
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++)
			{
				float x2 = float(W) * float(x) / float(w), y2 = float(H) * float(y) / float(h);
				data[y * w + x] = ((int*)this->data)[int(y2) * W + int(x2)];
			}
		free(this->data);
		this->data = data;
		W = w;
		H = h;
	}

	void SetRGBCOL(RGBCOL val, int x, int y)
	{
		if (type == Texture_DataType::vtRGBCOL)
			((RGBCOL*)data)[y * W + x] = val;
		else ((RGBE*)data)[y * W + x] = SpectrumConverter::Float3ToRGBE(SpectrumConverter::COLORREFToFloat3(val));
	}

	void SetRGBE(RGBE val, int x, int y)
	{
		if (type == Texture_DataType::vtRGBE)
			((RGBE*)data)[y * W + x] = val;
		else ((RGBCOL*)data)[y * W + x] = SpectrumConverter::Float3ToCOLORREF(SpectrumConverter::RGBEToFloat3(val));
	}

	void Set(const Spectrum& val, int x, int y)
	{
		if (type == Texture_DataType::vtRGBCOL)
			SetRGBCOL(val.toRGBCOL(), x, y);
		else SetRGBE(val.toRGBE(), x, y);
	}
};

bool parseImage(const std::string& a_InputFile, imgData& data);

}