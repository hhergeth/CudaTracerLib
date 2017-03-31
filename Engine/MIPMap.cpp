#include "StdAfx.h"
#include "MIPMap.h"
#include "MIPMapHelper.h"
#include <Base/FileStream.h>
#include <Base/CudaMemoryManager.h>

namespace CudaTracerLib {

MIPMap::MIPMap(const std::string& a_InputFile, IInStream& a_In)
	: m_pPath(a_InputFile)
{
	a_In >> m_uWidth;
	a_In >> m_uHeight;
	a_In >> m_uBpp;
	a_In.operator>>(*(int*)&m_uType);
	a_In.operator>>(*(int*)&m_uWrapMode);
	a_In.operator>>(*(int*)&m_uFilterMode);
	a_In >> m_uLevels;
	a_In >> m_uSize;
	CUDA_MALLOC(&m_pDeviceData, m_uSize);
	m_pHostData = (unsigned int*)malloc(m_uSize);
	a_In.Read(m_pHostData, m_uSize);
	ThrowCudaErrors(cudaMemcpy(m_pDeviceData, m_pHostData, m_uSize, cudaMemcpyHostToDevice));
	a_In.Read(m_sOffsets, sizeof(m_sOffsets));
	a_In.Read(m_weightLut, sizeof(m_weightLut));
}

void MIPMap::Free()
{
	CUDA_FREE(m_pDeviceData);
	free(m_pHostData);
}

void MIPMap::CompileToBinary(const std::string& in, const std::string& out, bool a_MipMap)
{
	FileOutputStream o(out);
	CompileToBinary(in, o, a_MipMap);
	o.Close();
}

void MIPMap::CompileToBinary(const std::string& a_InputFile, FileOutputStream& a_Out, bool a_MipMap)
{
	imgData data;
	if (!parseImage(a_InputFile, data))
		throw std::runtime_error("Impossible to load texture file!");
	if (popc(data.w()) != 1 || popc(data.h()) != 1)
		data.RescaleToPowerOf2();

	unsigned int nLevels = 1 + math::Log2Int(min(float(data.w()), float(data.h())));
	//if(!a_MipMap)
	//	nLevels = 1;
	unsigned int size = 0;
	for (unsigned int i = 0, j = data.w(), k = data.h(); i < nLevels; i++, j = j >> 1, k = k >> 1)
		size += j * k * 4;

	a_Out << data.w();
	a_Out << data.h();
	a_Out << (unsigned int)4;
	a_Out << (int)data.t();
	a_Out << (int)TEXTURE_REPEAT;
	a_Out << (int)TEXTURE_Anisotropic;
	a_Out << nLevels;
	a_Out << size;
	a_Out.Write(data.d(), data.w() * data.h() * sizeof(RGBCOL));

	imgData tmpData;
	tmpData.Allocate(data.w() * 2, data.h() * 2, data.t());
	imgData* buffer[2] = { &data, &tmpData };
	unsigned int m_sOffsets[MAX_MIPS];
	m_sOffsets[0] = 0;
	unsigned int off = data.w() * data.h();
	for (unsigned int i = 1, j = data.w() / 2, k = data.h() / 2; i < nLevels; i++, j >>= 1, k >>= 1)
	{
		buffer[0]->SetInfo(j * 2, k * 2, buffer[0]->t()); buffer[1]->SetInfo(j, k, buffer[1]->t());
		for (unsigned int t = 0; t < k; t++)
			for (unsigned int s = 0; s < j; s++)
			{
				Spectrum v = 0.25f * (buffer[0]->Load(2 * s, 2 * t) + buffer[0]->Load(2 * s + 1, 2 * t) + 
									  buffer[0]->Load(2 * s, 2 * t + 1) + buffer[0]->Load(2 * s + 1, 2 * t + 1));
				buffer[1]->Set(v, s, t);
			}
		m_sOffsets[i] = off;
		off += j * k;
		a_Out.Write(buffer[1]->d(), j * k * sizeof(RGBCOL));
		swapk(buffer[0], buffer[1]);
	}
	a_Out.Write(m_sOffsets, sizeof(m_sOffsets));
	for (int i = 0; i < MTS_MIPMAP_LUT_SIZE; ++i)
	{
		float r2 = (float)i / (float)(MTS_MIPMAP_LUT_SIZE - 1);
		float val = math::exp(-2.0f * r2) - math::exp(-2.0f);
		a_Out << val;
	}
	data.Free();
	tmpData.Free();
}

KernelMIPMap MIPMap::getKernelData()
{
	KernelMIPMap r;
	r.m_pDeviceData = m_pDeviceData;
	r.m_pHostData = m_pHostData;
	r.m_uType = m_uType;
	r.m_uWrapMode = m_uWrapMode;
	r.m_uFilterMode = m_uFilterMode;
	r.m_uWidth = m_uWidth;
	r.m_uHeight = m_uHeight;
	r.m_fDim = Vec2f((float)m_uWidth - 1, (float)m_uHeight - 1);
	r.m_uLevels = m_uLevels;
	memcpy(r.m_sOffsets, m_sOffsets, sizeof(m_sOffsets));
	memcpy(r.m_weightLut, m_weightLut, sizeof(m_weightLut));
	return r;
}

}
