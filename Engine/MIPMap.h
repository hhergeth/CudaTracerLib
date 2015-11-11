#pragma once
#include "MIPMap_device.h"
#include <Base/FixedString.h>

namespace CudaTracerLib {

class IInStream;
class FileOutputStream;

class MIPMap
{
	unsigned int* m_pDeviceData;
	unsigned int* m_pHostData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uBpp;
	unsigned int m_uLevels;
	unsigned int m_uSize;
	Texture_DataType m_uType;
	ImageWrap m_uWrapMode;
	unsigned int m_sOffsets[MAX_MIPS];
	float m_weightLut[MTS_MIPMAP_LUT_SIZE];
public:
	ImageFilter m_uFilterMode;
	std::string m_pPath;
	MIPMap() { m_pDeviceData = 0; m_uWidth = m_uHeight = m_uBpp = UINT_MAX; }
	MIPMap(const std::string& a_InputFile, IInStream& a_In);
	void Free();
	static void CompileToBinary(const std::string& a_InputFile, FileOutputStream& a_Out, bool a_MipMap);
	static void CompileToBinary(const std::string& in, const std::string& out, bool a_MipMap);
	static void CreateSphericalSkydomeTexture(const std::string& front, const std::string& back, const std::string& left, const std::string& right, const std::string& top, const std::string& bottom, const std::string& outFile);
	static void CreateRelaxedConeMap(const std::string& a_InputFile, FileOutputStream& Out);
	KernelMIPMap getKernelData();
	unsigned int getNumMips()
	{
		return m_uLevels;
	}
	unsigned int getBufferSize()
	{
		return m_uSize;
	}
};

}