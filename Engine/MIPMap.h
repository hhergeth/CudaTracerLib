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
	MIPMap() = default;
	CTL_EXPORT MIPMap(const std::string& a_InputFile, IInStream& a_In);
	CTL_EXPORT void Free();
	CTL_EXPORT static void CompileToBinary(const std::string& a_InputFile, FileOutputStream& a_Out, bool a_MipMap);
	CTL_EXPORT static void CompileToBinary(const std::string& in, const std::string& out, bool a_MipMap);
	CTL_EXPORT static void CreateSphericalSkydomeTexture(const std::string& front, const std::string& back, const std::string& left, const std::string& right, const std::string& top, const std::string& bottom, const std::string& outFile);
	CTL_EXPORT static void CreateRelaxedConeMap(const std::string& a_InputFile, FileOutputStream& Out);
	CTL_EXPORT KernelMIPMap getKernelData();
	unsigned int getNumMips() const
	{
		return m_uLevels;
	}
	unsigned int getBufferSize() const
	{
		return m_uSize;
	}
};

}