#pragma once

#include <Math/Spectrum.h>
#include <string>

namespace CudaTracerLib {

struct MaterialEntry {
	const std::string name;
	float sigmaS[3];
	float sigmaA[3];
	float g[3];
	float eta;
};

class MaterialLibrary
{
private:
	static MaterialEntry* getMat(const std::string& name);
public:
	static bool hasMat(const std::string& name)
	{
		return getMat(name) != 0;
	}
	static Spectrum getSigmaS(const std::string& name)
	{
		MaterialEntry* m = getMat(name);
		if (m)
			return Spectrum(m->sigmaS[0], m->sigmaS[1], m->sigmaS[2]) * 100.0f;
		return 0.0f;
	}
	static Spectrum getSigmaA(const std::string& name)
	{
		MaterialEntry* m = getMat(name);
		if (m)
			return Spectrum(m->sigmaA[0], m->sigmaA[1], m->sigmaA[2]) * 100.0f;
		return 0.0f;
	}
	static Spectrum getG(const std::string& name)
	{
		MaterialEntry* m = getMat(name);
		if (m)
			return Spectrum(m->g[0], m->g[1], m->g[2]);
		return 0.0f;
	}
	static float getEta(const std::string& name)
	{
		MaterialEntry* m = getMat(name);
		if (m)
			return m->eta;
		return 0.0f;
	}
	static size_t getNumMats();
	static const std::string& getMatName(size_t idx);
};

}