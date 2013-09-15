#pragma once

#include "..\MathTypes.h"

struct MaterialEntry {
	const char *name;
	float sigmaS[3];
	float sigmaA[3];
	float g[3];
	float eta;
};

class e_MaterialLibrary
{
private:
	static MaterialEntry* getMat(const char* name);
public:
	static Spectrum getSigmaS(const char* name)
	{
		MaterialEntry* m = getMat(name);
		if(m)
			return Spectrum(m->sigmaS[0], m->sigmaS[1], m->sigmaS[2]) * 100.0f;
		return 0.0f;
	}
	static Spectrum getSigmaA(const char* name)
	{
		MaterialEntry* m = getMat(name);
		if(m)
			return Spectrum(m->sigmaA[0], m->sigmaA[1], m->sigmaA[2]) * 100.0f;
		return 0.0f;
	}
	static Spectrum getG(const char* name)
	{
		MaterialEntry* m = getMat(name);
		if(m)
			return Spectrum(m->g[0], m->g[1], m->g[2]);
		return 0.0f;
	}
	static float getEta(const char* name)
	{
		MaterialEntry* m = getMat(name);
		if(m)
			return m->eta;
		return 0.0f;
	}
};