#pragma once
#include <Engine/SpatialStructures/Grid/SpatialGridList.h>
#include <Math/Spectrum.h>
#include <Engine/DifferentialGeometry.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

class Image;
class DynamicScene;

class PathSpaceFilteringBuffer : public ISynchronizedBufferParent
{
public:
	struct path_entry
	{
		Vec3f p;
		RGBE Li;
		unsigned short nor;
		unsigned short wi;
	};
public:
	SpatialGridList_Linked<path_entry> m_hitPointBuffer;
	float m_pixelRad0;
	unsigned int m_numIteration;
public:
	PathSpaceFilteringBuffer(unsigned int numSamples)
		: m_hitPointBuffer(Vec3u(100), numSamples)
	{

	}
	void Free()
	{
		m_hitPointBuffer.Free();
	}

	void PrepareForRendering(Image& I, DynamicScene* scene);

	void StartIteration()
	{
		m_numIteration++;
		m_hitPointBuffer.ResetBuffer();
	}

	//wi is pointing away from the surface
	CUDA_DEVICE void add_sample(const DifferentialGeometry& dg, const NormalizedT<Vec3f>& wi, const Spectrum& Li)
	{
		path_entry e;
		e.Li = Li.toRGBE();
		e.nor = NormalizedFloat3ToUchar2(dg.sys.n);
		e.p = dg.P;
		e.wi = NormalizedFloat3ToUchar2(dg.toLocal(wi));
		m_hitPointBuffer.Store(dg.P, e);
	}

	void ComputePixelValues(Image& I, DynamicScene* scene);
};

}