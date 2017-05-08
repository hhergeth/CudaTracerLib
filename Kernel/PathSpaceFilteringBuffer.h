#pragma once
#include <Engine/SpatialStructures/Grid/SpatialGridList.h>
#include <Math/Spectrum.h>
#include <Engine/DifferentialGeometry.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

class Image;
class DynamicScene;
struct DeviceDepthImage;

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

	void StartFrame(unsigned int w, unsigned int h, float rad_scale)
	{
		auto eye_hit_points = m_hitPointBuffer.getHashGrid().getAABB();
		m_pixelRad0 = eye_hit_points.Size().length() / max(w, h) * rad_scale;
		m_numIteration = 0;
	}

	void StartIteration()
	{
		m_numIteration++;
		m_hitPointBuffer.ResetBuffer();
	}

	//wi is pointing away from the surface and in local(!) coordinates
	CUDA_DEVICE void add_sample(const DifferentialGeometry& dg, const NormalizedT<Vec3f>& wi_local, const Spectrum& Li)
	{
		path_entry e;
		e.Li = Li.toRGBE();
		e.nor = NormalizedFloat3ToUchar2(dg.sys.n);
		e.p = dg.P;
		e.wi = NormalizedFloat3ToUchar2(wi_local);
		m_hitPointBuffer.Store(dg.P, e);
	}

	void ComputePixelValues(Image& I, DynamicScene* scene, DeviceDepthImage* depthImage = 0);
};

}