#pragma once
#include <Engine/SpatialStructures/Grid/SpatialGridList.h>
#include <Math/Spectrum.h>
#include <Engine/DifferentialGeometry.h>
#include <Math/Compression.h>
#include <Kernel/TracerSettings.h>
#include <SceneTypes/Sensor.h>

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

	PARAMETER_KEY(float, GlobalRadiusScale)
	PARAMETER_KEY(bool, UseRadius_GlobalScale)

	PARAMETER_KEY(float, PixelFootprintScale)
	PARAMETER_KEY(bool, UseRadius_PixelFootprintSize)

	PARAMETER_KEY(float, PrevFrameAlpha)
	PARAMETER_KEY(bool, UsePreviousFrames)

private:
	SpatialGridList_Linked<path_entry> m_hitPointBuffer;
	float m_pixelRad;
	Sensor m_lastSensor;
	Vec3f* m_accumBuffer1, *m_accumBuffer2;
	int buf_w, buf_h;
	TracerParameterCollection m_paraSettings;
	bool m_firstPass;
public:
	PathSpaceFilteringBuffer(unsigned int numSamples)
		: m_hitPointBuffer(Vec3u(100), numSamples), m_accumBuffer1(0), m_accumBuffer2(0), buf_w(0), buf_h(0)
	{
		m_paraSettings << KEY_GlobalRadiusScale()					<< CreateInterval(1.0f, 0.0f, FLT_MAX)
					   << KEY_UseRadius_GlobalScale()				<< CreateSetBool(true)
					   << KEY_PixelFootprintScale()					<< CreateInterval(1.0f, 0.0f, FLT_MAX)
					   << KEY_UseRadius_PixelFootprintSize()		<< CreateSetBool(true)
					   << KEY_PrevFrameAlpha()						<< CreateInterval(0.2f, 0.0f, 1.0f)
					   << KEY_UsePreviousFrames()					<< CreateSetBool(true);
	}

	void Free()
	{
		m_hitPointBuffer.Free();
		if (m_accumBuffer1)
		{
			CUDA_FREE(m_accumBuffer1);
			CUDA_FREE(m_accumBuffer2);
		}
	}

	void ResizeHitPointBuffer(int N)
	{
		m_hitPointBuffer.Free();
		m_hitPointBuffer = SpatialGridList_Linked<path_entry>(Vec3u(100), N);
	}

	TracerParameterCollection& getParameterCollection()
	{
		return m_paraSettings;
	}

	void PrepareForRendering(Image& I, DynamicScene* scene);

	void StartFrame(unsigned int w, unsigned int h)
	{
		auto eye_hit_points = m_hitPointBuffer.getHashGrid().getAABB();
		m_pixelRad = eye_hit_points.Size().length() / max(w, h);
		if (buf_w != w || buf_h != h)
		{
			buf_w = w; buf_h = h;
			if (m_accumBuffer1)
			{
				CUDA_FREE(m_accumBuffer1);
				CUDA_FREE(m_accumBuffer2);
			}
			CUDA_MALLOC(&m_accumBuffer1, sizeof(Vec3f) * w * h);
			cudaMemset(m_accumBuffer1, 0, sizeof(Vec3f) * w * h);
			CUDA_MALLOC(&m_accumBuffer2, sizeof(Vec3f) * w * h);
			cudaMemset(m_accumBuffer2, 0, sizeof(Vec3f) * w * h);
		}
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