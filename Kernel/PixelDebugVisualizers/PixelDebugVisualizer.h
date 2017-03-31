#pragma once

#include <map>
#include <string>
#include <limits>
#include <Base/SynchronizedBuffer.h>
#include <Math/Vector.h>
#include <Math/Spectrum.h>
#include <Math/Frame.h>

namespace CudaTracerLib {

class Image;

class IDebugDrawer
{
public:
	virtual void DrawLine(const Vec3f& p1, const Vec3f& p2, const Spectrum& col = Spectrum(1, 0, 0)) const = 0;
	virtual void DrawEllipse(const Vec3f& p, const NormalizedT<Vec3f>& t1, const NormalizedT<Vec3f>& t2, float l1, float l2, const Spectrum& col = Spectrum(1, 0, 0)) const
	{
		//draw ellipse
		const int N = 32;

		float C = 2.0f * PI * math::sqrt((l1 * l1 + l2 * l2) / 2.0f);
		for (int i = 0; i < N; i++)
		{
			float t_1 = (i) / float(N - 1) * C, t_2 = (i + 1) / float(N - 1) * C;
			Vec2f c_1(l1 * math::cos(t_1), l2 * math::sin(t_1)),
				  c_2(l1 * math::cos(t_2), l2 * math::sin(t_2));

			auto p_1 = p + c_1.x * t1 + c_1.y * t2,
				 p_2 = p + c_2.x * t1 + c_2.y * t2;

			DrawLine(p_1, p_2, col);
		}

		//draw axes
		DrawLine(p + t1 * l1, p - t1 * l1, col);
		DrawLine(p + t2 * l2, p - t2 * l2, col);
	}
	virtual void DrawEllipsoid(const Vec3f& p, const NormalizedT<Vec3f>& t1, const NormalizedT<Vec3f>& t2, const NormalizedT<Vec3f>& t3, float l1, float l2, float l3, const Spectrum& col = Spectrum(1, 0, 0)) const
	{
		DrawEllipse(p, t1, t2, l1, l2, col);
		DrawEllipse(p, t1, t3, l1, l3, col);
		DrawEllipse(p, t2, t3, l2, l3, col);
	}
	virtual void DrawCone(const Vec3f& p, const NormalizedT<Vec3f>& d, float theta, float length, const Spectrum& col = Spectrum(1, 0, 0)) const
	{
		const int N = 32;

		auto c = p + d * length;
		float rad = length * math::tan(theta / 2.0f);
		Frame sys(d);
		for (int i = 0; i < N; i++)
		{
			float t_1 = (i) / float(N - 1) * rad, t_2 = (i + 1) / float(N - 1) * rad;
			Vec2f c_1(rad * math::cos(t_1), rad * math::sin(t_1)),
				  c_2(rad * math::cos(t_2), rad * math::sin(t_2));

			auto p_1 = c + sys.toWorld(Vec3f(c_1.x, c_1.y, 0.0f)),
				 p_2 = c + sys.toWorld(Vec3f(c_2.x, c_2.y, 0.0f));

			DrawLine(p_1, p_2, col);
			DrawLine(p, p_1);
		}
	}
};

class IPixelDebugVisualizer
{
protected:
	unsigned int m_width, m_height;
	std::string m_name;
	IPixelDebugVisualizer(const std::string& name)
		: m_name(name)
	{
		m_width = m_height = std::numeric_limits<unsigned int>::max();
	}
public:
	virtual ~IPixelDebugVisualizer()
	{

	}
	virtual void Free() = 0;
	virtual void Clear() = 0;
	virtual void Visualize(Image& img) = 0;
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer) = 0;
	virtual void CopyFromGPU() = 0;
	virtual void Resize(unsigned int w, unsigned int h)
	{
		m_width = w;
		m_height = h;
	}
	virtual const std::string& getName() const
	{
		return m_name;
	}

	enum class FeatureVisualizer
	{
		Vertex,
		Edge,
	};
	virtual void VisualizeFeatures(const IDebugDrawer& drawer, FeatureVisualizer features);
};

template<typename T> class PixelDebugVisualizerBase : public IPixelDebugVisualizer, public ISynchronizedBufferParent
{
protected:
	float m_uniform_scale;
	SynchronizedBuffer<T> m_buffer;
	PixelDebugVisualizerBase(const std::string& name)
		: IPixelDebugVisualizer(name), ISynchronizedBufferParent(m_buffer), m_buffer(1), m_uniform_scale(1)
	{
	}

public:
	enum class NormalizationType
	{
		None,
		Adaptive,
		Range,
	};
	struct NormalizationData
	{
		NormalizationType type;
		T min, max;

		NormalizationData()
		{
			type = NormalizationType::None;
		}
	};
	NormalizationData m_normalizationData;

	virtual void Free()
	{
		m_buffer.Free();
	}

	virtual void Resize(unsigned int w, unsigned int h)
	{
		m_buffer.Resize(w * h);
		IPixelDebugVisualizer::Resize(w, h);
	}

	virtual void Clear()
	{
		m_buffer.Memset((unsigned char)0);
	}

	virtual void CopyFromGPU()
	{
		m_buffer.setOnGPU();
		m_buffer.Synchronize();
	}

	CUDA_FUNC_IN T& operator()(unsigned int x, unsigned int y)
	{
		return m_buffer[y * m_width + x];
	}

	CUDA_FUNC_IN const T& operator()(unsigned int x, unsigned int y) const
	{
		return m_buffer[y * m_width + x];
	}

	CUDA_FUNC_IN T getScaledValue(unsigned int x, unsigned int y) const
	{
		return operator()(x, y) * m_uniform_scale;
	}

	void setScale(float f)
	{
		m_uniform_scale = f;
	}
};

template<typename T> class PixelDebugVisualizer : public PixelDebugVisualizerBase<T>
{
};

class PixelDebugVisualizerManager
{
	std::map<std::string, IPixelDebugVisualizer*> m_visualizers;
	unsigned int m_width, m_height;
public:
	template<typename T> PixelDebugVisualizer<T>& findOrCreate(const std::string& name)
	{
		auto it = m_visualizers.find(name);
		if (it != m_visualizers.end())
		{
			auto* ptr = dynamic_cast<PixelDebugVisualizer<T>*>(it->second);
			if (it != m_visualizers.end() && ptr)
				return *ptr;
		}

		auto* ptr = new PixelDebugVisualizer<T>(name);
		ptr->Resize(m_width, m_height);
		m_visualizers[name] = ptr;
		return *ptr;
	}

	template<typename F> void iterateAllVisualizers(F clb) const
	{
		for (auto ent : m_visualizers)
		{
			clb(ent.second);
		}
	}

	void Resize(unsigned int w, unsigned int h)
	{
		m_width = w;
		m_height = h;
		for (auto ent : m_visualizers)
			ent.second->Resize(w, h);
	}

	void Free()
	{
		for (auto ent : m_visualizers)
			delete ent.second;
	}

	void ClearAll()
	{
		for (auto ent : m_visualizers)
			ent.second->Clear();
	}

	void CopyFromGPU()
	{
		for (auto ent : m_visualizers)
			ent.second->CopyFromGPU();
	}
};

}

#include "FloatPixelDebugVisualizer.h"
#include "Vec2fPixelDebugVisualizer.h"
#include "Vec3fPixelDebugVisualizer.h"