#pragma once
#include <StdAfx.h>
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Engine/Light.h>
#include <Engine/Sensor.h>
#include <Engine/Samples.h>
#include <Engine/DifferentialGeometry.h>
#include <Engine/Material.h>

namespace CudaTracerLib {

class PathVertex
{
public:
	virtual Vec3f getPos() const = 0;
	virtual NormalizedT<Vec3f> getNor() const = 0;
	virtual Frame getSys() const
	{
		Frame sys(getNor());
		return sys;
	}
	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next) = 0;
	Vec3f getPos_uv(const Vec2f& uv) const
	{
		return getPos() + getSys().toWorld(Vec3f(uv, 0));
	}
	virtual bool sameSurface(const TraceResult& r2) const = 0;
	virtual PathVertex* Clone() const = 0;

	static float G(const PathVertex* v1, const PathVertex* v2)
	{
		return CudaTracerLib::G(v1->getNor(), v2->getNor(), v1->getPos(), v2->getPos());
		//Vec3f w = v2->getPos() - v1->getPos();
		//float l = length(w);
		//w /= l;
		//float cos1 = dot(w, v1->getNor()), cos2 = -dot(w, v2->getNor());
		//return cos1 * cos2 / (l * l);
	}
};

class SurfacePathVertex : public PathVertex
{
public:
	DifferentialGeometry dg;
	bool hasSampledDelta;
	TraceResult res;

	SurfacePathVertex(const TraceResult& r)
		: res(r)
	{

	}

	virtual bool sameSurface(const TraceResult& r2) const
	{
		return res.getNodeIndex() == r2.getNodeIndex();// && res.getTriIndex() == r2.getTriIndex();
	}

	virtual Vec3f getPos() const
	{
		return dg.P;
	}

	virtual NormalizedT<Vec3f> getNor() const
	{
		return dg.sys.n;
	}

	virtual Frame getSys() const
	{
		return dg.sys;
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		BSDFSamplingRecord bRec(dg);
		bRec.eta = 1.0f;
		bRec.sampledType = 0;
		bRec.typeMask = ETypeCombinations::EAll;
		auto wi = normalize(prev->getPos() - dg.P), wo = normalize(next->getPos() - dg.P);
		bRec.wi = bRec.dg.toLocal(wi);
		bRec.wo = bRec.dg.toLocal(wo);
		return res.getMat().bsdf.f(bRec, hasSampledDelta ? EDiscrete : ESolidAngle);
	}

	virtual PathVertex* Clone() const
	{
		return new SurfacePathVertex(*this);
	}
};

class LightPathVertex : public PathVertex
{
public:
	Light light;
	Vec3f p;
	NormalizedT<Vec3f> n;

	LightPathVertex()
	{

	}

	virtual bool sameSurface(const TraceResult& r2) const
	{
		if (!light.Is<DiffuseLight>())
			return false;
		return r2.getNodeIndex() == light.As<DiffuseLight>()->m_uNodeIdx && r2.LightIndex() != UINT_MAX;
	}

	virtual Vec3f getPos() const
	{
		return p;
	}

	virtual NormalizedT<Vec3f> getNor() const
	{
		return n;
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		PositionSamplingRecord pRec(p, n, 0);
		DirectionSamplingRecord dRec(normalize(prev->getPos() - p));
		return light.evalPosition(pRec) * light.evalDirection(dRec, pRec);
	}

	virtual LightPathVertex* Clone() const
	{
		return new LightPathVertex(*this);
	}
};

class CameraPathVertex : public PathVertex
{
public:
	Sensor sensor;
	Vec3f p;
	NormalizedT<Vec3f> n;

	CameraPathVertex()
	{

	}

	virtual bool sameSurface(const TraceResult& r2) const
	{
		return false;
	}

	virtual Vec3f getPos() const
	{
		return p;
	}

	virtual NormalizedT<Vec3f> getNor() const
	{
		return n;
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		PositionSamplingRecord pRec(p, n, 0, EDiscrete);
		DirectionSamplingRecord dRec(normalize(next->getPos() - p));
		return sensor.evalPosition(pRec) * sensor.evalDirection(dRec, pRec);
	}

	virtual CameraPathVertex* Clone() const
	{
		return new CameraPathVertex(*this);
	}
};

class Path
{
public:
	std::vector<PathVertex*> vertices;

	Path Clone() const
	{
		Path p;
		p.vertices.resize(vertices.size());
		for (size_t i = 0; i < vertices.size(); i++)
			p.vertices[i] = vertices[i]->Clone();
		return p;
	}

	CameraPathVertex* cameraVertex() const
	{
		return (CameraPathVertex*)vertices[0];
	}

	LightPathVertex* lightVertex() const
	{
		return (LightPathVertex*)vertices.back();
	}

	size_t k() const
	{
		return vertices.size() - 1;
	}

	Spectrum I() const
	{
		return cameraVertex()->eval(0, vertices[1]);
	}

	Spectrum L_e() const
	{
		return lightVertex()->eval(vertices[k() - 1], 0);
	}

	Spectrum f_i(size_t i) const
	{
		return vertices[i]->eval(i > 0 ? vertices[i - 1] : 0, i < k() ? vertices[i + 1] : 0);
	}

	Spectrum f() const
	{
		Spectrum s = I() * PathVertex::G(vertices[0], vertices[1]);
		for (size_t i = 1; i <= k() - 1; i++)
		{
			s *= f_i(i) * PathVertex::G(vertices[i], vertices[i + 1]);
		}
		return s * L_e();
	}

	void applyDiffX(const float* dXF)
	{
		size_t dXSize = k() + 1;
		const Vec2f* dX = (const Vec2f*)dXF;
		Vec3f* pos = (Vec3f*)alloca(sizeof(Vec3f) * dXSize);
		size_t x_i_changed = 0;
		for (size_t i = 0; i < dXSize; i++)
		{
			pos[i] = vertices[i]->getPos_uv(dX[i]);
			x_i_changed |= size_t(dX[i].lenSqr() != 0) << i;
		}
		for (size_t i = 0; i < dXSize - 1; i++)
		{
			if (((x_i_changed >> i) & 3) == 0)
				continue;
			Ray r(vertices[i]->getPos(), normalize(pos[i + 1] - vertices[i]->getPos()));
			TraceResult r2 = traceRay(r);
			float rho = length(r(r2.m_fDist) - pos[i + 1]);//diff to proposed
			if (rho > r2.m_fDist / 2 || !vertices[i + 1]->sameSurface(r2))
			{
				pos[i + 1] = vertices[i + 1]->getPos();
				continue;
			}

			if (i == dXSize - 2)//modifying light vertex
			{
				DifferentialGeometry dg;
				r2.fillDG(dg);
				LightPathVertex* v = (LightPathVertex*)vertices[i + 1];
				v->n = dg.sys.n;
				v->p = r(r2.m_fDist);
			}
			else
			{
				SurfacePathVertex* v = (SurfacePathVertex*)vertices[i + 1];
				v->res = r2;
				r2.fillDG(v->dg);
				v->dg.P = r(r2.m_fDist);
			}
		}
	}

	void applyDiffX(const std::vector<Vec2f>& dX)
	{
		if (dX.size() != k() + 1 || dX.size() > 64)
			throw std::runtime_error("invalid dX!");
		applyDiffX(dX);
	}
};

void TracePath(NormalizedT<Ray> r, std::vector<PathVertex*>& p, int N2, ETransportMode mode, CudaRNG& rng);

Path ConnectPaths(const std::vector<PathVertex*>& cameraPath, const std::vector<PathVertex*>& emitterPath);

void ConstructPath(const Vec2i& pixel, Path& P, int s, int t);

inline void ConstructPath(int x, int y, Path& P, int s, int t)
{
	ConstructPath(Vec2i(x, y), P, s, t);
}

std::vector<Vec2f> DifferientiatePath(Path& P);

void OptimizePath(Path& P);

}