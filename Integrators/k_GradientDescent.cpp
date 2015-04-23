#include <StdAfx.h>
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"
#include "k_PrimTracer.h"

class PathVertex
{
public:
	virtual Vec3f getPos() const = 0;
	virtual Vec3f getNor() const = 0;
	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next) = 0;
	virtual Vec3f getPos_uv(const Vec2f& uv) const
	{
		Frame sys(getNor());
		return getPos() + sys.toWorld(Vec3f(uv, 0));
	}
	virtual bool sameSurface(const TraceResult& r2) const = 0;

	static float G(const PathVertex* v1, const PathVertex* v2)
	{
		return ::G(v1->getNor(), v2->getNor(), v1->getPos(), v2->getPos());
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
		return res.getNodeIndex() == r2.getNodeIndex() && res.getTriIndex() == r2.getTriIndex();
	}

	virtual Vec3f getPos() const
	{
		return dg.P;
	}

	virtual Vec3f getNor() const
	{
		return dg.sys.n;
	}

	virtual Vec3f getPos_uv(const Vec2f& uv) const
	{
		return dg.P + dg.sys.toWorld(Vec3f(uv, 0));
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		BSDFSamplingRecord bRec(dg);
		bRec.eta = 1.0f;
		bRec.sampledType = 0;
		bRec.typeMask = ETypeCombinations::EAll;
		Vec3f wi = normalize(prev->getPos() - dg.P), wo = normalize(next->getPos() - dg.P);
		bRec.wi = normalize(bRec.dg.toLocal(-wi));
		bRec.wo = normalize(bRec.dg.toLocal(wo));
		return res.getMat().bsdf.f(bRec, hasSampledDelta ? EDiscrete : ESolidAngle);
	}
};

class LightPathVertex : public PathVertex
{
public:
	const e_KernelLight* light;
	Vec3f p, n;

	LightPathVertex()
	{

	}

	virtual bool sameSurface(const TraceResult& r2) const
	{
		if (!light->Is<e_DiffuseLight>())
			return false;
		return r2.getNodeIndex() == light->As<e_DiffuseLight>()->m_uNodeIdx;
	}

	virtual Vec3f getPos() const
	{
		return p;
	}

	virtual Vec3f getNor() const
	{
		return n;
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		PositionSamplingRecord pRec(p, n, 0);
		DirectionSamplingRecord dRec(normalize(next->getPos() - p));
		return light->evalPosition(pRec) * light->evalDirection(dRec, pRec);
	}
};

class CameraPathVertex : public PathVertex
{
public:
	const e_Sensor* sensor;
	Vec3f p, n;

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

	virtual Vec3f getNor() const
	{
		return n;
	}

	virtual Spectrum eval(const PathVertex* prev, const PathVertex* next)
	{
		PositionSamplingRecord pRec(p, n, 0, EDiscrete);
		DirectionSamplingRecord dRec(normalize(next->getPos() - p));
		return sensor->evalPosition(pRec) * sensor->evalDirection(dRec, pRec);
	}
};

class Path
{
public:
	std::vector<PathVertex*> vertices;

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

	Spectrum f() const
	{
		Spectrum s = cameraVertex()->eval(0, vertices[1]) * PathVertex::G(vertices[0], vertices[1]);
		for (int i = 1; i < k() - 1; i++)
		{
			s *= vertices[i]->eval(vertices[i - 1], vertices[i + 1]) * PathVertex::G(vertices[i], vertices[i + 1]);
		}
		return s * lightVertex()->eval(0, vertices[k() - 1]);
	}

	void applyDiffX(const std::vector<Vec2f>& dX)
	{
		if (dX.size() != k() - 1 || dX.size() > 64)
			throw std::runtime_error("invalid dX!");
		Vec3f* pos = (Vec3f*)alloca(sizeof(Vec3f) * dX.size());
		unsigned long long x_i_changed = 0;
		for (size_t i = 0; i < dX.size() - 1; i++)
		{
			pos[i] = vertices[i]->getPos_uv(dX[i]);
			x_i_changed |= (dX[i].lenSqr() != 0) << i;
		}
		for (size_t i = 0; i < dX.size() - 1; i++)
		{
			if ((x_i_changed >> i) & 3 == 0)
				continue;
			Ray r(pos[i], normalize(pos[i + 1] - pos[i]));
			TraceResult r2 = k_TraceRay(r);
			float rho = length(r(r2.m_fDist) - pos[i + 1]);//diff to proposed
			if (rho > r2.m_fDist / 2 || !vertices[i + 1]->sameSurface(r2))
			{
				pos[i + 1] = vertices[i + 1]->getPos();
				continue;
			}

			if (rho > 1e-3f)
			{
				if (i == dX.size() - 2)//modifying light vertex
				{
					DifferentialGeometry dg;
					r2.fillDG(dg);
					LightPathVertex* v = (LightPathVertex*)vertices[i + 1];
					v->n = dg.sys.n;
					v->p = dg.P;
				}
				else
				{
					SurfacePathVertex* v = (SurfacePathVertex*)vertices[i + 1];
					v->res = r2;
					r2.fillDG(v->dg);
				}
			}
		}
	}
};

void TracePath(Ray r, std::vector<PathVertex*>& p, int N2, ETransportMode mode, CudaRNG& rng)
{
	TraceResult r2 = k_TraceRay(r);
	if (!r2.hasHit())
		return;
	int i = 1;
	while (r2.hasHit() && i++ < N2)
	{
		SurfacePathVertex v(r2);
		BSDFSamplingRecord bRec(v.dg);
		r2.getBsdfSample(r, bRec, mode, &rng);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		v.hasSampledDelta = (bRec.sampledType & ETypeCombinations::EDelta) != 0;
		p.push_back(new SurfacePathVertex(v));
		r = Ray(v.dg.P, bRec.getOutgoing());
		r2 = k_TraceRay(r);
	}
}

Path ConnectPaths(const std::vector<PathVertex*>& cameraPath, const std::vector<PathVertex*>& emitterPath)
{
	Path p;
	for (std::vector<PathVertex*>::const_iterator it1 = cameraPath.begin(); it1 != cameraPath.end(); ++it1)
	{
		p.vertices.push_back(*it1);
		for (std::vector<PathVertex*>::const_iterator it2 = emitterPath.begin(); it2 != emitterPath.end(); ++it2)
		{
			if (::V((*it1)->getPos(), (*it2)->getPos()))
			{
				int i = it2 - emitterPath.begin();
				for (int j = i; j >= 0; j--)
					p.vertices.push_back(emitterPath[j]);
				return p;
			}
		}
	}
	return p;
}

void ConstructPath(const Vec2i& pixel, Path& P)
{
	CudaRNG rng = g_RNGData();
	int maxSubPathLength = 6;

	std::vector<PathVertex*> sensorPath, emitterPath;
	Ray r;

	g_SceneData.m_Camera.sampleRay(r, Vec2f(0.5f), Vec2f(0.5f));
	CameraPathVertex* c_v = new CameraPathVertex();
	c_v->n = r.direction;
	r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	c_v->p = r.origin;
	c_v->sensor = &g_SceneData.m_Camera;
	sensorPath.push_back(c_v);
	TracePath(r, sensorPath, maxSubPathLength, ETransportMode::ERadiance, rng);

	float f = rng.randomFloat();
	Vec2f f2 = rng.randomFloat2();
	LightPathVertex* l_v = new LightPathVertex();
	l_v->light = g_SceneData.sampleLight(f, f2);
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;
	l_v->light->samplePosition(pRec, rng.randomFloat2());
	l_v->light->sampleDirection(dRec, pRec, rng.randomFloat2());
	l_v->n = pRec.n;
	l_v->p = pRec.p;
	emitterPath.push_back(l_v);
	TracePath(Ray(pRec.p, dRec.d), emitterPath, maxSubPathLength, ETransportMode::EImportance, rng);

	P = ConnectPaths(sensorPath, emitterPath);

	g_RNGData(rng);
}

void VisualizePath(Path& P, ITracerDebugger* debugger)
{
	debugger->StartNewPath(P.cameraVertex()->sensor, P.cameraVertex()->getPos(), Spectrum(1.0f));
	for (size_t i = 1; i < P.vertices.size() - 1; i++)
	{
		debugger->AppendVertex(ITracerDebugger::PathType::Camera, P.vertices[i]->getPos());
	}
}