#include <StdAfx.h>
#include "GradientDescent.h"
#include "PathDifferientials.h"
#include <assert.h>

namespace CudaTracerLib {

void TracePath(NormalizedT<Ray> r, std::vector<PathVertex*>& p, int N2, ETransportMode mode, Sampler& rng)
{
	TraceResult r2 = traceRay(r);
	if (!r2.hasHit())
		return;
	int i = 1;
	while (r2.hasHit() && i++ < N2)
	{
		SurfacePathVertex v(r2);
		BSDFSamplingRecord bRec(v.dg);
		r2.getBsdfSample(r, bRec, mode);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		v.hasSampledDelta = (bRec.sampledType & ETypeCombinations::EDelta) != 0;
		p.push_back(new SurfacePathVertex(v));
		r = NormalizedT<Ray>(v.dg.P, bRec.getOutgoing());
		r2 = traceRay(r);
	}
}

Path ConnectPaths(const std::vector<PathVertex*>& cameraPath, const std::vector<PathVertex*>& emitterPath, int s, int t)
{
	Path p;
	for (std::vector<PathVertex*>::const_iterator it1 = cameraPath.begin(); it1 != cameraPath.end(); ++it1)
	{
		p.vertices.push_back(*it1);
		for (std::vector<PathVertex*>::const_iterator it2 = emitterPath.begin(); it2 != emitterPath.end(); ++it2)
		{
			bool b1 = s == -1 || it1 - cameraPath.begin() == s, b2 = t == -1 || it2 - emitterPath.begin() == t;
			if (V((*it1)->getPos(), (*it2)->getPos()) && p.vertices.size() > 1 && b1 && b2)
			{
				int64_t i = it2 - emitterPath.begin();
				for (int64_t j = i; j >= 0; j--)
					p.vertices.push_back(emitterPath[j]);
				return p;
			}
		}
	}
	return p;
}

void ConstructPath(const Vec2i& pixel, Path& P, int s, int t)
{
	auto rng = g_SamplerData();
	int maxSubPathLength = 6;

	std::vector<PathVertex*> sensorPath, emitterPath;
	NormalizedT<Ray> r;

	g_SceneData.m_Camera.sampleRay(r, Vec2f(0.5f), Vec2f(0.5f));
	CameraPathVertex* c_v = new CameraPathVertex();
	c_v->n = r.dir();
	r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	c_v->p = r.ori();
	c_v->sensor = g_SceneData.m_Camera;
	sensorPath.push_back(c_v);
	TracePath(r, sensorPath, maxSubPathLength, ETransportMode::ERadiance, rng);

	float f = rng.randomFloat();
	Vec2f f2 = rng.randomFloat2();
	LightPathVertex* l_v = new LightPathVertex();
	l_v->light = *g_SceneData.sampleEmitter(f, f2);
	PositionSamplingRecord pRec;
	DirectionSamplingRecord dRec;
	l_v->light.samplePosition(pRec, Vec2f(1.0f / 3.0f));
	l_v->light.sampleDirection(dRec, pRec, rng.randomFloat2());
	l_v->n = pRec.n;
	l_v->p = pRec.p;
	emitterPath.push_back(l_v);
	TracePath(NormalizedT<Ray>(pRec.p, dRec.d), emitterPath, maxSubPathLength, ETransportMode::EImportance, rng);

	P = ConnectPaths(sensorPath, emitterPath, s, t);

	g_SamplerData(rng);
}

qMatrix<float, 1, 4> dG_du12_v12(const Path& P, size_t i)
{
	return CudaTracerLib::dG_du12_v12(P.vertices[i]->getPos(), P.vertices[i + 1]->getPos(), P.vertices[i]->getSys(), P.vertices[i + 1]->getSys());
}

qMatrix<float, 1, 4> dI_du12_v12(const Path& P)
{
	return qMatrix<float, 1, 4>::Zero();
}

qMatrix<float, 1, 4> dLe_du12_v12(const Path& P)
{
	return qMatrix<float, 1, 4>::Zero();
}

qMatrix<float, 1, 6> dfi_du123_v123(const Path& P, size_t i)
{
	return CudaTracerLib::dfi_diffuse_du123_v123(P.vertices[i - 1]->getPos(), P.vertices[i]->getPos(), P.vertices[i + 1]->getPos(), P.vertices[i - 1]->getSys(), P.vertices[i]->getSys(), P.vertices[i + 1]->getSys(), 1);
}

template<typename MAT, int L> inline void set(MAT& M, const qMatrix<float, 1, L>& v, int i, int j)
{
	for (int k = 0; k < L; k++)
	{
		M(i, j + k) = v(k);
	}
}
template<typename VEC> inline std::vector<Vec2f> to_vec(const VEC& v)
{
	std::vector<Vec2f> r;
	int N = VEC::SIZE::DIM;
	r.resize(N);
	for (int i = 0; i < N / 2; i++)
		r[i] = Vec2f(v(2 * i + 0), v(2 * i + 1));
	return r;
}

template<int N> struct diff_helper
{
	static std::vector<Vec2f> exec(Path& P)
	{
		if (N == P.k() + 1)
		{
			const int k = N - 1, n = 3 + 2 * (k - 1);
			float* a = (float*)alloca(sizeof(float) * n);
			a[0] = P.I().avg(); a[1] = PathVertex::G(P.vertices[0], P.vertices[1]); a[2] = P.L_e().avg();
			for (int j = 1; j < k; j++)
			{
				a[3 + 2 * (j - 1) + 0] = ((SurfacePathVertex*)P.vertices[j])->hasSampledDelta ? 1.0f : P.f_i(j).avg();
				a[3 + 2 * (j - 1) + 1] = PathVertex::G(P.vertices[j], P.vertices[j + 1]);
			}
			float lhs = 1, rhs = 1;
			for (int i = 1; i <= n - 1; i++)
				rhs *= a[i];
			qMatrix<float, 1, n> A;
			A.zero();
			for (int i = 0; i <= n - 1; i++)
			{
				rhs /= a[i];
				lhs *= a[i];
				A(i) = lhs * rhs;
			}

			qMatrix<float, n, (k + 1) * 2> M;
			M.zero();
			set(M, dI_du12_v12(P), 0, 0);
			set(M, dG_du12_v12(P, 0), 1, 0);
			set(M, dLe_du12_v12(P), 2, 2 * (k - 2));
			//std::cout << M.ToString("M") << "\n";
			for (int j = 1; j <= k - 1; j++)
			{
				set(M, dfi_du123_v123(P, j), 3 + 2 * (j - 1) + 0, 2 * (j - 1));
				set(M, dG_du12_v12(P, j), 3 + 2 * (j - 1) + 1, 2 * j);
			}
			//std::cout << M.ToString("M") << "\n";

			qMatrix<float, 1, (k + 1) * 2> dX = A * M;
			//std::cout << dX.ToString("dX") << "\n";
			return to_vec(dX);
		}
		else return diff_helper<N - 1>::exec(P);
	}
};

template<> struct diff_helper < 0 >
{
	static std::vector<Vec2f> exec(Path& P)
	{
		return std::vector<Vec2f>();
	}
};

std::vector<Vec2f> DifferientiatePath(Path& P)
{
	return diff_helper<16>::exec(P);
}
/*
void OptimizePath(Path& P)
{
class ipprob : public TNLP
{
Path& P;
const float bounds;
public:
ipprob(Path& p)
: P(p), bounds(10)
{

}
virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
Index& nnz_h_lag, IndexStyleEnum& index_style)
{
n = P.vertices.size() * 2;
m = 1;
nnz_jac_g = 1;
nnz_h_lag = n* n;
index_style = IndexStyleEnum::C_STYLE;
return true;
}
virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
Index m, Number* g_l, Number* g_u)
{
for (int i = 0; i < n; i++) {
x_l[i] = -bounds;
x_u[i] = +bounds;
}
g_l[0] = g_u[0] = 0;
return true;
}
virtual bool get_starting_point(Index n, bool init_x, Number* x,
bool init_z, Number* z_L, Number* z_U,
Index m, bool init_lambda,
Number* lambda)
{
if (init_x)
for (int i = 0; i < n; i++)
x[i] = 0;
assert(init_z == false && init_lambda == false);

return true;
}
virtual bool eval_f(Index n, const Number* x, bool new_x,
Number& obj_value)
{
Path p = P.Clone();
float* d = (float*)alloca(sizeof(float) * n);
for (int i = 0; i < n; i++)
d[i] = (float)x[i];
p.applyDiffX(d);
obj_value = p.f().average();
return true;
}
virtual bool eval_grad_f(Index n, const Number* x, bool new_x,
Number* grad_f)
{
Path p = P.Clone();
float* d = (float*)alloca(sizeof(float) * n);
for (int i = 0; i < n; i++)
d[i] = (float)x[i];
p.applyDiffX(d);
std::vector<Vec2f> g = DifferientiatePath(p);
assert(g.size() == 2 * n);
for (size_t i = 0; i < g.size(); i++)
{
grad_f[2 * i + 0] = g[i].x;
grad_f[2 * i + 1] = g[i].y;
}
return true;
}
virtual bool eval_g(Index n, const Number* x, bool new_x,
Index m, Number* g)
{
assert(m == 1);
g[0] = 0;
return true;
}
virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
Index m, Index nele_jac, Index* iRow,
Index *jCol, Number* values)
{
assert(m == 1);
if (values == 0)
iRow[0] = jCol[0] = 0;
else values[0] = 0;
return true;
}
virtual void finalize_solution(SolverReturn status,
Index n, const Number* x, const Number* z_L, const Number* z_U,
Index m, const Number* g, const Number* lambda,
Number obj_value,
const IpoptData* ip_data,
IpoptCalculatedQuantities* ip_cq)
{
float* d = (float*)alloca(sizeof(float) * n);
for (int i = 0; i < n; i++)
d[i] = (float)x[i];
P.applyDiffX(d);
}
};

ipprob p(P);
SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
app->Options()->SetNumericValue("tol", 1e-9);
app->Options()->SetStringValue("mu_strategy", "adaptive");
app->Options()->SetStringValue("output_file", "ipopt.out");
Ipopt::ApplicationReturnStatus status = app->Initialize();
status = app->OptimizeTNLP(&p);
}
*/

}