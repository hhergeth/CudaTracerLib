#pragma once

#include "Base\FrameworkInterop.h"
#include "Engine\e_FPCamera.h"
#include "Engine\e_DynamicScene.h"
#include "Kernel\k_Tracer.h"
#include "Kernel\k_PrimTracer.h"
#include "Kernel\k_PathTracer.h"
#include "Kernel\k_sPpmTracer.h"

inline void line(float3 a, float3 b, float4x4& vp, FW::GLContext* O, float3 col)
{
	RGBCOL q = Float3ToCOLORREF(col);
	float4 q0 = vp * make_float4(a, 1), q1 = vp * make_float4(b, 1);

	O->strokeLine(*(FW::Vec4f*)&q0, *(FW::Vec4f*)&q1, *(int*)&q);
}

inline void rect(float3 a, float3 b, float3 c, float3 d, float4x4& vp, FW::GLContext* O, float3 col)
{
	line(a, b, vp, O, col);
	line(b, c, vp, O, col);
	line(c, d, vp, O, col);
	line(d, a, vp, O, col);
}

void plotBox(AABB& box, float4x4& vp, FW::GLContext* O, float3 col)
{
	float3 d = box.maxV - box.minV;
#define A(x,y,z) make_float3(x,y,z) * d + box.minV
	float3 v[8] = {A(0,0,0), A(1,0,0), A(1,0,1), A(0,0,1),
				   A(0,1,0), A(1,1,0), A(1,1,1), A(0,1,1)};
#define B(x1,x2,x3,x4) rect(v[x1], v[x2], v[x3], v[x4], vp, O, col);

	B(0,1,2,3)
	B(4,5,6,7)
	B(0,4,5,1)
	B(2,6,7,3)

#undef B
}

void plot(e_BVHNodeData* node, e_BVHNodeData* nodes, uint depth, float4x4& vp, FW::GLContext* O)
{
	float f = 1.0f - (float)depth / 60.0f;
	float3 col = make_float3(f,0,0);
	glColor3f(f,0,0);
	plotBox(node->getLeft(), vp, O, col);
	plotBox(node->getRight(), vp, O, col);
	int2 c = node->getChildren();
	if(c.x > 0)
		plot(nodes + c.x, nodes, depth + 1, vp, O);
	if(c.y > 0)
		plot(nodes + c.y, nodes, depth + 1, vp, O);
}

void plot(int i, e_BVHNodeData* nodes, uint depth, float4x4& mat, float4x4& vp, FW::GLContext* O)
{
	float3 c = make_float3(0,1,0);
	AABB l, r;
	int2 d = nodes[i / 4].getChildren();
	nodes[i / 4].getBox(l, r);
	plotBox(l.Transform(mat), vp, O, c);
	plotBox(r.Transform(mat), vp, O, c);
	if(d.x > 0)
		plot(d.x, nodes, depth + 1, mat, vp, O);
	if(d.y > 0)
		plot(d.y, nodes, depth + 1, mat, vp, O);
}

class Renderer
{
public:
	uint m_uState;
	k_Tracer* m_pTracer_0;
	k_Tracer* m_pTracer_1;
	bool oldMove0, oldMove1;
public:
	Renderer(bool stdTracers = true)
	{
		if(stdTracers)
			setTracers(new k_PrimTracer(), new k_sPpmTracer());
			//setTracers(new k_PrimTracer(), new k_PathTracer());
		else setTracers(0, 0);
		oldMove0 = oldMove1 = false;
		m_uState = 1;
	}
	~Renderer()
	{
		delete m_pTracer_0;
		delete m_pTracer_1;
	}
	void setTracer(k_Tracer* t, int i)
	{
		*(i ? &m_pTracer_1 : &m_pTracer_0) = t;
		oldMove0 = oldMove1 = true;
	}
	void setTracers(k_Tracer* t0, k_Tracer* t1)
	{
		oldMove0 = oldMove1 = true;
		m_pTracer_0 = t0;
		m_pTracer_1 = t1;
	}
	void InitializeScene(e_DynamicScene* S, e_Camera* C)
	{
		m_pTracer_0->InitializeScene(S, C);
		m_pTracer_1->InitializeScene(S, C);
	}
	void Resize(int w, int h)
	{
		oldMove0 = oldMove1 = true;
		m_pTracer_0->Resize(w, h);
		m_pTracer_1->Resize(w, h);
	}
	void Toggle()
	{
		m_uState = m_uState == 1 ? 2 : 1;
		oldMove0 = oldMove1 = true;
	}
	void Render(FW::GLContext* gl, e_DynamicScene* S, e_Camera* C, FW::Image* I, bool a_DrawSceneBvh, bool a_DrawObjectBvhs, const e_Node* N)
	{
		bool& b = m_uState == 1 ? oldMove0 : oldMove1;
		bool moved = C->Update() || b;
		b = false;
		getCurrent()->DoPass((RGBCOL*)I->getBuffer().getMutableCudaPtr(), moved);

		FW_ASSERT(gl);
		FW::Mat4f oldXform = gl->setVGXform(FW::Mat4f());
		glPushAttrib(GL_ENABLE_BIT);
		glDisable(GL_DEPTH_TEST);
		gl->drawImage(*I, FW::Vec2f(0.0f), 0.5f, false);
		gl->setVGXform(oldXform);
		glPopAttrib();
		
		float4x4 vpq = C->getViewProjection();
		if(m_uState == 1)
		{
			AABB box = S->getKernelSceneData().m_sBox;
			float eps = Distance(box.maxV, box.minV) / 100.0f;
			for(int i = 0; i < S->getLightCount(); i++)
				plotBox(S->getLights(i)->getBox(eps), vpq, gl, make_float3(1,0,0));
			for(int i = 0; i < S->getVolumes().getLength(); i++)
			{
				AABB box = S->getVolumes()(i)->WorldBound();
				plotBox(box, vpq, gl, make_float3(1,1,0));
			}
		}

		if(!ShowGui())
			return;

		float4x4 vp = C->getViewProjection();
		if(N != 0)
			plotBox(N->getWorldBox(), vp, gl, make_float3(0,0,1));
		if(a_DrawSceneBvh)
		{
			//plot(S->getSceneBVH()->m_pNodes->getHost(0), S->getSceneBVH()->m_pNodes->getHost(0), 0, vp, gl);
		}
		//for(int i = 0; i < g_pScene->getNodeCount(); i++)
		//	plotBox(g_pScene->getNodes()[i].getWorldBox());
		if(a_DrawObjectBvhs)
		{
			for(int i = 0; i < S->getNodeCount(); i++)
			{
				unsigned int j = S->getNodes()[i].m_pMesh->getKernelData().m_uBVHNodeOffset / 4;
				plot(0, S->m_pBVHStream->getHost(j), 0, S->getNodes()[i].getWorldMatrix(), vp, gl);
			}
		}
	}
	void Render(e_DynamicScene* S, e_Camera* C, FW::Image* I)
	{
		bool& b = m_uState == 1 ? oldMove0 : oldMove1;
		bool moved = C->Update() || b;
		b = false;
		//k_Tracer* c = !i ? m_pTracer_0 : m_pTracer_1;
		k_Tracer* c = getCurrent();
		c->DoPass((RGBCOL*)I->getBuffer().getMutableCudaPtr(), moved);
	}
	k_Tracer* getCurrent()
	{
		return m_uState == 1 ? (k_Tracer*)m_pTracer_0 : (k_Tracer*)m_pTracer_1;
	}
	k_Tracer* getTracer(int a_KernelIndex)
	{
		return !a_KernelIndex ? m_pTracer_0 : m_pTracer_1;
	}
	void InvalidateCurrent()
	{
		if(m_uState == 1)
			oldMove0 = true;
		else if(m_uState == 2)
			oldMove1 = true;
	}
	bool ShowGui()
	{
		return m_uState == 1;
	}
	void DebugPixel(int2 p)
	{
		//getCurrent()->Debug(p);
		m_pTracer_0->Debug(p);
		m_pTracer_1->Debug(p);
	}
};

class Gizmo
{
public:
	float m_fAxisLength;
	float m_fPlaneLength;
	int m_uActiveComponent;
	float3 m_vOldHit;
	float3 m_cActiveColor;
public:
	Gizmo()
	{
		m_fAxisLength = 100;
		m_fPlaneLength = 100;
		m_uActiveComponent = 0;
		m_cActiveColor = make_float3(1,0,1);
	}
	bool HandleDown(e_Node* N, e_DynamicScene* S, e_Camera* C, int2 p, int2 s)
	{
		e_CameraData c;
		C->getData(c);
		float4x4 m = N->getWorldMatrix(), m2 = N->getInvWorldMatrix();
		Ray r = c.GenRay(p, s);
		float3 o = m.Translation(), d = o;
		float d_s[3] = { inter(m.Forward(), r, o), inter(m.Right(), r, o), inter(m.Up(), r, o)};
		float h = FLT_MAX;
		for(int i = 0; i < 3; i++)
		{
			float3 q = r(d_s[i]);
			if(d_s[i] < h && inter(q, m2))
			{
				m_vOldHit = q;
				m_uActiveComponent = 4 + i;
			}
		}
		return m_uActiveComponent;
	}
	void HandleMove(e_Node* N, e_DynamicScene* S, e_Camera* C, int2 p, int2 s)
	{
		e_CameraData c;
		C->getData(c);
		Ray r = c.GenRay(p, s);
		float4x4 m = N->getWorldMatrix();
		float v_Dist = inter(m_uActiveComponent == 4 ? m.Forward() : m_uActiveComponent == 5 ? m.Right() : m.Up(), r, m.Translation());
		float3 hit = r.origin + r.direction * v_Dist;
		float3 o = m.Translation(), d = hit - m_vOldHit;
		m.Translation(o + d * 0.2f);
		m_vOldHit = m.Translation();
		S->SetNodeTransform(m, N);
	}
	void HandleUp(e_Node* N, e_DynamicScene* S, const e_Camera* C, int2 p, int2 s)
	{
		m_uActiveComponent = 0;
	}
	void DrawPlanes(const e_Node* N, const e_Camera* C, FW::GLContext* O)
	{
		float4x4 m = N->getWorldMatrix(), vp = C->getViewProjection();
		float3 r = normalize(m.Right()), u = normalize(m.Up()), f = normalize(m.Forward());
		float3 o = m.Translation(), x = o + r * m_fPlaneLength, y = o + u * m_fPlaneLength, z = o + f * m_fPlaneLength;
		float3 xy = o + (r + u) * m_fPlaneLength, yz = o + (u + f) * m_fPlaneLength, zx = o + (f + r) * m_fPlaneLength;
		rect(o, y, xy, x, vp, O, m_uActiveComponent == 4 ? m_cActiveColor : make_float3(1,0,0));
		rect(o, y, yz, z, vp, O, m_uActiveComponent == 5 ? m_cActiveColor : make_float3(0,1,0));
		rect(o, x, zx, z, vp, O, m_uActiveComponent == 6 ? m_cActiveColor : make_float3(0,0,1));
	}
private:
	bool inter(float3 p, float4x4& m)
	{
		float3 scale = make_float3(length(!m.X), length(!m.Y), length(!m.Z));
		float3 q = (m * p) / scale;
		float e = -EPSILON;
		return m_fPlaneLength >= q.x && q.x >= e && m_fPlaneLength >= q.y && q.y >= e && m_fPlaneLength >= q.z && q.z >= e;
	}
	float inter(float3 n, Ray r, float3 o)
	{
		float q = dot(n, o);
		return (q - dot(r.origin, n)) / dot(r.direction, n);
	}
};