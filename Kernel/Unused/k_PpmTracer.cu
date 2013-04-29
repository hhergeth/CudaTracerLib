#include "k_PpmTracer.h"
#include "k_TraceHelper.h"

#define MAX_PRIMARY_DEPTH 1
#define COUNT MAX_PRIMARY_DEPTH * w * h
#define SIZE(TYPE) COUNT * sizeof(TYPE)

#include <xmmintrin.h>

#define AREA(x) (2.0f * (x.m128_f32[3] * x.m128_f32[2] + x.m128_f32[2] * x.m128_f32[1] + x.m128_f32[3] * x.m128_f32[1]))
#define TOVEC3(x) make_float3(x.m128_f32[3], x.m128_f32[2], x.m128_f32[1])
#define TOSSE3(v) _mm_set_ps(v.x, v.y, v.z, 0)
#define TOBOX(b,t) AABB(TOVEC3(b), TOVEC3(t))
#define V_P_R0(min, max, i) (max.m128_f32[i] - min.m128_f32[i] + 2.0f * maxRadius)
#define V_P_R(min, max) (V_P_R0(min, max, 1) * V_P_R0(min, max, 2) * V_P_R0(min, max, 3))

__declspec(align(128)) struct BBoxTmp
{
    __declspec(align(16)) __m128 _pos;
	unsigned int _index;
};

template<typename T> class nativelist
{
public:
	T* buffer;
	unsigned int p;
	nativelist(T* a)
	{
		buffer = a;
		p = 0;
	}
	T* Add()
	{
		return &buffer[p++];
	}
	void Add(T& a)
	{
		buffer[p++] = a;
	}
	unsigned int index(T* a)
	{
		return ((unsigned int)a - (unsigned int)buffer) / sizeof(T);
	}
	inline T* operator[](int n) { return &buffer[n]; }
};

int addleaf2(BBoxTmp* lwork, int lsize, nativelist<unsigned int>& a_Indices, nativelist<e_PointBVHNode>& a_Nodes)
{
	int start = a_Indices.p;
	for(int j=0; j<lsize; j++)
		a_Indices.Add(lwork[j]._index);
	unsigned int end = -1;
	a_Indices.Add(end);
	return ~start;
}

int RecurseNative2(float maxRadius, int size, BBoxTmp* work, nativelist<unsigned int>& a_Indices, nativelist<e_PointBVHNode>& a_Nodes, __m128& ssebottom, __m128& ssetop, int depth=0)
{
	const int minCount = 4;
    if (size <= minCount)
		return addleaf2(work, size, a_Indices, a_Nodes);
	__m128 a0 = _mm_sub_ps(ssetop, ssebottom);
	float area0 = AREA(a0);
	int bestCountLeft, bestCountRight;
	__m128 bestLeftBottom, bestLeftTop, bestRightBottom, bestRightTop;
	float bestCost = FLT_MAX;
	int bestDim = -1;
	for(int dim = 1; dim < 4; dim++)
	{
		int c = size < 100 ? 1 : 50;
		for(int n = 0; n < size; n+=c)
		{
			int countLeft = n / 2, countRight = size - n / 2;
			if (countLeft<1 || countRight<1)
				continue;
			__m128 lbottom(_mm_set1_ps(FLT_MAX)), ltop(_mm_set1_ps(-FLT_MAX));
			__m128 rbottom(_mm_set1_ps(FLT_MAX)), rtop(_mm_set1_ps(-FLT_MAX));
			for(int i = 0; i < n / 2; i++)
			{
				lbottom = _mm_min_ps(lbottom, work[i]._pos);
				ltop = _mm_max_ps(ltop, work[i]._pos);
			}
			for(int i = n / 2; i < size; i++)
			{
				rbottom = _mm_min_ps(rbottom, work[i]._pos);
				rtop = _mm_max_ps(rtop, work[i]._pos);
			}
			__m128 ltopMinusBottom = _mm_sub_ps(ltop, lbottom);
			__m128 rtopMinusBottom = _mm_sub_ps(rtop, rbottom);
			float vt = V_P_R(ssebottom, ssetop), totalCost = 1.0f + V_P_R(lbottom, ltop) / vt * (float)countLeft + V_P_R(rbottom, rtop) / vt * (float)countRight;
			if (totalCost < bestCost)
			{
				bestDim = dim;
				bestCost = totalCost;
				bestCountLeft = countLeft;
				bestCountRight = countRight;
				bestLeftBottom = lbottom;
				bestLeftTop = ltop;
				bestRightBottom = rbottom;
				bestRightTop = rtop;
			}
		}
	}
	if(bestDim == -1)
		return addleaf2(work, size, a_Indices, a_Nodes);
	BBoxTmp* left = (BBoxTmp*)_mm_malloc(bestCountLeft * sizeof(BBoxTmp), 128), *right = (BBoxTmp*)_mm_malloc(bestCountRight * sizeof(BBoxTmp), 128);
	int l = bestCountLeft, r = bestCountRight;
	for(int i = 0; i < bestCountLeft; i++)
		left[i] = work[i];
	for(int i = 0; i < bestCountRight; i++)
		right[i] = work[bestCountLeft + i];
	e_PointBVHNode* n = a_Nodes.Add();
	int ld = RecurseNative2(maxRadius, l, left, a_Indices, a_Nodes, bestLeftBottom, bestLeftTop, depth + 1),
		rd = RecurseNative2(maxRadius, r, right, a_Indices, a_Nodes, bestRightBottom, bestRightTop, depth + 1); 
	n->setData(ld, rd, TOBOX(bestLeftBottom, bestLeftTop), TOBOX(bestRightBottom, bestRightTop));
	_mm_free(left);
	_mm_free(right);
	return a_Nodes.index(n);
}

int BuildBVH(k_CamHit* a_Data, int a_Count, e_PointBVHNode* a_NodeOut, unsigned int* a_IndexOut, float maxRadius)
{
	__m128 bottom(_mm_set1_ps(FLT_MAX)), top(_mm_set1_ps(-FLT_MAX));
	BBoxTmp* data = (BBoxTmp*)_mm_malloc(a_Count * sizeof(BBoxTmp), 128);
	int c = 0;
	for(unsigned int i = 0; i < a_Count; i++)
	{
		if(!a_Data[i].isValid())
			continue;
		data[c]._pos = TOSSE3(a_Data[i].getPos());
		data[c]._index = i;
		bottom = _mm_min_ps(bottom, data[c]._pos);
		top = _mm_max_ps(top, data[c++]._pos);
	}
	int startNode = RecurseNative2(maxRadius, c, data, nativelist<unsigned int>(a_IndexOut), nativelist<e_PointBVHNode>(a_NodeOut), bottom, top);
	_mm_free(data);
	return startNode;
}

CUDA_FUNC_IN bool SphereAABBIntersection(float3& c, float r, AABB& box)
{
	float3 q = box.minV - c, p = c - box.maxV, e = make_float3(MAX(q.x, 0.0f), MAX(q.y, 0.0f), MAX(q.z, 0.0f)) + make_float3(MAX(p.x, 0.0f), MAX(p.y, 0.0f), MAX(p.z, 0.0f));
	float d = e.x * e.x + e.y * e.y + e.z * e.z;
	return d <= r * r;
}

__global__ void k_FirstPass(int w, int h, k_CamHit* a_Data, e_CameraData g_CameraData, e_KernelDynamicScene g_SceneData, int a_PassIndex)
{
	CudaRNG rnd(12345 * a_PassIndex * w * h + blockDim.x * blockIdx.x + threadIdx.y * MaxBlockHeight + threadIdx.x);
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		float3 dir, ori;
		g_CameraData.GenRay(x2, y2, &dir, &ori,  rnd.randomFloat(), rnd.randomFloat());
		int i = 0;
		TraceResult r;
		while(i < MAX_PRIMARY_DEPTH && k_TraceRay<true>(dir, ori, &r))
		{
			float3 p = ori + dir * r.m_fDist;
			float3 n = r.m_pTri->getNormal(r.m_fUV);
			n = normalize(r.m_pNode->getWorldMatrix().TransformNormal(n));
			float3 nl = dot(n, dir) < 0 ? n : (n * -1.0f);

			a_Data[N * MAX_PRIMARY_DEPTH + i] = k_CamHit(p, nl, dir, x, y, r.m_pTri->MatIndex);
			//if(m->isSpecular())
			//	reflect || i++
			break;
		}
	}
}

__global__ void k_SecondPass(int w, int h, int startNode, k_CamHit* a_HitData, e_PointBVHNode* a_NodeData, unsigned int* a_IndexData, e_KernelDynamicScene g_SceneData, int a_PassIndex, int a_SPP)
{
	CudaRNG rng(12345 * a_PassIndex * w * h + blockDim.x * blockIdx.x + threadIdx.y * MaxBlockHeight + threadIdx.x);
	for(int s = 0; s < a_SPP; s++)
	{
		int li = (int)((float)g_SceneData.m_sLightData.UsedCount * rng.randomFloat());
		Ray r = g_SceneData.m_sLightData[li].SampleRay(rng);
		int depth = -1;
		TraceResult r2;
		float3 cl = g_SceneData.m_sLightData[li].m_cPower;
		while(++depth < 10 && k_TraceRay<false>(r.direction, r.origin, &r2))
		{
			e_KernelMaterial m = g_SceneData.m_sMatData[g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset + r2.m_pTri->MatIndex];

			float3 x = r.origin + r.direction * r2.m_fDist;
			float3 n = r2.m_pTri->getNormal(r2.m_fUV);
			n = normalize(r2.m_pNode->getWorldMatrix().TransformNormal(n));
			float3 nl = dot(n, r.direction) < 0 ? n : (n * -1.0f);

			float3 f = make_float3(0);
			if(m.m_uDiffuseTexIndex < g_SceneData.m_sTexData.UsedCount)
				f = !g_SceneData.m_sTexData[m.m_uDiffuseTexIndex].Sample(r2.m_pTri->lerpTexCoord(r2.m_fUV));
			cl = cl * f + m.Emission;

			if(depth)
			{
				int Stack[32];
				Stack[0] = startNode;
				int pos = 1;
				while(pos)
				{
					int n = Stack[--pos];
					while(n >= 0)
					{
						bool tl = SphereAABBIntersection(x, START_RADIUS, a_NodeData[n].leftBox), tr = SphereAABBIntersection(x, START_RADIUS, a_NodeData[n].rightBox);
						if(tl || tr)
						{
							int li = a_NodeData[n].leftIndex, ri = a_NodeData[n].rightIndex;
							n = (tl) ? li : ri;
							if (tl && tr)
								Stack[pos++] = ri;
						}
						else break;
					}
					if(n < 0)
					{
						for(int index = ~n; ; index++)
						{
							unsigned int pi = a_IndexData[index];
							if(pi == -1)
								break;
							k_CamHit* h = a_HitData + pi;

							float3 qa = h->HitPoint, q0 = qa - x;
							float sqD = dot(q0, q0), no = dot(nl, h->Normal);
							if(sqD < h->Radius * h->Radius && no > 0.1f)
							{
								unsigned int N = atomicExch(&h->N, -1);
								if(N != -1)
								{
									float radj = ((float)N * ALPHA + ALPHA) / ((float)N * ALPHA + 1.0f);
									h->Radius *= radj;
									h->flux = (h->flux + h->weight * cl * (1.0f / PI)) * radj;
									h->N = N + 1;
								}
							}
						}
					}
				}
			}

			r.direction = SampleCosineHemisphere(nl, rng.randomFloat(), rng.randomFloat());
			r.origin = x + 1 * r.direction;
		}
	}
}

__global__ void k_ThirdPass(int w, int h, k_CamHit* a_HitData, float numEmitted, RGBCOL* a_Data)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	if(x < w && y < h)
	{
		float3 r = make_float3(0);
		for(int i = 0; i < MAX_PRIMARY_DEPTH; i++)
		{
			k_CamHit* v = a_HitData + N + i;
			//if(v.N)
			//	r = make_float3(1,0,0);
			//	r *= 1.1f;
			r += v->flux * (1.0f / (PI * v->Radius * numEmitted));
		}
		a_Data[(h - y - 1) * w + x] = Float3ToCOLORREF(clamp01(r));
	}
}

void k_PpmTracer::DoPass(RGBCOL* a_Buf, bool a_NewTrace)
{
	unsigned int p = 16;
	m_pScene->UpdateInvalidated();
	cudaEventRecord(start, 0);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	if(a_NewTrace)
	{
		cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
		cudaMemset(m_pDeviceHitData, 0, SIZE(k_CamHit));
		cudaMemset(m_pDeviceIndexData, -1, SIZE(int) * 2);
		e_CameraData q;
		m_pCamera->getData(q);
		m_uPass = 1;
		m_uNumTraced = 0;
		k_FirstPass<<< dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, m_pDeviceHitData, q, m_pScene->getKernelSceneData(), m_uPass);
		cudaMemcpy(m_pHostHitData, m_pDeviceHitData, SIZE(k_CamHit), cudaMemcpyDeviceToHost);
		startNode = BuildBVH(m_pHostHitData, COUNT, m_pHostBVHData, m_pHostIndexData, START_RADIUS);
		cudaMemcpy(m_pDeviceBVHData, m_pHostBVHData, SIZE(e_PointBVHNode) * 2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_pDeviceIndexData, m_pHostIndexData, SIZE(int) * 2, cudaMemcpyHostToDevice);
	}
	m_uPass = a_NewTrace ? 1 : m_uPass + 1;
	int spp = 1, n0 = 180, n1 = 6;
	k_SecondPass<<< n0, dim3(32, n1, 1)>>>(w, h, startNode, m_pDeviceHitData, m_pDeviceBVHData, m_pDeviceIndexData, m_pScene->getKernelSceneData(), m_uPass, spp);
	m_uNumTraced += spp * n0 * 32 * n1;
	if(m_uPass % 25 == 0)
		k_ThirdPass<<< dim3(w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(w, h, m_pDeviceHitData, m_uNumTraced, a_Buf);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 
	m_uTimePassed = (unsigned int)elapsedTime;
	m_uRaysTraced = m_uNumTraced;
}

void k_PpmTracer::InitializeScene(e_DynamicScene* a_Scene, e_Camera* a_Camera)
{
	m_pScene = a_Scene;
	m_pCamera = a_Camera;
}

void k_PpmTracer::Resize(unsigned int _w, unsigned int _h)
{
	w = _w;
	h = _h;
	if(m_pDeviceHitData != 0)
	{
		cudaFree(m_pDeviceHitData);
		cudaFree(m_pDeviceIndexData);
		cudaFree(m_pDeviceBVHData);
		delete [] m_pHostHitData;
		delete [] m_pHostIndexData;
		delete [] m_pHostBVHData;
	}
	cudaMalloc(&m_pDeviceHitData, SIZE(k_CamHit));
	cudaMalloc(&m_pDeviceIndexData, SIZE(unsigned int) * 2);//at worst every index has its own node
	cudaMalloc(&m_pDeviceBVHData, SIZE(e_PointBVHNode) * 2);
	m_pHostHitData = new k_CamHit[COUNT];
	m_pHostIndexData = new unsigned int[COUNT * 2];
	m_pHostBVHData = new e_PointBVHNode[COUNT * 2];

	if(m_pTmpData != 0)
		cudaFree(m_pTmpData);
	cudaMalloc(&m_pTmpData, sizeof(float4) * w * h);
	cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
}