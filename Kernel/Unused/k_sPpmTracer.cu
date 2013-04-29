#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"

/*
		float3 dir = -1.0f * woW;
		float3 d = dir - sys.m_normal * 2.0f * dot(sys.m_normal, dir);
		bool into = dot(sys.m_normal, nl) > 0;
		float nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=dot(dir, nl), cos2t;
		f = make_float4(brdf.u_SpecTrans.Transmission, 1);
		pdf = 1;
		if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)
		{
			wiW = (d);
		}
		else
		{
			float3 tdir = normalize(dir * nnt - sys.m_normal * ((into ? 1.0f : -1.0f) * (ddn * nnt + sqrtf(cos2t))));
			float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1.0f - (into ? -ddn : dot(tdir, sys.m_normal));
			float Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
			if(rng.randomFloat() < P)
			{
				f *= RP;
				wiW = (d);
			}
			else
			{
				f *= TP;
				wiW = (tdir);
			}
		}
		return;
*/

CUDA_DEVICE unsigned int g_FreeSlot;
#define EPS 0.01f
texture<float4, 1> t_Buf0;
surface<void, 2> s_Buf1;

#define RNG_COUNT (256 * 256)

__global__ void k_FirstPass(int2 off, int w, int h, CudaRNG* a_Rngs, k_Kernel_sPpmBuffers buf)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	int ind = (y * w + x) % RNG_COUNT;
	CudaRNG localState = a_Rngs[ind];
	x += off.x; y += off.y;
	if(x < w && y < h)
	{
		float x2 = 2.0f * ((float)x / (float)w) - 1.0f,
			  y2 = 2.0f * ((float)y / (float)h) - 1.0f;
		float3 dir, ori;
		g_CameraData.GenRay(x2, y2, &dir, &ori,  localState.randomFloat(), localState.randomFloat());
		TraceResult r;
		struct intRes
		{
			float3 p;
			float3 d;
			float3 c;
			CUDA_FUNC_IN intRes(){}
			CUDA_FUNC_IN intRes(float3 _p, float3 _d, float3 _c)
			{
				p = _p;
				d = _d;
				c = _c;
			}
		};
		const unsigned int stackLength = 32;
		intRes stack[stackLength];
		stack[0] = intRes(ori, dir, make_float3(1));
		unsigned int stackpos = 1, counter = -1;
		while(stackpos)
		{
			if(++counter > 32)
				break;
			stackpos--;
			float3 cl = stack[stackpos].c;
			ori = stack[stackpos].p;
			dir = stack[stackpos].d;
			r.Init();
			if(k_TraceRay<true>(dir, ori, &r))
			{
				float3 p = ori + dir * r.m_fDist;

				float2 uv = r.m_pTri->lerpUV(r.m_fUV);
				Onb sys = r.m_pTri->lerpOnb(r.m_fUV);
				float3 nl = sys.getDirNormal(dir);
				float3 inc;
				float pdf;
				e_KernelMaterial& mat = g_SceneData.m_sMatData[r.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r.m_pNode->m_uMeshIndex].m_uMaterialOffset)];
				float3 ndir = -1.0f * dir;
				BxDFType sType;
				float3 col = mat.m_sBSDF.Sample_f(ndir, &inc, localState.randomFloat(), localState.randomFloat(), &pdf, sys, BSDF_ALL, &sType);
				float3 q = cl * col + mat.Emission;
				if(sType & BSDF_DIFFUSE == BSDF_DIFFUSE)
				{
					unsigned int ind2 = atomicInc(&g_FreeSlot, -1);
					if(ind2 >= buf.BufLength)
						break;//ugly better allocating please
					buf.writeEntry(ind2, k_sPpmEntry(x, y, p + dir * EPS, sys.m_normal, dir, q));
				}
				else
				{
					stack[stackpos++] = intRes(p + inc * EPS, inc, q);
				}
			}
		}
	}
	a_Rngs[ind] = localState;
}

__global__ void k_SecondPass(unsigned int w, unsigned int a_PassIndex, unsigned int a_SPP, uint2* a_Grid, float3 mi, float3 dif, float3 idif, curandState* a_Rngs, float a_CurrRadius)
{
	CudaRNG rng;
	rng.state = a_Rngs[blockIdx.x * blockDim.x + threadIdx.x];
	for(int sample = 0; sample < a_SPP; sample++)
	{
		int li = (int)((float)g_SceneData.m_sLightData.UsedCount * rng.randomFloat());
		Ray r = g_SceneData.m_sLightData[li].SampleRay(rng);
		r.origin += r.direction * 0.01f;
	
		int depth = -1;
		TraceResult r2;
		r2.Init();
		float3 cl = g_SceneData.m_sLightData[li].m_cPower * LIGHT_FACTOR;
		while(++depth < 10 && k_TraceRay<false>(r.direction, r.origin, &r2))
		{
			e_KernelMaterial m = g_SceneData.m_sMatData[r2.m_pTri->getMatIndex(g_SceneData.m_sMeshData[r2.m_pNode->m_uMeshIndex].m_uMaterialOffset)];

			float2 uv = r2.m_pTri->lerpUV(r2.m_fUV);
			Onb sys = r2.m_pTri->lerpOnb(r2.m_fUV);
			float3 inc;
			float pdf;
			float3 col = m.m_sBSDF.Sample_f(-r.direction, &inc, rng.randomFloat(), rng.randomFloat(), &pdf, sys);
		
			float3 x = r(r2.m_fDist), n = sys.m_normal;

			float3 q = (x - mi) * idif;
			if(q.x > 0 && q.x < 1 && q.y > 0 && q.y < 1 && q.z > 0 && q.z < 1)
			{
				float3 q0 = (x - make_float3(a_CurrRadius) - mi) * idif * GRID_SUBS, q1 = (x + make_float3(a_CurrRadius) - mi) * idif * GRID_SUBS;
				uint3 lo = make_uint3(q0), hi = make_uint3(q1), g0 = make_uint3(0), g1 = make_uint3(GRID_SUBS - 1);
				lo = clamp(lo, g0, g1);
				hi = clamp(hi, g0, g1);
				for(int a = lo.x; a <= hi.x; a++)
					for(int b = lo.y; b <= hi.y; b++)
						for(int c = lo.z; c <= hi.z; c++)
						{
							uint2 rng = a_Grid[CALC_INDEX(make_uint3(a,b,c))];
#ifndef USE_BLOCK
							for(int k = 0; k < rng.y; k++)
#else 
							if(!rng.y)
								continue;
							int k = 0;
#endif
							{
								int j = rng.x + k;
								float4 ent = tex1Dfetch(t_Buf0, j);
								float4 ent2;
								unsigned int yp = j / w, xp = (j % w) * 16;
								surf2Dread(&ent2, s_Buf1, xp, yp);
								float3 pos = !ent;
								unsigned int pt = float_as_int(ent.w);
								float3 nor = make_float3(pt & 255, (pt >> 8) & 255, (pt >> 16) & 255) / make_float3(127) - make_float3(1);
								int qt = float_as_int(ent2.x);
								float R2 = __half2float(qt & 65535), N = qt >> 16;
								float sqD = dot(pos - x, pos - x), no = dot(n, nor);
								if(sqD < R2 && no > 1e-3)
								{
									float q = (N * ALPHA + ALPHA) / (N * ALPHA + 1);
									float4 res;
									float r2 = R2 * q;
									int ts = int(N + 1) << 16, ts2 = __float2half_rn(R2 * q);
									res.x = int_as_float(ts | ts2);
									res.y = (ent2.y + cl.x / PI) * q;
									res.z = (ent2.z + cl.y / PI) * q;
									res.w = (ent2.w + cl.z / PI) * q;
									surf2Dwrite(res, s_Buf1, xp, yp, cudaBoundaryModeTrap);
								}
							}
						}
			}

			cl = cl * col + m.Emission;
			if(pdf == 0)
				break;
			if(depth > 3)
			{
				float a = fmaxf(cl), prob = a;
				if(rng.randomFloat() > prob)
					break;
				cl /= prob;
			}
			r.direction = inc;
			r.origin = x + EPS * r.direction;
			r2.Init();
		}
	}
	a_Rngs[blockIdx.x * blockDim.x + threadIdx.x] = rng.state;
}

__global__ void k_ThirdPass(k_Kernel_sPpmBuffers a_Data, float numEmitted, RGBCOL* a_Target, unsigned int m_uLastValidIndex, unsigned int w, unsigned int h)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < m_uLastValidIndex)
	{
		k_sPpmEntry::Memory m = a_Data.b0[i];
#ifdef USE_BLOCK
		if(m.ref != -1)
			i = m.ref;
#endif
		unsigned int yp = i / w, xp = (i % w) * 16;
		float4 ent2;
		surf2Dread(&ent2, s_Buf1, xp, yp);
		int qt = float_as_int(ent2.x);
		float R2 = __half2float(qt & 65535), N = qt >> 16;
		float3 r = make_float3(ent2.y, ent2.z, ent2.w) * (1.0f / (PI * R2 * numEmitted)) * m.Weight;
		RGBCOL c = Float3ToCOLORREF(clamp01(r));
		unsigned int i2 = (h - m.y - 1) * w + m.x;
		atomicAdd((unsigned int*)a_Target + i2, *(unsigned int*)&c);
	}
}

void k_sPpmTracer::DoRender(RGBCOL* a_Buf)
{
	const unsigned long long p = 6 * 32, n = 180, SPP = 3, PhotonsPerPass = p * n * SPP, passIndex = m_uNumPhotonsEmitted / PhotonsPerPass;
	m_sTimer.StartTimer();
	k_SecondPass<<< n, p >>> (w, passIndex, SPP, m_pGrid->m_pDevice, m_vLow, (m_vHigh - m_vLow), make_float3(1) / (m_vHigh - m_vLow), m_pRngData->m_pDevice, sqrtf(m_fCurrRadius));
	cudaThreadSynchronize();
	m_uNumPhotonsEmitted += PhotonsPerPass;
	double tRendering = m_sTimer.EndTimer();
	m_dTimeRendering += tRendering;
	m_dTimeSinceLastUpdate += tRendering;
	float q = ((float)m_uPassesDone * ALPHA + ALPHA) / ((float)m_uPassesDone * ALPHA + 1);
	m_fCurrRadius *= q;
	if(m_dTimeSinceLastUpdate > 0)
	{
		m_dTimeSinceLastUpdate = 0;
		cudaMemset(a_Buf, 0, w * h * sizeof(RGBCOL));
		const unsigned int p = 512;
		k_ThirdPass<<< m_uLastValidIndex / p, p >>> (k_Kernel_sPpmBuffers(m_pEyeHits), (float)m_uNumPhotonsEmitted, a_Buf, m_uLastValidIndex, w, h);
	}
}

void k_sPpmTracer::StartNewTrace()
{
	m_dTimeSinceLastUpdate = m_dTimeRendering = 0;
	m_uNumEyeHits = 0;
	cudaMemcpyToSymbol(g_FreeSlot, &m_uNumEyeHits, 4);
	m_pEyeHits->MemsetDevice(0);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	m_uNumPhotonsEmitted = 0;
	const unsigned int p = 16;
	k_FirstPass<<< dim3( w / p + 1, h / p + 1, 1), dim3(p, p, 1)>>>(make_int2(0,0), w, h, (CudaRNG*)m_pRngData->m_pDevice, k_Kernel_sPpmBuffers(m_pEyeHits));
	cudaError_t r = cudaThreadSynchronize();
	size_t offset;
	cudaChannelFormatDesc cd0 = cudaCreateChannelDesc<float4>(), cd1 = cudaCreateChannelDesc<uint4>();
	r = cudaBindTexture(&offset, &t_Buf0, m_pEyeHits->m_pBuf1->m_pDevice, &cd1, m_uLastValidIndex * sizeof(uint4));
	r = cudaBindSurfaceToArray(s_Buf1, m_pEyeHits->m_pBuf2->m_pDevice, m_pEyeHits->m_pBuf2->getFormatDesc());
	cudaMemcpyFromSymbol(&m_uNumEyeHits, g_FreeSlot, 4);
	m_pEyeHits->CopyToHost();
	float3 mi = make_float3(FLT_MAX), ma = make_float3(-FLT_MAX);
	for(int i = 0; i < m_uNumEyeHits; i++)
	{
		mi = fminf(mi, m_pEyeHits->m_pBuf1->m_pHost[i].Pos);
		ma = fmaxf(ma, m_pEyeHits->m_pBuf1->m_pHost[i].Pos);
	}
	float eps = 1.1f;
	float3 mid = (mi + ma) / 2.0f;
	m_vLow = (mi - mid) * eps + mid;
	m_vHigh = (ma - mid) * eps + mid;
	m_fStartRadius = fsumf(m_vHigh - m_vLow) / ((float)w) * m_fInitialRadiusScale; //m_fStartRadius = 1;
	m_fStartRadius *= 1.5f;
	m_fCurrRadius = m_fStartRadius * m_fStartRadius;
	float3 idiff = make_float3(1) / (m_vHigh - m_vLow);
	half h(m_fCurrRadius);
	for(int i = 0; i < m_uNumEyeHits; i++)
		m_pEyeHits->m_pBuf2->m_pHost[i].R2 = h.bits();
	m_pGrid->MemsetHost(0);
	SortHostData();
	m_pEyeHits->CopyToDevice();
	m_pGrid->CopyToDevice();
}

void k_sPpmTracer::Debug(int2 pixel)
{
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera);
	k_FirstPass<<< 1, 1>>>(pixel, w, h, (CudaRNG*)m_pRngData->m_pDevice, k_Kernel_sPpmBuffers(m_pEyeHits));
}

__global__ void genRNG(curandState* A, unsigned int B, unsigned int C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < C)
		curand_init ( B, i, 0, A + i );
}

void k_sPpmTracer::GenerateRngs()
{
	cudaFuncSetCacheConfig( k_SecondPass, cudaFuncCachePreferL1 );
	m_pRngData = new k_PpmBuf<curandState>(RNG_COUNT);
	genRNG<<< RNG_COUNT / 1024, 1024 >>> (m_pRngData->m_pDevice, time(NULL), RNG_COUNT);
}