#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>

namespace CudaTracerLib {

struct Beam
{
	unsigned short dir;
	RGBE phi;
	float t;
	Vec3f pos;

	CUDA_FUNC_IN Beam() {}
	CUDA_FUNC_IN Beam(const Vec3f& p, const NormalizedT<Vec3f>& d, float t, const Spectrum& ph)
		: pos(p), dir(NormalizedFloat3ToUchar2(d)), t(t), phi(ph.toRGBE())
	{

	}

	CUDA_FUNC_IN Vec3f getPos() const
	{
		return pos;
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> getDir() const
	{
		return Uchar2ToNormalizedFloat3(dir);
	}
	CUDA_FUNC_IN Spectrum getL() const
	{
		Spectrum s;
		s.fromRGBE(phi);
		return s;
	}

	CUDA_FUNC_IN AABB getAABB(float r) const
	{
		const Vec3f beamStart = getPos();
		const Vec3f beamEnd = beamStart + dir * t;
		const Vec3f startMargins(r);
		const Vec3f endMargins(r);
		const Vec3f minPt = min(beamStart - startMargins, beamEnd - endMargins);
		const Vec3f maxPt = max(beamStart + startMargins, beamEnd + endMargins);
		return AABB(minPt, maxPt);
	}

	CUDA_FUNC_IN AABB getSegmentAABB(float splitMin, float splitMax, float r) const
	{
		splitMin *= t;
		splitMax *= t;
		const Vec3f P = getPos();
		const Vec3f beamStart = P + dir * splitMin;
		const Vec3f beamEnd = P + dir * splitMax;
		const Vec3f startMargins(r);
		const Vec3f endMargins(r);
		const Vec3f minPt = min(beamStart - startMargins, beamEnd - endMargins);
		const Vec3f maxPt = max(beamStart + startMargins, beamEnd + endMargins);
		return AABB(minPt, maxPt);
	}

	//this does not(!) account for cone formed beams; this has to be handled afterwards
	CUDA_FUNC_IN static bool testIntersectionBeamBeam(
		const Vec3f& O1,
		const Vec3f& d1,
		const float minT1,
		const float maxT1,
		const Vec3f& O2,
		const Vec3f& d2,
		const float minT2,
		const float maxT2,
		const float maxDistSqr,
		float& oDistance,
		float& oSinTheta,
		float& oT1,
		float& oT2)
	{
		const Vec3f  d1d2c = cross(d1, d2);
		const float sinThetaSqr = dot(d1d2c, d1d2c); // Square of the sine between the two lines (||cross(d1, d2)|| = sinTheta).

		const float ad = dot((O2 - O1), d1d2c);

		// Lines too far apart.
		if (ad*ad >= maxDistSqr*sinThetaSqr)//multiply 1/l * 1/l to the rhs, l = sqrt(sinThetaSqr)
			return false;

		// Cosine between the two lines.
		const float d1d2 = dot(d1, d2);
		const float d1d2Sqr = d1d2*d1d2;
		const float d1d2SqrMinus1 = d1d2Sqr - 1.0f;

		// Parallel lines?
		if (d1d2SqrMinus1 < 1e-5f && d1d2SqrMinus1 > -1e-5f)
			return false;

		const float d1O1 = dot(d1, O1);
		const float d1O2 = dot(d1, O2);

		oT1 = (d1O1 - d1O2 - d1d2 * (dot(d2, O1) - dot(d2, O2))) / d1d2SqrMinus1;

		// Out of range on ray 1.
		if (oT1 <= minT1 || oT1 >= maxT1)
			return false;

		oT2 = (oT1 + d1O1 - d1O2) / d1d2;
		// Out of range on ray 2.
		if (oT2 <= minT2 || oT2 >= maxT2 || math::IsNaN(oT2))
			return false;

		const float sinTheta = math::sqrt(sinThetaSqr);

		oDistance = math::abs(ad) / sinTheta;

		oSinTheta = sinTheta;

		return true; // Found an intersection.
	}
};

typedef Beam _Beam;

struct BeamBeamGrid : public IVolumeEstimator
{
	SpatialLinkedMap<int> m_sStorage;

	SynchronizedBuffer<_Beam> m_sBeamStorage;

	unsigned int m_uBeamIdx;

	CUDA_FUNC_IN static constexpr int DIM()
	{
		return 1;
	}

	BeamBeamGrid(unsigned int gridDim, unsigned int numBeams, int N = 100)
		: IVolumeEstimator(m_sStorage, m_sBeamStorage), m_sStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * N), m_sBeamStorage(numBeams)
	{
		
	}

	virtual void Free()
	{
		m_sStorage.Free();
		m_sBeamStorage.Free();
	}

	CUDA_FUNC_IN _Beam operator()(unsigned int idx)
	{
		return m_sBeamStorage.operator[](idx);
	}

	CTL_EXPORT virtual void StartNewPass(DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box)
	{
		m_sStorage.SetSceneDimensions(box);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_sBeamStorage.getLength();
	}

	CUDA_FUNC_IN unsigned int getNumEntries() const
	{
		return m_sBeamStorage.getLength();
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual void getStatusInfo(size_t& length, size_t& count) const
	{
		length = m_sBeamStorage.getLength();
		count = m_uBeamIdx;
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Beam grid indices", (float)m_sStorage.getNumStoredEntries() / m_sStorage.getNumEntries() * 100));
		a_Buf.push_back(format("%.2f%% Beams", (float)m_uBeamIdx / m_sBeamStorage.getLength() * 100));
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	CTL_EXPORT virtual void PrepareForRendering();

	template<typename BEAM> CUDA_ONLY_FUNC unsigned int StoreBeam(const BEAM& b)
	{
		unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
		if (beam_idx < m_sBeamStorage.getLength())
		{
			m_sBeamStorage[beam_idx] = b;
			bool storedAll = true;
#ifdef ISCUDA
			TraverseGridRay(Ray(b.pos, b.getDir()), m_sStorage.getHashGrid(), 0.0f, b.t, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
			{
				storedAll &= m_sStorage.Store(cell_pos, beam_idx) != 0xffffffff;
			});
#endif
			return storedAll ? beam_idx : 0xffffffff;
		}
		else return 0xffffffff;
	}

	template<typename PHOTON> CUDA_ONLY_FUNC unsigned int StorePhoton(const PHOTON& ph, const Vec3f& pos)
	{
		return 0xffffffff;
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float rad, float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr, float& pl_est)
	{
		Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
		int nPhotons = 0;

		for (unsigned int i = 0; i < min(m_uBeamIdx, m_sBeamStorage.getLength()); i++)
		{
			const Beam& B = m_sBeamStorage[i];
			float beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist;
			if (Beam::testIntersectionBeamBeam(r.ori(), r.dir(), tmin, tmax, B.getPos(), B.getDir(), 0, B.t, math::sqr(rad), beamBeamDistance, sinTheta, queryIsectDist, beamIsectDist))
			{
				nPhotons++;
				Spectrum photon_tau = vol.tau(Ray(B.getPos(), B.getDir()), 0, beamIsectDist);
				Spectrum camera_tau = vol.tau(r, tmin, queryIsectDist);
				Spectrum camera_sc = vol.sigma_s(r(queryIsectDist), r.dir());
				PhaseFunctionSamplingRecord pRec(-r.dir(), B.getDir());
				float p = vol.p(r(queryIsectDist), pRec);
				L_n += B.getL() / NumEmitted * (-photon_tau).exp() * camera_sc * Kernel::k<1>(beamBeamDistance, rad) / sinTheta * (-camera_tau).exp();//this is not correct; the phase function is missing
			}
		}
		Tr = (-vol.tau(r, tmin, tmax)).exp();
		pl_est += nPhotons / (PI * rad * rad * (tmax - tmin));
		return L_n;
	}
};

}