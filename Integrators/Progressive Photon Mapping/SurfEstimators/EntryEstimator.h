#pragma once

#include "../../PhotonMapHelper.h"
#include <Engine/SpatialGrid.h>
#include <Engine/Samples.h>

namespace CudaTracerLib {

class EntryEstimator : public SpatialLinkedMap<PPPMPhoton>
{
public:
	EntryEstimator(const Vec3u& size, unsigned long numPhotons)
		: SpatialLinkedMap<PPPMPhoton>(size, numPhotons)
	{

	}

	CUDA_FUNC_IN bool storePhoton(PPPMPhoton& ph, const Vec3f& pos)
	{
		Vec3u cell_idx = getHashGrid().Transform(pos);
		ph.setPos(getHashGrid(), cell_idx, pos);
		return Store(cell_idx, ph) != 0xffffffff;
	}

	CUDA_FUNC_IN Spectrum estimateRadiance(BSDFSamplingRecord& bRec, const NormalizedT<Vec3f>& wi, float r, const Material& mat, unsigned int numPhotonsEmitted, float& pl_est)
	{
		const float LOOKUP_NORMAL_THRESH = 0.5f;

		bool hasGlossy = mat.bsdf.hasComponent(EGlossy);
		Spectrum Lp = Spectrum(0.0f);
		auto surface_region = bRec.dg.ComputeOnSurfaceDiskBounds(r);
		ForAll(surface_region.minV, surface_region.maxV, [&](const Vec3u& cell_idx, unsigned int p_idx, const PPPMPhoton& ph)
		{
			float dist2 = distanceSquared(ph.getPos(getHashGrid(), cell_idx), bRec.dg.P);
			Vec3f photonNormal = ph.getNormal();
			float wiDotGeoN = absdot(photonNormal, wi);
			if (dist2 < r * r && dot(photonNormal, bRec.dg.sys.n) > LOOKUP_NORMAL_THRESH && wiDotGeoN > 1e-2f)
			{
				bRec.wo = bRec.dg.toLocal(ph.getWi());
				float cor_fac = math::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
				float ke = Kernel::k<2>(math::sqrt(dist2), r);
				Spectrum l = ph.getL();
				if (hasGlossy)
					l *= mat.bsdf.f(bRec) / Frame::cosTheta(bRec.wo);//bsdf.f returns f * cos(thetha)
				Lp += ke * l;
				pl_est += ke;
			}
		});

		if (!hasGlossy)
		{
			auto wi_l = bRec.wi;
			bRec.wo = bRec.wi = NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f);
			Lp *= mat.bsdf.f(bRec);
			bRec.wi = wi_l;
		}

		return Lp / (float)numPhotonsEmitted;
	}
};

}