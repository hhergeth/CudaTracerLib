#include "FresnelHelper.h"
#include "Integrator.h"

namespace CudaTracerLib {

namespace {
/// Integrand used by fresnelDiffuseReflectance
CUDA_FUNC_IN float fresnelDiffuseIntegrand(float eta, float xi) {
	return FresnelHelper::fresnelDielectricExt(math::sqrt(xi), eta);
}
}

float FresnelHelper::fresnelDiffuseReflectance(float eta, bool fast) {
	if (fast)
	{
		/* Fast mode: the following code approximates the
		* diffuse Frensel reflectance for the eta<1 and
		* eta>1 cases. An evalution of the accuracy led
		* to the following scheme, which cherry-picks
		* fits from two papers where they are best.
		*/
		if (eta < 1) {
			/* Fit by Egan and Hilgeman (1973). Works
			reasonably well for "normal" IOR values (<2).

			Max rel. error in 1.0 - 1.5 : 0.1%
			Max rel. error in 1.5 - 2   : 0.6%
			Max rel. error in 2.0 - 5   : 9.5%
			*/
			return -1.4399f * (eta * eta)
				+ 0.7099f * eta
				+ 0.6681f
				+ 0.0636f / eta;
		}
		else {
			/* Fit by d'Eon and Irving (2011)
			*
			* Maintains a good accuracy even for
			* unrealistic IOR values.
			*
			* Max rel. error in 1.0 - 2.0   : 0.1%
			* Max rel. error in 2.0 - 10.0  : 0.2%
			*/
			float invEta = 1.0f / eta,
				invEta2 = invEta*invEta,
				invEta3 = invEta2*invEta,
				invEta4 = invEta3*invEta,
				invEta5 = invEta4*invEta;

			return 0.919317f - 3.4793f * invEta
				+ 6.75335f * invEta2
				- 7.80989f * invEta3
				+ 4.98554f * invEta4
				- 1.36881f * invEta5;
		}
	}
	else
	{
		GaussLobattoIntegrator quad(1024, 0, 1e-5f);
		return quad.integrate([&](float xi) {return fresnelDiffuseIntegrand(eta, xi); }, 0, 1);
	}
}

}