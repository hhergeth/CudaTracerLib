#include "DifferentialGeometry.h"
#include <Math/AlgebraHelper.h>
#include <Math/Ray.h>

//Implementation copied from Mitsuba.

namespace CudaTracerLib {

void DifferentialGeometry::computePartials(const Ray& r, const Ray& rx, const Ray& ry)
{
	float A[2][2], Bx[2], By[2], x[2];
	int axes[2];
	hasUVPartials = true;
	if (dot(dpdu, dpdu) == 0 && dot(dpdv, dpdv) == 0) {
		dudx = dvdx = dudy = dvdy = 0.0f;
		return;
	}

	const float
		pp = dot(n, P),
		pox = dot(n, rx.ori()),
		poy = dot(n, ry.ori()),
		prx = dot(n, rx.dir()),
		pry = dot(n, ry.dir());

	if (prx == 0 || pry == 0)
	{
		dudx = dvdx = dudy = dvdy = 0.0f;
		return;
	}

	const float tx = (pp - pox) / prx, ty = (pp - poy) / pry;

	/* Calculate the U and V partials by solving two out
	of a set of 3 equations in an overconstrained system */
	float absX = math::abs(n.x),
		absY = math::abs(n.y),
		absZ = math::abs(n.z);

	if (absX > absY && absX > absZ) {
		axes[0] = 1; axes[1] = 2;
	}
	else if (absY > absZ) {
		axes[0] = 0; axes[1] = 2;
	}
	else {
		axes[0] = 0; axes[1] = 1;
	}

	float dpduA[] = { dpdu.x, dpdu.y, dpdu.z };
	float dpdvA[] = { dpdv.x, dpdv.y, dpdv.z };

	A[0][0] = dpduA[axes[0]];
	A[0][1] = dpdvA[axes[0]];
	A[1][0] = dpduA[axes[1]];
	A[1][1] = dpdvA[axes[1]];

	Vec3f px = rx.ori() + rx.dir() * tx,
		py = ry.ori() + ry.dir() * ty;
	float pA[] = { P.x, P.y, P.z };
	float pxA[] = { px.x, px.y, px.z };
	float pyA[] = { py.x, py.y, py.z };

	Bx[0] = pxA[axes[0]] - pA[axes[0]];
	Bx[1] = pxA[axes[1]] - pA[axes[1]];
	By[0] = pyA[axes[0]] - pA[axes[0]];
	By[1] = pyA[axes[1]] - pA[axes[1]];

	if (AlgebraHelper::solveLinearSystem2x2(A, Bx, x))
	{
		dudx = x[0];
		dvdx = x[1];
	}
	else
	{
		dudx = 1;
		dvdx = 0;
	}

	if (AlgebraHelper::solveLinearSystem2x2(A, By, x))
	{
		dudy = x[0];
		dvdy = x[1];
	}
	else
	{
		dudy = 0;
		dvdy = 1;
	}
}

void DifferentialGeometry::compute_dp_ds(Vec3f& dp_dx, Vec3f& dp_dy) const
{
	dp_dx = dpdu * dudx + dpdv * dvdx;
	dp_dy = dpdu * dudy + dpdv * dvdy;
}

}