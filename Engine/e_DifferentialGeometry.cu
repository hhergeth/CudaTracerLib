#include "e_DifferentialGeometry.h"

CUDA_FUNC_IN bool solveLinearSystem2x2(const float a[2][2], const float b[2], float x[2])
{
	float det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

	if (abs(det) <= RCPOVERFLOW)
		return false;

	float inverse = (float) 1.0f / det;

	x[0] = (a[1][1] * b[0] - a[0][1] * b[1]) * inverse;
	x[1] = (a[0][0] * b[1] - a[1][0] * b[0]) * inverse;

	return true;
}

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
		pox = dot(n, rx.origin),
		poy = dot(n, ry.origin),
		prx = dot(n, rx.direction),
		pry = dot(n, ry.direction);

	if (prx == 0 || pry == 0)
	{
		dudx = dvdx = dudy = dvdy = 0.0f;
		return;
	}

	const float tx = (pp - pox) / prx, ty = (pp - poy) / pry;

	/* Calculate the U and V partials by solving two out
	of a set of 3 equations in an overconstrained system */
	float absX = abs(n.x),
		absY = abs(n.y),
		absZ = abs(n.z);

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

	Vec3f px = rx.origin + rx.direction * tx,
		  py = ry.origin + ry.direction * ty;
	float pA[] = { P.x, P.y, P.z };
	float pxA[] = { px.x, px.y, px.z };
	float pyA[] = { py.x, py.y, py.z };

	Bx[0] = pxA[axes[0]] - pA[axes[0]];
	Bx[1] = pxA[axes[1]] - pA[axes[1]];
	By[0] = pyA[axes[0]] - pA[axes[0]];
	By[1] = pyA[axes[1]] - pA[axes[1]];

	if (solveLinearSystem2x2(A, Bx, x))
	{
		dudx = x[0];
		dvdx = x[1];
	}
	else
	{
		dudx = 1;
		dvdx = 0;
	}

	if (solveLinearSystem2x2(A, By, x))
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