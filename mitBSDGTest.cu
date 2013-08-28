#include "Math\vector.h"

class BSDFALL;

class diffuse
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class roughdiffuse
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b) + 1;
	}
};

class dielectric
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class thindielectric
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class roughdielectric
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class conductor
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class roughconductor
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class plastic
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class roughplastic
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class phong
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};

class ward
{
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		return dot(a,b);
	}
};
/*
class BSDF0
{
private:
	unsigned char Data[255];
	unsigned int type;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		switch(type)
		{
		case 123:
			return ((diffuse*)Data)->pdf(a,b);
		case 2:
			return ((roughdiffuse*)Data)->pdf(a,b);
		case 3:
			return ((dielectric*)Data)->pdf(a,b);
		case 4:
			return ((thindielectric*)Data)->pdf(a,b);
		case 5:
			return ((roughdielectric*)Data)->pdf(a,b);
		case 6:
			return ((conductor*)Data)->pdf(a,b);
		case 7:
			return ((roughconductor*)Data)->pdf(a,b);
		case 8:
			return ((phong*)Data)->pdf(a,b);
		case 9:
			return ((ward*)Data)->pdf(a,b);
		}
		return 0;
	}
};
*/
class coating
{
private:
	BSDFALL* bsdf;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};

class roughcoating
{
private:
	BSDFALL* bsdf;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};
/*
class BSDF1
{
private:
	unsigned char Data[512];
	unsigned int type;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		switch(type)
		{
		case 456:
			return ((BSDF0*)Data)->pdf(a,b);
		case 2:
			return ((coating*)Data)->pdf(a,b);
		case 3:
			return ((roughcoating*)Data)->pdf(a,b);
		}
		return 0;
	}
};
*/
class mixturebsdf
{
private:
	BSDFALL* bsdfs[10];
	float weights[10];
	int num;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};

class blend
{
private:
	BSDFALL* bsdfs[2];
	float weight;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};
/*
class BSDF
{
private:
	unsigned char Data[2048];
	unsigned int type;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		switch(type)
		{
		case 789:
			return ((BSDF1*)Data)->pdf(a,b);
		case 2:
			return ((mixturebsdf*)Data)->pdf(a,b);
		case 3:
			return ((blend*)Data)->pdf(a,b);
		}
		return 0;
	}
};
*/
class BSDFALL
{
private:
	unsigned char Data[255];
	unsigned int type;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b)
	{
		switch(type)
		{
		case 123:
			return ((diffuse*)Data)->pdf(a,b);
		case 2:
			return ((roughdiffuse*)Data)->pdf(a,b);
		case 3:
			return ((dielectric*)Data)->pdf(a,b);
		case 4:
			return ((thindielectric*)Data)->pdf(a,b);
		case 5:
			return ((roughdielectric*)Data)->pdf(a,b);
		case 6:
			return ((conductor*)Data)->pdf(a,b);
		case 7:
			return ((roughconductor*)Data)->pdf(a,b);
		case 8:
			return ((phong*)Data)->pdf(a,b);
		case 9:
			return ((ward*)Data)->pdf(a,b);
		case 10:
			return ((coating*)Data)->pdf(a,b);
		case 11:
			return ((roughcoating*)Data)->pdf(a,b);
		case 12:
			return ((mixturebsdf*)Data)->pdf(a,b);
		case 13:
			return ((blend*)Data)->pdf(a,b);
		}
		return 0;
	}
};

CUDA_FUNC_IN float coating::pdf(float3& a, float3& b)
{
	return bsdf->pdf(a, b) / 2.0f;
}

CUDA_FUNC_IN float roughcoating::pdf(float3& a, float3& b)
{
	return bsdf->pdf(a, b) / 2.0f;
}

CUDA_FUNC_IN float mixturebsdf::pdf(float3& a, float3& b)
{
	float r = 0;
	for(int i = 0; i < num; i++)
		r += bsdfs[i]->pdf(a, b) * weights[i];
	return r;
}

CUDA_FUNC_IN float blend::pdf(float3& a, float3& b)
{
	return lerp(bsdfs[0]->pdf(a, b), bsdfs[1]->pdf(a, b), weight);
}

__global__ void test(BSDFALL* f, float* r, float3 a, float3 b)
{
	*r = f->pdf(a, b);
}