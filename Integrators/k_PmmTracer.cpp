#include <StdAfx.h>
#include "k_PmmTracer.h"
#include <sstream>      // std::istringstream
#ifdef ISWINDOWS
#include <Windows.h>
#endif

k_PmmTracer::k_PmmTracer()
	: sMap(100, 10000), dMap(20)
{
}

void toClipboard(std::string str)
{
#ifdef ISWINDOWS
	const char* output = str.c_str();
	const size_t len = strlen(output) + 1;
	HGLOBAL hMem =  GlobalAlloc(GMEM_MOVEABLE, len);
	memcpy(GlobalLock(hMem), output, len);
	GlobalUnlock(hMem);
	OpenClipboard(0);
	EmptyClipboard();
	SetClipboardData(CF_TEXT, hMem);
	CloseClipboard();
#endif
}

void plotPoints(Vec3f* dirs, unsigned int N)
{
	std::ostringstream str1, str2;
	str1 << "x = [";
	str2 << "y = [";
	for(size_t i = 0; i < N; i++)
	{
		dirs[i] = normalize(dirs[i]);
		float2 d = make_float2(dirs[i].x, dirs[i].y);
		str1 << d.x;
		str2 << d.y;
		if(i != N - 1)
		{
			str1 << ", ";
			str2 << ", ";
		}
	}
	str1 << "];";
	str2 << "];";
	str1 << "\n" << str2.str() << "\n" << "scatter(x,y,5)\ngrid on\naxis equal ";
	std::string plt = str1.str();
	toClipboard(plt);
}

void plotModel(const DirectionModel& model)
{
	std::ostringstream str1;	
	str1 << "Z = [";
	int N = 10;
	for(int x = 0; x < N; x++)
	{
		if(x != 0)
			str1 << ";\n";
		for(int y = 0; y < N; y++)
		{
			float a = float(x) / float(N), b = float(y) / float(N);
			float pdf = model.gmm.p(VEC<float, 2>() % a % b);
			str1 << pdf;
			if(y != N - 1)
				str1 << ", ";
		}
	}
	str1 << "];\nX = linspace(0, 1, " << N << ");\nY = linspace(0, 1, " << N << ");\nsurf(X,Y,Z)";
	toClipboard(str1.str());
}