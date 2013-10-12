#include "StdAfx.h"
#include "e_MeshCompiler.h"
#include "..\Base\FrameworkInterop.h"

bool e_ObjCompiler::IsApplicable(const char* a_InputFile, e_MeshCompileType* out)
{
	bool b = FW::String(a_InputFile).toLower().endsWith(".obj");
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_ObjCompiler::Compile(const char* a_InputFile, OutputStream& a_Out)
{
	e_Mesh::CompileObjToBinary(a_InputFile, a_Out);
}

bool e_Md5Compiler::IsApplicable(const char* a_InputFile, e_MeshCompileType* out)
{
	bool b = FW::String(a_InputFile).toLower().endsWith(".md5mesh");
	if(out && b)
		*out = e_MeshCompileType::Animated;
	return b;
}

void e_Md5Compiler::Compile(const char* a_InputFile, OutputStream& a_Out)
{
	c_StringArray A;
	char dir[255];
	ZeroMemory(dir, sizeof(dir));
	_splitpath(FW::String(a_InputFile).getPtr(), 0, dir, 0, 0);
	WIN32_FIND_DATA dat;
	HANDLE hFind = FindFirstFile(FW::String(dir).append("\\*.md5anim").getPtr(), &dat);
	while(hFind != INVALID_HANDLE_VALUE)
	{
		FW::String* q = new FW::String(FW::String(dir).append(dat.cFileName));
		A((char*)q->getPtr());
		if(!FindNextFile(hFind, &dat))
			break;
	}
	e_AnimatedMesh::CompileToBinary(a_InputFile, A, a_Out);
}

bool e_PlyCompiler::IsApplicable(const char* a_InputFile, e_MeshCompileType* out)
{
	bool b = FW::String(a_InputFile).toLower().endsWith(".ply");
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_PlyCompiler::Compile(const char* a_InputFile, OutputStream& a_Out)
{
	compileply(a_InputFile, a_Out);
}

void e_MeshCompilerManager::Compile(const char* a_InputFile, OutputStream& a_Out, e_MeshCompileType* out)
{
	e_MeshCompileType t;
	for(unsigned int i = 0; i < m_sCompilers.size(); i++)
		if(m_sCompilers[i]->IsApplicable(a_InputFile, &t))
		{
			if(out)
				*out = t;
			a_Out << (unsigned int)t;
			m_sCompilers[i]->Compile(a_InputFile, a_Out);
			break;
		}
}