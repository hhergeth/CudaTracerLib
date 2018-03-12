#include "StdAfx.h"
#include "MeshCompiler.h"
#include <Engine/AnimatedMesh.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <filesystem.h>
#include <Base/Platform.h>

namespace CudaTracerLib {

bool hasEnding(std::string const &fullString, std::string const &_ending)
{
	std::string ending = _ending;
	to_lower(ending);
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}

bool e_ObjCompiler::IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".obj");
	if (out && b)
		*out = MeshCompileType::Static;
	return b;
}

void e_ObjCompiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	compileobj(in, a_Out);
}

bool e_Md5Compiler::IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".md5mesh");
	if (out && b)
		*out = MeshCompileType::Animated;
	return b;
}

void e_Md5Compiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	std::vector<IInStream*> animFiles;
	std::filesystem::path p_file(in.getFilePath());
	for (std::filesystem::directory_iterator it(p_file.parent_path()); it != std::filesystem::directory_iterator(); ++it)
	{
		std::string ext = it->path().extension().string();
		to_lower(ext);
		if (ext == ".md5anim")
		{
			animFiles.push_back(new FileInputStream(it->path().string()));
		}
	}
	compilemd5(in, animFiles, a_Out);
	for (size_t i = 0; i < animFiles.size(); i++)
		delete animFiles[i];
}

bool e_PlyCompiler::IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".ply");
	if (b)
	{
		char magic[4];
		magic[3] = 0;
		in.Read(magic, 3);
		b = std::string(magic) == "ply";
		in.Move(-3);
	}
	if (out && b)
		*out = MeshCompileType::Static;
	return b;
}

void e_PlyCompiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	compileply(in, a_Out);
}

void MeshCompilerManager::Compile(IInStream& in, const std::string& a_InputFile, FileOutputStream& a_Out, MeshCompileType* out)
{
	MeshCompileType t;
	for (unsigned int i = 0; i < m_sCompilers.size(); i++)
	{
		bool b = m_sCompilers[i]->IsApplicable(a_InputFile, in, &t);
		if (b)
		{
			if (out)
				*out = t;
			a_Out << (unsigned int)t;
			m_sCompilers[i]->Compile(in, a_Out);
			break;
		}
	}
}

}
