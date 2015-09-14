#include "StdAfx.h"
#include "e_MeshCompiler.h"
#include "e_AnimatedMesh.h"
#include <algorithm>
#include <string>
#define BOOST_FILESYSTEM_DEPRECATED
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
using namespace boost::filesystem;

bool hasEnding (std::string const &fullString, std::string const &_ending)
{
	std::string ending = _ending;
	boost::algorithm::to_lower(ending);
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

bool e_ObjCompiler::IsApplicable(const std::string& a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".obj");
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_ObjCompiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	compileobj(in, a_Out);
}

bool e_Md5Compiler::IsApplicable(const std::string& a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".md5mesh");
	if(out && b)
		*out = e_MeshCompileType::Animated;
	return b;
}

void e_Md5Compiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	std::vector<IInStream*> animFiles;
	boost::filesystem::path p_file(in.getFilePath());
	for (directory_iterator it(p_file.parent_path()); it != directory_iterator(); ++it)
	{
		if (!is_directory(*it))
		{
			std::string ext = it->path().extension().string();
			boost::algorithm::to_lower(ext);
			if (ext == ".md5anim")
			{
				animFiles.push_back(new FileInputStream(it->path().string()));
			}
		}
	}
	compilemd5(in, animFiles, a_Out);
	for (size_t i = 0; i < animFiles.size(); i++)
		delete animFiles[i];
}

bool e_PlyCompiler::IsApplicable(const std::string& a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".ply");
	if(b)
	{
		char magic[4];
		magic[3] = 0;
		in.Read(magic, 3);
		b = std::string(magic) == "ply";
		in.Move(-3);
	}
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_PlyCompiler::Compile(IInStream& in, FileOutputStream& a_Out)
{
	compileply(in, a_Out);
}

void e_MeshCompilerManager::Compile(IInStream& in, const std::string& a_InputFile, FileOutputStream& a_Out, e_MeshCompileType* out)
{
	e_MeshCompileType t;
	for(unsigned int i = 0; i < m_sCompilers.size(); i++)
	{
		bool b = m_sCompilers[i]->IsApplicable(a_InputFile, in, &t);
		if(b)
		{
			if(out)
				*out = t;
			a_Out << (unsigned int)t;
			m_sCompilers[i]->Compile(in, a_Out);
			break;
		}
	}
}