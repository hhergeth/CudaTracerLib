#pragma once

#include <Base/FileStream.h>
#include <vector>

namespace CudaTracerLib {

CTL_EXPORT void compileply(IInStream& in, FileOutputStream& a_Out);
CTL_EXPORT void compileobj(IInStream& in, FileOutputStream& a_Out);
CTL_EXPORT void compilemd5(IInStream& in, std::vector<IInStream*>& animFiles, FileOutputStream& a_Out);

enum MeshCompileType
{
	Static,
	Animated,
};

class MeshCompiler
{
public:
	virtual void Compile(IInStream& in, FileOutputStream& a_Out) = 0;
	virtual bool IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out = 0) = 0;
};

class e_ObjCompiler : public MeshCompiler
{
public:
	CTL_EXPORT virtual void Compile(IInStream& in, FileOutputStream& a_Out);
	CTL_EXPORT virtual bool IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out);
};

class e_Md5Compiler : public MeshCompiler
{
public:
	CTL_EXPORT virtual void Compile(IInStream& in, FileOutputStream& a_Out);
	CTL_EXPORT virtual bool IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out);
};

class e_PlyCompiler : public MeshCompiler
{
public:
	CTL_EXPORT virtual void Compile(IInStream& in, FileOutputStream& a_Out);
	CTL_EXPORT virtual bool IsApplicable(const std::string& a_InputFile, IInStream& in, MeshCompileType* out);
};

class MeshCompilerManager
{
private:
	std::vector<MeshCompiler*> m_sCompilers;
public:
	MeshCompilerManager()
	{
		Register(new e_ObjCompiler());
		Register(new e_Md5Compiler());
		Register(new e_PlyCompiler());
	}
	~MeshCompilerManager()
	{
		for (unsigned int i = 0; i < m_sCompilers.size(); i++)
			delete m_sCompilers[i];
	}
	CTL_EXPORT void Compile(IInStream& in, const std::string& a_Token, FileOutputStream& a_Out, MeshCompileType* out = 0);
	void Register(MeshCompiler* C)
	{
		m_sCompilers.push_back(C);
	}
};

}