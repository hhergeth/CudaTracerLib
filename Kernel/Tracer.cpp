#include <StdAfx.h>
#include "Tracer.h"
#include "TraceHelper.h"
#include "Sampler.h"
#include "BlockSampler/UniformBlockSampler.h"
#include "BlockSampler/VarianceBlockSampler.h"
#include "BlockSampler/DifferenceBlockSampler.h"
#include "Sampler.h"

namespace CudaTracerLib {

TracerBase::TracerBase()
	: m_pScene(0), m_pBlockSampler(0), m_pSamplingSequenceGenerator(0), m_pPixelVarianceBuffer(0), w(0xffffffff), h(0xffffffff)
{
	ThrowCudaErrors(cudaEventCreate(&start));
	ThrowCudaErrors(cudaEventCreate(&stop));
	m_sParameters << KEY_SamplingSequenceType()			<< SamplingSequenceGeneratorTypes::Independent
				  << KEY_BlockSamplerType()				<< BlockSamplerTypes::Uniform;
	setCorrectSamplingSequenceGenerator();
	setCorrectBlockSampler();
}

TracerBase::~TracerBase()
{
	if (start == 0)
	{
		std::cout << "Calling ~TracerBase() multiple times!\n";
		return;
	}
	ThrowCudaErrors(cudaEventDestroy(start));
	ThrowCudaErrors(cudaEventDestroy(stop));
	start = stop = 0;
	if (m_pSamplingSequenceGenerator)
	{
		delete m_pSamplingSequenceGenerator;
		m_pSamplingSequenceGenerator = 0;
	}
	if (m_pBlockSampler)
		delete m_pBlockSampler;
	if (m_pPixelVarianceBuffer)
	{
		m_pPixelVarianceBuffer->Free();
		delete m_pPixelVarianceBuffer;
	}
	m_pBlockSampler = 0;
	m_pPixelVarianceBuffer = 0;
	m_debugVisualizerManager.Free();
}

void TracerBase::generateNewRandomSequences()
{
	GenerateNewRandomSequences(*m_pSamplingSequenceGenerator);
}

template<typename T> struct check_type_ssg
{
	void operator()(ISamplingSequenceGenerator*& gen, SamplingSequenceGeneratorTypes new_type, SamplingSequenceGeneratorTypes T_type)
	{
		if (dynamic_cast<SamplingSequenceGeneratorHost<T>*>(gen) == 0 && new_type == T_type)
		{
			delete gen;
			gen = new SamplingSequenceGeneratorHost<T>();
		}
	}
};
void TracerBase::setCorrectSamplingSequenceGenerator()
{
	auto new_type = m_sParameters.getValue(KEY_SamplingSequenceType());

	check_type_ssg<IndependantSamplingSequenceGenerator>()(m_pSamplingSequenceGenerator, new_type, Independent);
	check_type_ssg<StratifiedSamplingSequenceGenerator>()(m_pSamplingSequenceGenerator, new_type, Stratified);
}

template<typename T> struct check_type_bst
{
	void operator()(TracerParameterCollection& col, IBlockSampler*& bSampler, int w, int h, BlockSamplerTypes new_type, BlockSamplerTypes T_type)
	{
		if (dynamic_cast<T*>(bSampler) == 0 && new_type == T_type)
		{
			delete bSampler;
			bSampler = new T(w, h);

			//add parameters to settings
			const std::string name = "Block Sampler";
			if(col.getCollection(name))
				col.removeChildCollection(name);
			col.addChildParameterCollection(name, &bSampler->getParameterCollection());
		}
	}
};
void TracerBase::setCorrectBlockSampler()
{
	if (w == 0xffffffff || h == 0xffffffff)
		return;

	auto new_type = m_sParameters.getValue(KEY_BlockSamplerType());

	check_type_bst<UniformBlockSampler>()(m_sParameters, m_pBlockSampler, w, h, new_type, BlockSamplerTypes::Uniform);
	check_type_bst<VarianceBlockSampler>()(m_sParameters, m_pBlockSampler, w, h, new_type, BlockSamplerTypes::Variance);
	check_type_bst<DifferenceBlockSampler>()(m_sParameters, m_pBlockSampler, w, h, new_type, BlockSamplerTypes::Difference);
}

}