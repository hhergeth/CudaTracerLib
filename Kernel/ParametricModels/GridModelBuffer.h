#pragma once

#include <Engine/SpatialStructures/Grid/SpatialGridList.h>
#include <Engine/SpatialStructures/Grid/SpatialGridSet.h>
#include <Kernel/PixelDebugVisualizers/PixelDebugVisualizer.h>
#include <Base/FixedSizeArray.h>
#include <Base/CudaRandom.h>
#include <qMatrixHelper.h>

namespace CudaTracerLib {

//Implementation of "On-line Learning of Parametric Mixture Models for Light Transport Simulation"

namespace __ParametricMixtureModelBuffer__
{
	template<int D> using vec = qMatrix<float, D, 1>;
	template<int D> struct entry
	{
		vec<D> val;
		float weight;
		entry()
		{
		}
		CUDA_FUNC_IN entry(const vec<D>& v, float w)
			: val(v), weight(w)
		{
		}
	};

	template<typename Model, int D> CUDA_FUNC_IN void updateCell(const Vec3u& idx, SpatialGridList_Linked<entry<D>>& entryBuffer, SpatialGridSet<Model>& mixtureBuffer, const vec<D>& range_min, const vec<D>& range_max)
	{
		auto& model = mixtureBuffer(idx);
		auto helper = model.getTrainingHelper();

		//collect all samples for this mixture model
		entryBuffer.ForAllCellEntries(idx, [&](unsigned int entry_idx, const entry<D>& entry)
		{
			helper.addSample(entry.val, entry.weight, range_min, range_max);
		});
		helper.finish(range_min, range_max);
	}

	template<int D, typename Model> CUDA_GLOBAL void updateMixtureModels(SpatialGridList_Linked<entry<D>> entryBuffer, SpatialGridSet<Model> mixtureBuffer, const vec<D> range_min, const vec<D> range_max)
	{
		Vec3u idx = Vec3u(blockIdx.x * blockDim.x + threadIdx.x,
						  blockIdx.y * blockDim.y + threadIdx.y,
						  blockIdx.z * blockDim.z + threadIdx.z);
		auto& grid = entryBuffer.getHashGrid();
		if (idx.x < grid.m_gridDim.x && idx.y < grid.m_gridDim.y && idx.z < grid.m_gridDim.z)
			updateCell(idx, entryBuffer, mixtureBuffer, range_min, range_max);
	}

	template<int N, int DIM> struct matrix_type
	{

	};

	template<int N> struct matrix_type<N, 1>
	{
		typedef qMatrix<float, N, 1> type;
	};

	template<int N> struct matrix_type<N, 2>
	{
		typedef qMatrix<float, N, N> type;
	};

	template<int N_SAMPLES_PER_DIM, typename Model> static qMatrix<float, N_SAMPLES_PER_DIM, 1> VisualizeModel(const Model& model, const vec<1>& range_min, const vec<1>& range_max)
	{
		qMatrix<float, N_SAMPLES_PER_DIM, 1> buf;

		for (int i = 0; i < N_SAMPLES_PER_DIM; i++)
		{
			float x = i / float(N_SAMPLES_PER_DIM);
			auto pdf = model.pdf(VEC<float, 1>() % x, range_min, range_max);
			buf(i) = pdf;
		}

		return buf;
	}

	template<int N_SAMPLES_PER_DIM, typename Model> static qMatrix<float, N_SAMPLES_PER_DIM, N_SAMPLES_PER_DIM> VisualizeModel(const Model& model, const vec<2>& range_min, const vec<2>& range_max)
	{
		qMatrix<float, N_SAMPLES_PER_DIM, N_SAMPLES_PER_DIM> buf;

		for (int i = 0; i < N_SAMPLES_PER_DIM; i++)
		{
			for (int j = 0; j < N_SAMPLES_PER_DIM; j++)
			{
				float x = i / float(N_SAMPLES_PER_DIM);
				float y = j / float(N_SAMPLES_PER_DIM);
				auto pdf = model.pdf(VEC<float, 2>() % x % y, range_min, range_max);
				buf(i, j) = pdf;
			}
		}

		return buf;
	}
}

template<int D, typename Model> class GridModelBuffer : public ISynchronizedBufferParent
{
	using vec = __ParametricMixtureModelBuffer__::vec<D>;
	using entry = __ParametricMixtureModelBuffer__::entry<D>;

	SpatialGridList_Linked<entry> m_valueBuffer;
	SpatialGridSet<Model> m_mixtureBuffer;
	int N_TOTAL;
	vec range_min, range_max;
	CudaRNG init_RNG;
public:
	GridModelBuffer(const Vec3u& gridSize, unsigned int numEntriesToStore, const vec& range_min, const vec& range_max)
		: m_valueBuffer(gridSize, numEntriesToStore), m_mixtureBuffer(gridSize), ISynchronizedBufferParent(m_valueBuffer, m_mixtureBuffer), range_min(range_min), range_max(range_max), init_RNG(7251629), N_TOTAL(0)
	{
		ResetBuffer();
	}

	void Free()
	{
		m_valueBuffer.Free();
		m_mixtureBuffer.Free();
	}

	void SetGridDimensions(const AABB& box)
	{
		m_valueBuffer.SetGridDimensions(box);
		m_mixtureBuffer.SetGridDimensions(box);
	}

	void ResetBuffer()
	{
		m_valueBuffer.ResetBuffer();

		N_TOTAL = 0;
		for (unsigned int i = 0; i < m_mixtureBuffer.getNumCells(); i++)
		{
			m_mixtureBuffer(i) = Model(init_RNG, range_min, range_max);
		}
		m_mixtureBuffer.setOnCPU();
		m_mixtureBuffer.Synchronize();
	}

	template<int N_SAMPLES_PER_DIM> typename __ParametricMixtureModelBuffer__::matrix_type<N_SAMPLES_PER_DIM, D>::type VisualizeModel(const Model& model) const
	{
		return __ParametricMixtureModelBuffer__::VisualizeModel<N_SAMPLES_PER_DIM>(model, range_min, range_max);
	}

	CUDA_FUNC_IN unsigned int StoreEntry(const Vec3f& pos, const vec& val, float weight)
	{
		return m_valueBuffer.Store(pos, entry(val, weight));
	}

	CUDA_FUNC_IN const Model& getMixtureModel(const Vec3f& pos) const
	{
		return m_mixtureBuffer(pos);
	}

	void UpdateMixtureModels(bool useGPUToUpdate = true)
	{
		N_TOTAL += std::min(m_valueBuffer.getNumStoredEntries(), m_valueBuffer.getNumEntries());

#ifdef __CUDACC__
		if (useGPUToUpdate)
		{
			int l = 6;
			auto L = m_valueBuffer.getHashGrid().m_gridDim / l + 1;

			__ParametricMixtureModelBuffer__::updateMixtureModels<D, Model> <<<dim3(L.x, L.y, L.z), dim3(l,l,l)>>>(m_valueBuffer, m_mixtureBuffer, range_min, range_max);

			m_mixtureBuffer.setOnGPU();
		}
		else
#endif
		{
			//do host-device synchronization and update models
			m_mixtureBuffer.Synchronize();
			m_valueBuffer.Synchronize();
			auto L = m_valueBuffer.getHashGrid().m_gridDim;
			for (unsigned int i = 0; i < L.x; i++)
				for (unsigned int j = 0; j < L.y; j++)
					for (unsigned int k = 0; k < L.z; k++)
						__ParametricMixtureModelBuffer__::updateCell(Vec3u(i,j,k), m_valueBuffer, m_mixtureBuffer, range_min, range_max);
			m_mixtureBuffer.setOnCPU();
			m_mixtureBuffer.Synchronize();
		}

		//clear the entry buffer for the next iteration
		m_valueBuffer.ResetBuffer();
	}
};

}
