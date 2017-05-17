#pragma once

#include <Defines.h>
#include <Base/CudaMemoryManager.h>
#include "TraceHelper.h"

namespace CudaTracerLib {

//a buffer which stores payload elements associated with "primary" rays
//each primary rays has the option to launch associated "secondary" rays.
//The buffer is "double buffered" in the sense that it can be used to read
//rays from it and in the same iteration write new ones to it.
template<typename T> class DoubleRayBuffer
{
	T* m_payload_buffer;
	traversalRay* m_payload_ray_buffer;
	traversalResult* m_payload_res_buffer;
	struct secondary_buf
	{
		traversalRay* m_ray_buffer;
		traversalResult* m_res_buffer;

		secondary_buf(unsigned int N)
		{
			CUDA_MALLOC(&m_ray_buffer, sizeof(traversalRay) * N);
			CUDA_MALLOC(&m_res_buffer, sizeof(traversalResult) * N);
		}

		void Free()
		{
			CUDA_FREE(m_ray_buffer);
			CUDA_FREE(m_res_buffer);
		}
	};
	secondary_buf m_secondary_buf1;
	secondary_buf m_secondary_buf2;

	//number of payload elements and thereby number of primary rays
	unsigned int m_payload_length;
	//number of secondary rays
	unsigned int m_num_secondary_rays;

	//index from where to fetch the next payload element
	unsigned int m_fetch_index;
	//number of stored payload elements
	unsigned int m_num_payload_elements;

	//insert index for payload buffer
	unsigned int m_insert_payload_index;
	//insert index for secondary buffer
	unsigned int m_insert_secondary_index;
public:
	DoubleRayBuffer(unsigned int payload_length, unsigned int secondary_length)
		: m_payload_length(payload_length), m_num_secondary_rays(secondary_length), m_fetch_index(0), m_insert_payload_index(0), m_insert_secondary_index(0), m_secondary_buf1(secondary_length), m_secondary_buf2(secondary_length)
	{
		CUDA_MALLOC(&m_payload_buffer, sizeof(T) * m_payload_length);
		CUDA_MALLOC(&m_payload_ray_buffer, sizeof(traversalRay) * m_payload_length);
		CUDA_MALLOC(&m_payload_res_buffer, sizeof(traversalResult) * m_payload_length);
	}

	void Free()
	{
		CUDA_FREE(m_payload_buffer);
		CUDA_FREE(m_payload_ray_buffer);
		CUDA_FREE(m_payload_res_buffer);
		m_secondary_buf1.Free();
		m_secondary_buf2.Free();
	}

	void StartFrame()
	{
		m_fetch_index = 0;
		m_insert_payload_index = 0;
		m_insert_secondary_index = 0;
		m_num_payload_elements = 0;
	}

	void FinishIteration(bool skip_outer = false, bool any_hit_secondary = false)
	{
		if (m_insert_payload_index > m_payload_length)
			throw std::runtime_error("Storing too many primary rays in buffer!");
		if (m_insert_secondary_index > m_num_secondary_rays)
			throw std::runtime_error("Storing too many secondary rays in buffer!");

		ThrowCudaErrors(cudaMemset(m_payload_res_buffer, 0, sizeof(traversalResult) * m_payload_length));
		ThrowCudaErrors(cudaMemset(m_secondary_buf2.m_res_buffer, 0, sizeof(traversalResult) * m_num_secondary_rays));

		//intersect [0, .., m_insert_payload_index] from payload buffer
		__internal__IntersectBuffers(m_insert_payload_index, m_payload_ray_buffer, m_payload_res_buffer, skip_outer, false);
		//intersect [0, .., m_insert_secondary_index] from secondary buffer
		if(m_insert_secondary_index)
			__internal__IntersectBuffers(m_insert_secondary_index, m_secondary_buf2.m_ray_buffer, m_secondary_buf2.m_res_buffer, skip_outer, any_hit_secondary);

		m_num_payload_elements = m_insert_payload_index;
		m_fetch_index = 0;
		m_insert_payload_index = 0;
		m_insert_secondary_index = 0;
		std::swap(m_secondary_buf1, m_secondary_buf2);
	}

	//checks whether the buffer will be empty (<=> no more payload elements) in the next iteration
	bool isEmpty() const
	{
		return m_insert_payload_index == 0;
	}

	CUDA_ONLY_FUNC bool tryFetchPayloadElement(T& payload_el, traversalRay& ray, traversalResult& res, unsigned int* idx = 0)
	{
		//if (m_fetch_index >= m_num_payload_elements)
		//	return false;
		unsigned payload_idx = atomicInc(&m_fetch_index, UINT_MAX);
		if (payload_idx >= m_num_payload_elements)
			return false;

		if (idx)
			*idx = payload_idx;
		payload_el = m_payload_buffer[payload_idx];
		ray = m_payload_ray_buffer[payload_idx];
		res = m_payload_res_buffer[payload_idx];
		return true;
	}

	CUDA_ONLY_FUNC bool insertPayloadElement(const T& payload_el, const traversalRay& ray, unsigned int* idx = 0)
	{
		// || m_insert_payload_index >= m_fetch_index
		//if (m_insert_payload_index >= m_payload_length)
		//	return false;
		unsigned int payload_idx = atomicInc(&m_insert_payload_index, UINT_MAX);
		if (payload_idx >= m_payload_length)
			return false;

		if (idx)
			*idx = payload_idx;
		m_payload_buffer[payload_idx] = payload_el;
		m_payload_ray_buffer[payload_idx] = ray;
		return true;
	}

	CUDA_ONLY_FUNC bool accessSecondaryRay(unsigned int idx, traversalRay& ray, traversalResult& res)
	{
		if (idx > m_num_secondary_rays)
			return false;

		ray = m_secondary_buf1.m_ray_buffer[idx];
		res = m_secondary_buf1.m_res_buffer[idx];
		return true;
	}

	CUDA_ONLY_FUNC bool insertSecondaryRay(const traversalRay& ray, unsigned int& idx)
	{
		if (m_insert_secondary_index >= m_num_secondary_rays)
			return false;
		idx = atomicInc(&m_insert_secondary_index, UINT_MAX);
		if (idx >= m_num_secondary_rays)
			return false;

		m_secondary_buf2.m_ray_buffer[idx] = ray;
		return true;
	}

	//helper functions to convert trivial structs to usable types
	CUDA_ONLY_FUNC bool tryFetchPayloadElement(T& payload_el, NormalizedT<Ray>& ray, TraceResult& res, unsigned int* idx = 0)
	{
		traversalRay r1;
		traversalResult r2;
		if (tryFetchPayloadElement(payload_el, r1, r2, idx))
		{
			convert(r1, r2, ray, res);
			return true;
		}
		else return false;
	}

	CUDA_ONLY_FUNC bool insertPayloadElement(const T& payload_el, const NormalizedT<Ray>& ray, unsigned int* idx = 0)
	{
		traversalRay r1;
		convert(ray, r1);
		return insertPayloadElement(payload_el, r1, idx);
	}

	CUDA_ONLY_FUNC bool accessSecondaryRay(unsigned int idx, NormalizedT<Ray>& ray, TraceResult& res)
	{
		traversalRay r1;
		traversalResult r2;
		if (accessSecondaryRay(idx, r1, r2))
		{
			convert(r1, r2, ray, res);
			return true;
		}
		else return false;
	}

	CUDA_ONLY_FUNC bool insertSecondaryRay(const NormalizedT<Ray>& ray, unsigned int& idx)
	{
		traversalRay r1;
		convert(ray, r1);
		return insertSecondaryRay(r1, idx);
	}
private:
	CUDA_FUNC_IN void convert(const traversalRay& r1, const traversalResult& r2, NormalizedT<Ray>& ray, TraceResult& res)
	{
		ray = NormalizedT<Ray>(Vec3f(r1.a.getXYZ()), NormalizedT<Vec3f>(r1.b.getXYZ()));
		r2.toResult(&res, g_SceneData);
	}
	CUDA_FUNC_IN void convert(const NormalizedT<Ray>& ray, traversalRay& r1)
	{
		r1.a = Vec4f(ray.ori(), 1e-2f);
		r1.b = Vec4f(ray.dir(), FLT_MAX);
	}
};

}