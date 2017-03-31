#include <stdafx.h>
#include "Volumes.h"
#include <Base/Buffer.h>
#include <Base/CudaRandom.h>
#include "Samples.h"
#include <Math/MonteCarlo.h>

namespace CudaTracerLib {

	DenseVolGridBaseType::DenseVolGridBaseType(Stream<char>* a_Buffer, Vec3u dim, size_t sizePerElement, size_t alignment)
	{
		StreamReference<char> streamRef = a_Buffer->malloc_aligned(dim.x * dim.y * dim.z * (unsigned int)sizePerElement, (unsigned int)alignment);
		data = streamRef.AsVar<char>();
	}

	void DenseVolGridBaseType::InvalidateDeviceData(Stream<char>* a_Buffer)
	{
		a_Buffer->translate(data).Invalidate();
	}

	KernelAggregateVolume::KernelAggregateVolume(Stream<VolumeRegion>* D, bool devicePointer)
	{
		m_uVolumeCount = 0;
		for (Stream<VolumeRegion>::iterator it = D->begin(); it != D->end(); ++it)
		{
			m_pVolumes[m_uVolumeCount] = *(*it);
			m_uVolumeCount++;
		}
		box = AABB::Identity();
		for (unsigned int i = 0; i < m_uVolumeCount; i++)
			box = box.Extend(D->operator()(i)->WorldBound());
	}

}