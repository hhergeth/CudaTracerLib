#include <stdafx.h>
#include "Light.h"
#include <Engine/MIPMap.h>
#include <Math/MonteCarlo.h>
#include <Math/Warp.h>
#include <Base/Buffer.h>

namespace CudaTracerLib {

	InfiniteLight::InfiniteLight(Stream<char>* a_Buffer, BufferReference<MIPMap, KernelMIPMap>& mip, const Spectrum& scale, const AABB* scenBox)
		: LightBase(false), radianceMap(mip->getKernelData()), m_scale(scale), m_pSceneBox(scenBox)
	{
		m_size = Vec2f((float)radianceMap.m_uWidth, (float)radianceMap.m_uHeight);
		unsigned int nEntries = (radianceMap.m_uWidth + 1) * radianceMap.m_uHeight;
		StreamReference<char> m1 = a_Buffer->malloc_aligned<float>(nEntries * sizeof(float)),
			m2 = a_Buffer->malloc_aligned<float>((radianceMap.m_uHeight + 1) * sizeof(float)),
			m3 = a_Buffer->malloc_aligned<float>(radianceMap.m_uHeight * sizeof(float));
		m_cdfCols = m1.AsVar<float>();
		m_cdfRows = m2.AsVar<float>();
		m_rowWeights = m3.AsVar<float>();
		unsigned int colPos = 0, rowPos = 0;
		float rowSum = 0.0f;
		m_cdfRows[rowPos++] = 0;
		for (unsigned int y = 0; y < radianceMap.m_uHeight; ++y)
		{
			float colSum = 0;

			m_cdfCols[colPos++] = 0;
			for (unsigned int x = 0; x < radianceMap.m_uWidth; ++x)
			{
				Spectrum value = radianceMap.Sample(0, (int)x, (int)y);

				colSum += value.getLuminance();
				m_cdfCols[colPos++] = (float)colSum;
			}

			float normalization = 1.0f / (float)colSum;
			for (unsigned int x = 1; x < radianceMap.m_uWidth; ++x)
				m_cdfCols[colPos - x - 1] *= normalization;
			m_cdfCols[colPos - 1] = 1.0f;

			float weight = sinf((y + 0.5f) * PI / m_size.y);
			m_rowWeights[y] = weight;
			rowSum += colSum * weight;
			m_cdfRows[rowPos++] = (float)rowSum;
		}
		float normalization = 1.0f / (float)rowSum;
		for (unsigned int y = 1; y < radianceMap.m_uHeight; ++y)
			m_cdfRows[rowPos - y - 1] *= normalization;
		m_cdfRows[rowPos - 1] = 1.0f;
		m_normalization = 1.0f / (rowSum * (2 * PI / m_size.x) * (PI / m_size.y));
		m_pixelSize = Vec2f(2 * PI / m_size.x, PI / m_size.y);
		m1.Invalidate(); m2.Invalidate(); m3.Invalidate();

		m_worldTransform = NormalizedT<OrthogonalAffineMap>::Identity();
	}

}