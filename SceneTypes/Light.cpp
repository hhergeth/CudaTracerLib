#include <StdAfx.h>
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
		StreamReference<char> cdfColsRef = a_Buffer->malloc_aligned<float>(nEntries * sizeof(float)),
            cdfRowsRef = a_Buffer->malloc_aligned<float>((radianceMap.m_uHeight + 1) * sizeof(float)),
            rowWeightsRef = a_Buffer->malloc_aligned<float>(radianceMap.m_uHeight * sizeof(float));

        float* cdfCols = (float*)cdfColsRef.operator char *();
        float* cdfRows = (float*)cdfRowsRef.operator char *();
        float* rowWeights = (float*)rowWeightsRef.operator char *();
	
		unsigned int colPos = 0, rowPos = 0;
		float rowSum = 0.0f;
        cdfRows[rowPos++] = 0;
		for (unsigned int y = 0; y < radianceMap.m_uHeight; ++y)
		{
			float colSum = 0;

            cdfCols[colPos++] = 0;
			for (unsigned int x = 0; x < radianceMap.m_uWidth; ++x)
			{
				Spectrum value = radianceMap.Sample(0, (int)x, (int)y);

				colSum += value.getLuminance();
                cdfCols[colPos++] = (float)colSum;
			}

			float normalization = 1.0f / (float)colSum;
			for (unsigned int x = 1; x < radianceMap.m_uWidth; ++x)
                cdfCols[colPos - x - 1] *= normalization;
            cdfCols[colPos - 1] = 1.0f;

			float weight = sinf((y + 0.5f) * PI / m_size.y);
            rowWeights[y] = weight;
			rowSum += colSum * weight;
            cdfRows[rowPos++] = (float)rowSum;
		}
		float normalization = 1.0f / (float)rowSum;
		for (unsigned int y = 1; y < radianceMap.m_uHeight; ++y)
            cdfRows[rowPos - y - 1] *= normalization;
        cdfRows[rowPos - 1] = 1.0f;
		m_normalization = 1.0f / (rowSum * (2 * PI / m_size.x) * (PI / m_size.y));
		m_pixelSize = Vec2f(2 * PI / m_size.x, PI / m_size.y);

        cdfColsRef.Invalidate(); cdfRowsRef.Invalidate(); rowWeightsRef.Invalidate();
        m_cdfColsIdx = cdfColsRef.getIndex(); m_cdfRowsIdx = cdfRowsRef.getIndex(); m_rowWeightsIdx = rowWeightsRef.getIndex();
        m_cdfColsLength = cdfColsRef.getLength(); m_cdfRowsLength = cdfRowsRef.getLength(); m_rowWeightsLength = rowWeightsRef.getLength();

		m_worldTransform = NormalizedT<OrthogonalAffineMap>::Identity();
	}

}