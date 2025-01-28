//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#include <numeric>

#include "DlQuantization/QuantizerFactory.hpp"
#include "EncodingAnalyzerWrapper.h"
#include "math_functions.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"


namespace DlQuantization
{

template <typename DTYPE>
EncodingAnalyzerWrapper<DTYPE>::EncodingAnalyzerWrapper(TensorDims shape, QuantizationMode mode) : _shape(shape)
{
    size_t numBlocks = getNumel(shape);
    _encodingAnalyzers.resize(numBlocks);
    for (auto& ptr: _encodingAnalyzers)
    {
        ptr = getEncodingAnalyzerInstance<DTYPE>(mode);
    }
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::updateStats(const DTYPE* tensor, const TensorDims& tensorShape,
                                                 ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    auto numBlocks = _encodingAnalyzers.size();
    auto numel     = getNumel(tensorShape);

    // Early exit for per-tensor mode
    if (numBlocks == 1)
    {
        return _updateStatsContiguous(tensor, tensorCpuGpuMode, numel, allocator, stream);
    }

    // View tensor and encoding as broadcastable shapes
    auto bcShapes      = getBroadcastableShapes(tensorShape, _shape);
    auto bcTensorShape = std::get<0>(bcShapes);
    auto bcEncShape    = std::get<1>(bcShapes);

    size_t blockSize = numel / numBlocks;

    std::vector<size_t> broadcastDims, nonBroadcastDims;

    // Determine the dim ordering such that all indexes in a single quantization block are contiguous
    for (size_t i = 0; i < bcTensorShape.size(); i++)
    {
        if (bcEncShape[i] == 1 && bcTensorShape[i] != 1)
        {
            broadcastDims.push_back(i);
        }
        else
        {
            nonBroadcastDims.push_back(i);
        }
    }
    std::vector<size_t> dimOrder = nonBroadcastDims;
    dimOrder.insert(dimOrder.end(), broadcastDims.begin(), broadcastDims.end());

    // Permute the input to have contiguous blocks if necessary and update stats
    if (not hasContiguousBlocks(bcTensorShape, bcEncShape))
    {
        DTYPE* tempBuffer = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(sizeof(DTYPE) * numel)
                                                          : MemoryAllocation(tensorCpuGpuMode, sizeof(DTYPE) * numel));
        permute(tensor, tempBuffer, bcTensorShape, dimOrder, tensorCpuGpuMode, stream);
        _updateStatsContiguous(tempBuffer, tensorCpuGpuMode, blockSize, allocator, stream);
        allocator ? allocator->deleteRaw(tempBuffer) : MemoryFree(tensorCpuGpuMode, tempBuffer);
    }
    else
    {
        _updateStatsContiguous(tensor, tensorCpuGpuMode, blockSize, allocator, stream);
    }
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::_updateStatsContiguous(const DTYPE* tensor, ComputationMode tensorCpuGpuMode,
                                                            size_t blockSize, IAllocator* allocator, void* stream)
{
    synchronizeStream(tensorCpuGpuMode, stream);
    for (size_t idx = 0; idx < _encodingAnalyzers.size(); idx++)
    {
        _encodingAnalyzers[idx]->updateStats(tensor + idx * blockSize, blockSize, tensorCpuGpuMode, allocator);
    }
}

template <typename DTYPE>
std::vector<TfEncoding> EncodingAnalyzerWrapper<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                                        bool useStrictSymmetric,
                                                                        bool useUnsignedSymmetric) const
{
    std::vector<TfEncoding> encodings(_encodingAnalyzers.size());
    for (size_t idx = 0; idx < encodings.size(); idx++)
    {
        encodings[idx] = _encodingAnalyzers[idx]->computeEncoding(bw, useSymmetricEncodings, useStrictSymmetric,
                                                                  useUnsignedSymmetric);
    }
    return encodings;
}

template <typename DTYPE>
std::vector<std::vector<std::tuple<double, double>>> EncodingAnalyzerWrapper<DTYPE>::getStatsHistogram() const
{
    std::vector<std::vector<std::tuple<double, double>>> statsHistograms(_encodingAnalyzers.size());
    for (size_t idx = 0; idx < _encodingAnalyzers.size(); idx++)
    {
        statsHistograms[idx] = _encodingAnalyzers[idx]->getStatsHistogram();
    }
    return statsHistograms;
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::setPercentileValue(float percentile)
{
    for (auto& encodingAnalyzer: _encodingAnalyzers)
    {
        encodingAnalyzer->setPercentileValue(percentile);
    }
}

template <typename DTYPE>
float EncodingAnalyzerWrapper<DTYPE>::getPercentileValue()
{
    return _encodingAnalyzers[0]->getPercentileValue();
}

template <typename DTYPE>
TensorDims EncodingAnalyzerWrapper<DTYPE>::getShape()
{
    return _shape;
}


template class EncodingAnalyzerWrapper<double>;

template class EncodingAnalyzerWrapper<float>;

}   // namespace DlQuantization