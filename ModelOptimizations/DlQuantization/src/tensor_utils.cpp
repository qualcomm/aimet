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

#include "tensor_utils.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace DlQuantization
{
TensorDims _padToBroadcastLength(const TensorDims& vector, const size_t length)
{
    TensorDims expanded(length);
    auto padding = length - vector.size();
    for (size_t idx = 0; idx < length; idx++)
    {
        if (idx < padding)
            expanded[idx] = 1;
        else
            expanded[idx] = vector[idx - padding];
    }
    return expanded;
}

std::tuple<TensorDims, TensorDims> getBroadcastableShapes(const TensorDims& tensorShape,
                                                          const TensorDims& encodingShape)
{
    auto maxLen      = tensorShape.size();
    auto expEncShape = _padToBroadcastLength(encodingShape, maxLen);

    TensorDims bcTensorShape;
    TensorDims bcEncodingShape;
    for (size_t idx = 0; idx < maxLen; idx++)
    {
        auto dim1 = tensorShape[idx];
        auto dim2 = expEncShape[idx];
        if (dim1 == dim2 || (dim1 == 1 && dim2 == 1))
        {
            bcTensorShape.push_back(dim1);
            bcEncodingShape.push_back(dim2);
        }
        else
        {
            if (dim1 < dim2 || dim1 % dim2 != 0)
            {
                throw std::runtime_error("Cannot interpret tensor and encoding dimensions as broadcastable");
            }
            bcTensorShape.push_back(dim2);
            bcTensorShape.push_back(dim1 / dim2);
            bcEncodingShape.push_back(dim2);
            bcEncodingShape.push_back(1);
        }
    }
    return std::make_tuple(std::move(bcTensorShape), std::move(bcEncodingShape));
}

size_t getNumel(const TensorDims& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

TensorDims shapeToStrides(const TensorDims& shape)
{
    TensorDims strides;
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--)
    {
        strides.push_back(stride);
        stride *= shape[i];
    }

    std::reverse(strides.begin(), strides.end());
    return strides;
}

bool hasContiguousBlocks(const TensorDims& tensorShape, const TensorDims& encodingShape)
{
    bool isPreviousDimBroadcast = false;
    auto padEncodingShape = _padToBroadcastLength(encodingShape, tensorShape.size());
    for (int idx = 0; idx < tensorShape.size(); idx++)
    {
        if (tensorShape[idx] == 1)
        {
            continue;
        }
        // If we have a broadcast dimension followed by a non-broadcast dimension, blocks are discontiguous
        if (isPreviousDimBroadcast and tensorShape[idx] == padEncodingShape[idx])
        {
            return false;
        }
        isPreviousDimBroadcast = tensorShape[idx] != padEncodingShape[idx];
    }
    return true;
}

template <typename T>
void permute(const T* input, T* output, const TensorDims& inputShape, std::vector<size_t> order, ComputationMode mode,
             void* stream)
{
    size_t numDims          = inputShape.size();
    TensorDims inputStrides = shapeToStrides(inputShape);
    TensorDims outputStrides(numDims);
    outputStrides[order[numDims - 1]] = 1;
    for (int64_t i = numDims - 2; i >= 0; --i)
    {
        outputStrides[order[i]] = outputStrides[order[i + 1]] * inputShape[order[i + 1]];
    }
    size_t numel = getNumel(inputShape);

    switch (mode)
    {
    case COMP_MODE_CPU:
        permuteKernelCPU(input, output, numel, inputStrides, outputStrides);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        permuteKernelGPU(input, output, numel, inputStrides, outputStrides, stream);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw std::runtime_error("Unknown computation mode.");
    }
}

template <typename T>
void permuteKernelCPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides)
{
    int64_t chunkSize = numel;
    size_t numDims    = inputStrides.size();
    // Get the largest already-contiguous chunk size
    for (int64_t i = numDims - 1; i >= 0; i--)
    {
        if (inputStrides[i] != outputStrides[i])
        {
            chunkSize = inputStrides[i];
            break;
        }
    }

    for (size_t i = 0; i < numel; i += chunkSize)
    {
        size_t outputIdx = 0;
        size_t remainder = i;
        for (auto dim = 0; dim < numDims; dim++)
        {
            size_t dimIdx = remainder / inputStrides[dim];
            remainder     = remainder - dimIdx * inputStrides[dim];
            outputIdx += outputStrides[dim] * dimIdx;
        }

        std::copy(inTensor + i, inTensor + i + chunkSize, outTensor + outputIdx);
    }
}

void synchronizeStream(ComputationMode mode, void* stream)
{
    if (mode == COMP_MODE_GPU)
    {
#ifdef GPU_QUANTIZATION_ENABLED
        synchronizeCudaStream(stream);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
}

template void permute(const float* input, float* output, const TensorDims& inputShape, std::vector<size_t> order,
                      ComputationMode mode, void* stream);

template void permute(const double* input, double* output, const TensorDims& inputShape, std::vector<size_t> order,
                      ComputationMode mode, void* stream);

template void permuteKernelCPU(const float* inTensor, float* outTensor, size_t numel, const TensorDims& inputStrides,
                               const TensorDims& outputStrides);

template void permuteKernelCPU(const double* inTensor, double* outTensor, size_t numel, const TensorDims& inputStrides,
                               const TensorDims& outputStrides);

}   // namespace DlQuantization
