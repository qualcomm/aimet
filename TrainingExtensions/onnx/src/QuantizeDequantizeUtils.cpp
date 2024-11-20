//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "QuantizeDequantizeUtils.hpp"

std::vector<int64_t> shapeToStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides;
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--)
    {
        strides.push_back(stride);
        stride *= shape[i];
    }

    std::reverse(strides.begin(), strides.end());
    return strides;
}

int64_t getNumElements(const std::vector<int64_t>& shape)
{
    int64_t numEncodings = 1;
    for (const auto& i: shape)
    {
        numEncodings *= i;
    }
    return numEncodings;
}

template <typename T>
void permuteTensorCPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides)
{
    int64_t chunkSize = numel;
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
            remainder = remainder - dimIdx * inputStrides[dim];
            outputIdx += outputStrides[dim] * dimIdx;
        }

        std::copy(inTensor + i, inTensor + i + chunkSize, outTensor + outputIdx);
    }
}


template void permuteTensorCPU(const float* intensor, float* outTensor, int64_t numel, int64_t numDims,
                               const int64_t* inputStrides, const int64_t* outputStrides);


BroadcastShapeInfo::BroadcastShapeInfo(const std::vector<int64_t>& inputShape, const int channelAxis,
                                       const int blockAxis, const uint blockSize)
{
    numElements = getNumElements(inputShape);
    std::vector<int64_t> encShape;
    std::vector<int64_t> newInputShape;

    // View the input and encodings with broadcastable shapes
    for (int i = 0; i < inputShape.size(); i++)
    {
        if (i == channelAxis)
        {
            newInputShape.push_back(inputShape[i]);
            encShape.push_back(inputShape[i]);
        }
        else if (i == blockAxis)
        {
            if (inputShape[i] % blockSize != 0)
            {
                throw std::runtime_error(std::string("Block dimension is not evenly divisible by block size."));
            }
            int64_t numBlocks = inputShape[i] / blockSize;
            newInputShape.push_back(numBlocks);
            newInputShape.push_back(blockSize);
            encShape.push_back(numBlocks);
            encShape.push_back(1);
        }
        else
        {
            newInputShape.push_back(inputShape[i]);
            encShape.push_back(1);
        }
    }

    numDims         = newInputShape.size();
    tensorShape     = newInputShape;
    encodingShape   = encShape;
    tensorStrides   = shapeToStrides(newInputShape);
    encodingStrides = shapeToStrides(encShape);
    numEncodings    = getNumElements(encShape);

    for (int i = 0; i < encShape.size(); i++)
    {
        if ((encShape[i] == 1) and newInputShape[i] != 1)
        {
            encodingStrides[i] = 0;
        }
    }

    if (tensorStrides.size() != numDims)
        throw std::runtime_error("Tensor stride vector length does not match expected number of dimensions");

    if (encodingStrides.size() != numDims)
        throw std::runtime_error("Encoding stride vector length does not match expected number of dimensions");
}


bool BroadcastShapeInfo::hasContiguousBlocks() const
{
    bool isPreviousDimBroadcast = false;
    for (int idx = 0; idx < numDims; idx++)
    {
        // If we have a broadcast dimension followed by a non-broadcast dimension, blocks are discontiguous
        if (isPreviousDimBroadcast and tensorShape[idx] == encodingShape[idx])
        {
            return false;
        }
        isPreviousDimBroadcast = tensorShape[idx] != encodingShape[idx];
    }
    return true;
}


template <typename T>
void copyToContiguousBlockLayout(const T* inTensor, T* outTensor, const BroadcastShapeInfo& shapeInfo, bool useCuda)
{
    auto encodingStrides = shapeInfo.encodingStrides;
    auto inputStrides    = shapeInfo.tensorStrides;
    int64_t numDims      = inputStrides.size();
    int64_t numel        = shapeInfo.numElements;

    std::vector<int64_t> broadcastDims, nonBroadcastDims;

    // Determine the dim ordering such that all indexes in a single quantization block are contiguous
    for (int64_t i = 0; i < shapeInfo.numDims; i++)
    {
        if (shapeInfo.encodingStrides[i] != 0)
        {
            nonBroadcastDims.push_back(i);
        }
        else
        {
            broadcastDims.push_back(i);
        }
    }
    std::vector<int64_t> dimOrder = nonBroadcastDims;
    dimOrder.insert(dimOrder.end(), broadcastDims.begin(), broadcastDims.end());

    // Determine what the strides of the output tensor will be in the new dimension ordering
    std::vector<int64_t> outputStrides(numDims);
    outputStrides[dimOrder[numDims - 1]] = 1;
    for (int64_t i = numDims - 2; i >= 0; --i)
    {
        outputStrides[dimOrder[i]] = outputStrides[dimOrder[i + 1]] * shapeInfo.tensorShape[dimOrder[i + 1]];
    }

    if (useCuda)
    {
#ifdef ONNX_CUDA
        permuteTensorGPU(inTensor, outTensor, numel, numDims, shapeInfo.tensorStrides.data(), outputStrides.data());
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        permuteTensorCPU(inTensor, outTensor, numel, numDims, shapeInfo.tensorStrides.data(), outputStrides.data());
    }
}

template void copyToContiguousBlockLayout(const float* inTensor, float* outTensor, const BroadcastShapeInfo& shapeInfo, bool useCuda);
