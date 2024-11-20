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


const int CUDA_NUM_THREADS = 512;

// Compute the number of blocks based on the total number of threads.
inline int CUDA_NUM_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename DTYPE>
__global__ void permuteTensorKernel(const DTYPE* in, DTYPE* out, int numElements, int numDims,
                                    const int64_t* inputStrides, const int64_t* outputStrides)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x)
    {
        size_t outputIdx = 0;
        size_t remainder = i;
        for (auto dim = 0; dim < numDims; dim++)
        {
            size_t dimIdx = remainder / inputStrides[dim];
            remainder = remainder - dimIdx * inputStrides[dim];
            outputIdx += outputStrides[dim] * dimIdx;
        }

        out[outputIdx] = in[i];
    }
}


template <typename T>
void permuteTensorGPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides)
{
    int64_t totalThreads = numel;
    int64_t gridSize     = CUDA_NUM_BLOCKS(totalThreads);
    int64_t strideData[2][numDims];

    // Copy the stride information to the cuda device
    for (int i = 0; i < numDims; i++)
    {
        strideData[0][i] = inputStrides[i];
        strideData[1][i] = outputStrides[i];
    }
    int64_t* deviceStrideData;
    cudaMalloc((void**) &deviceStrideData, 2 * numDims * sizeof(int64_t));
    cudaMemcpy(deviceStrideData, strideData, 2 * numDims * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch the cuda kernel
    permuteTensorKernel<<<gridSize, CUDA_NUM_THREADS>>>(inTensor, outTensor, numel, numDims, deviceStrideData,
                                                        deviceStrideData + numDims);

    // Free the device stride data
    cudaFree(deviceStrideData);
}


template void permuteTensorGPU(const float* intensor, float* outTensor, int64_t numel, int64_t numDims,
                               const int64_t* inputStrides, const int64_t* outputStrides);