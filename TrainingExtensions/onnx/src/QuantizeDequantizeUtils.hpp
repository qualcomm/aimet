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

#ifndef AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
#define AIMET_QUANTIZEDEQUANTIZEUTILS_HPP

#include "DlQuantization/TensorQuantizer.h"
#include "OnnxOpUtils.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif


template <typename T>
void sliceTensorChannelGPU(const T* inTensor, T* outTensor, long iters, long copyWidth, long inputStride,
                           long outputStride, long inputOffset, long outputOffset);


template <typename T>
void sliceTensorAlongAxis(const T* inTensor, std::vector<int64_t>& dims, size_t axis, size_t channel, T* outTensor,
                          bool useCuda)
{
    uint64_t copyWidth = 1;
    uint64_t iter      = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        if (i < axis)
        {
            iter *= dims[i];
        }
        else if (i > axis)
        {
            copyWidth *= dims[i];
        }
    }
    uint64_t incr         = copyWidth * dims[axis];
    uint64_t inputOffset  = copyWidth * channel;
    uint64_t outputStride = copyWidth;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        sliceTensorChannelGPU(inTensor, outTensor, iter, copyWidth, incr, outputStride, inputOffset, 0);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        for (long i = 0; i < iter; i++)
        {
            std::copy(inTensor + inputOffset, inTensor + inputOffset + copyWidth, outTensor + i * copyWidth);
            inputOffset += incr;
        }
    }
}


template <typename T>
void permuteTensorGPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides);

template <typename T>
void permuteTensorCPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides);

std::vector<int64_t> shapeToStrides(const std::vector<int64_t>& shape);

int64_t getNumElements(const std::vector<int64_t>& shape);

template <typename T>
void quantizeDequantizePerChannel(
    const T* inTensor, std::vector<int64_t>& shape, int axis, T* outTensor,
    std::vector<DlQuantization::TfEncoding*>& encodings,
    std::vector<DlQuantization::TensorQuantizer*>& tensorQuantizers, bool useCuda,
    DlQuantization::IAllocator* allocator, void* stream,
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float> >& tensorQuantizationSim)
{
    size_t channels   = shape[axis];
    size_t numElement = 1;
    size_t innerDims  = 1;
    for (int i = 0; i < shape.size(); i++)
    {
        numElement *= shape[i];
        if (i > axis)
        {
            innerDims *= shape[i];
        }
    }

    T encVec[4][channels];
    for (int i = 0; i < channels; i++)
    {
        encVec[0][i] = encodings[i]->min;
        encVec[1][i] = encodings[i]->max;
        encVec[2][i] = encodings[i]->delta;
        encVec[3][i] = encodings[i]->offset;
    }
    T* encodingVectorDevice;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        encodingVectorDevice = (T*) allocator->allocateRaw(4 * channels * sizeof(T));
        cudaMemcpy(encodingVectorDevice, encVec, 4 * channels * sizeof(T), cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        encodingVectorDevice = (T*) encVec;
    }

    T* encodingMin    = encodingVectorDevice;
    T* encodingMax    = encodingVectorDevice + channels;
    T* encodingDelta  = encodingVectorDevice + 2 * channels;
    T* encodingOffset = encodingVectorDevice + 3 * channels;

    tensorQuantizationSim->quantizeDequantizeTensorPerChannel(inTensor, channels, numElement, innerDims, outTensor,
                                                              encodingMin, encodingMax, encodingDelta, encodingOffset,
                                                              tensorQuantizers[0]->roundingMode, useCuda, stream);
    if (useCuda)
    {
        allocator->deleteRaw(encodingVectorDevice);
    }
}


struct BroadcastShapeInfo
{
    BroadcastShapeInfo(const std::vector<int64_t>& inputShape, int channelAxis, int blockAxis, int blockSize);

    bool hasContiguousBlocks() const;

    std::vector<int64_t> tensorShape;
    std::vector<int64_t> encodingShape;
    std::vector<int64_t> tensorStrides;
    std::vector<int64_t> encodingStrides;
    int64_t numElements;
    int64_t numEncodings;
    int64_t numDims;
};

// Permutes the input data so each entire encoding block is contiguous in memory
template <typename T>
void copyToContiguousBlockLayout(const T* inTensor, T* outTensor, const BroadcastShapeInfo& shapeInfo, bool useCuda);


template <typename T>
void quantizeDequantizeBroadcast(const T* inTensor, T* outTensor, const BroadcastShapeInfo& shapeInfo,
                                 std::vector<DlQuantization::TfEncoding*>& encodings, const bool useCuda,
                                 DlQuantization::IAllocator* allocator, void* stream)
{
    if (!shapeInfo.numEncodings == encodings.size())
        throw std::runtime_error("encodings.size() does not match shapeInfo.numEncodings");

    auto numEncodings              = shapeInfo.numEncodings;
    auto numDims                   = shapeInfo.numDims;
    const int64_t* inputStrides    = shapeInfo.tensorStrides.data();
    const int64_t* encodingStrides = shapeInfo.encodingStrides.data();

    // Kernels expect separate lists for each encoding type
    std::vector<T> encVec(4 * numEncodings);

    for (int i = 0; i < numEncodings; i++)
    {
        encVec[i] = encodings[i]->min;
        encVec[numEncodings + i] = encodings[i]->max;
        encVec[2 * numEncodings + i] = encodings[i]->delta;
        encVec[3 * numEncodings + i] = encodings[i]->offset;
    }
    T* encodingVectorDevice;
    int64_t* stridesDevice = nullptr;
    DlQuantization::ComputationMode mode;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        // Allocate device memory for strides and encodings
        stridesDevice        = static_cast<int64_t*>(allocator->allocateRaw(2 * numDims * sizeof(int64_t)));
        encodingVectorDevice = static_cast<T*>(allocator->allocateRaw(4 * numEncodings * sizeof(T)));

        // Send encoding information to device
        cudaMemcpyAsync(encodingVectorDevice, encVec.data(), 4 * numEncodings * sizeof(T), cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream));

        // Send stride information to device
        int64_t* strideBuffer[2 * numDims];
        memcpy(strideBuffer, inputStrides, numDims * sizeof(int64_t));
        memcpy(strideBuffer + numDims, encodingStrides, numDims * sizeof(int64_t));
        cudaMemcpyAsync(stridesDevice, strideBuffer, 2 * numDims * sizeof(int64_t), cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream));
        inputStrides    = stridesDevice;
        encodingStrides = stridesDevice + numDims;

        mode = DlQuantization::ComputationMode::COMP_MODE_GPU;
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        mode                 = DlQuantization::ComputationMode::COMP_MODE_CPU;
        encodingVectorDevice = encVec.data();
    }

    T* encodingMin    = encodingVectorDevice;
    T* encodingMax    = encodingVectorDevice + numEncodings;
    T* encodingDelta  = encodingVectorDevice + 2 * numEncodings;
    T* encodingOffset = encodingVectorDevice + 3 * numEncodings;

    DlQuantization::quantizeDequantizeBroadcast(inTensor, outTensor, shapeInfo.numElements, numDims, inputStrides,
                                                encodingStrides, encodingMin, encodingMax, encodingDelta,
                                                encodingOffset, mode, stream);


    if (useCuda)
    {
        allocator->deleteRaw(encodingVectorDevice);
        allocator->deleteRaw(stridesDevice);
    }
}


#endif   // AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
