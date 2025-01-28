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

#ifndef AIMET_MAIN_AIMETOPUTILS_H
#define AIMET_MAIN_AIMETOPUTILS_H

#include <numeric>
#include "DlQuantization/Fp16Quantization.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/TensorQuantizerOpFacade.h"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/Default/Half.h"
#include "OnnxOpUtils.h"

#include <cstdint>
#include <stdexcept>
#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef ONNX_CUDA
class OnnxCudaAllocator : public DlQuantization::IAllocator
{
public:
    void* allocateRaw(size_t bytes) override
    {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    void deleteRaw(void* ptr) override
    {
        cudaFree(ptr);
    }
};
#endif

class OnnxCpuAllocator : public DlQuantization::IAllocator
{
public:
    void* allocateRaw(size_t bytes) override
    {
        void* ptr;
        ptr = malloc(bytes);
        return ptr;
    }

    void deleteRaw(void* ptr) override
    {
        free(ptr);
    }
};

template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda, void* stream);

void quantizeDequantizeFp16Cpu(const float* in, int cnt, float* out);

template <typename T>
void modeSpecificActionBroadcastInt(const T* inTensor, T* outTensor, const std::vector<int64_t> inputShape,
                                    DlQuantization::BlockTensorQuantizer* tensorQuantizer,
                                    const DlQuantization::TensorQuantizerOpMode opMode,
                                    const bool useSymmetricEncoding, DlQuantization::IAllocator* allocator,
                                    bool useCuda, void* stream)
{
    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        tensorQuantizer->resetEncodingStats();
        tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
        auto computedEncodings = tensorQuantizer->computeEncodings(useSymmetricEncoding);
        tensorQuantizer->setEncodings(computedEncodings);
        // Continue to quantizeDequantize
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        tensorQuantizer->quantizeDequantize(inTensor, outTensor, inputShape, useCuda, stream);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
        // Continue to passThrough
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        int64_t numElements = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<size_t>());
        copyInputTensorsToOutputTensors(inTensor, numElements, outTensor, useCuda, stream);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

template <typename T>
void modeSpecificActionFloat(const T* inTensor, size_t count, T* outTensor,
                             const DlQuantization::TensorQuantizerOpMode opMode, DlQuantization::IAllocator* allocator,
                             bool useCuda, void* stream)
{
    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        if (useCuda)
        {
            DlQuantization::quantizeDequantizeFp16Gpu(inTensor, count, outTensor, stream);
        }
        else
            quantizeDequantizeFp16Cpu(inTensor, count, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda, stream);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

#endif   // AIMET_MAIN_AIMETOPUTILS_H
