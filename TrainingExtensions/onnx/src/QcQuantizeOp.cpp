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

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"


#include <vector>
#include <iostream>


#ifdef ONNX_CUDA
static OnnxCudaAllocator cudaAllocator;
#endif
static OnnxCpuAllocator cpuAllocator;


QcQuantizeOp::QcQuantizeOp(const OrtApi* api, const OrtKernelInfo* info) : api_(*api), info_(info)
{
    int64_t quantInfoPointer;
    api->KernelInfoGetAttribute_int64(info_, "quant_info", &quantInfoPointer);
    quantInfo = reinterpret_cast<struct QcQuantizeInfo*>(quantInfoPointer);
}


void QcQuantizeOp::computeImpl(const Ort::Custom::Tensor<float>& input, Ort::Custom::Tensor<float>& output,
                               void* stream, bool useCuda, DlQuantization::IAllocator* allocator)
{
    // Setup inputs
    auto inputData  = input.Data();
    auto inputShape = input.Shape();
    auto size     = input.NumberOfElement();
    auto result     = output.Allocate(inputShape);

    std::vector<DlQuantization::TfEncoding*> encodings = quantInfo->encoding;

    DlQuantization::TensorQuantizerOpMode opMode = quantInfo->opMode;
    // Disable unused quantizers
    if (!quantInfo->enabled)
    {
        opMode = DlQuantization::TensorQuantizerOpMode::passThrough;
    }

    if (quantInfo->isIntDataType)
    {
        if (quantInfo->usePerChannelMode)
        {
            const int channelAxis = quantInfo->channelAxis;
            const int blockAxis = (quantInfo-> blockSize == 0) ? -1 : quantInfo->blockAxis;
            const BroadcastShapeInfo shapeInfo {inputShape, channelAxis, blockAxis, quantInfo->blockSize};
            modeSpecificActionBroadcastInt(inputData, result, shapeInfo, quantInfo->tensorQuantizerRef, opMode,
                                   encodings, quantInfo->useSymmetricEncoding, allocator, useCuda, stream);
        }
        else
        {
            modeSpecificActionInt(inputData, size, result, quantInfo->tensorQuantizerRef[0], opMode, encodings[0],
                                  quantInfo->useSymmetricEncoding, allocator, useCuda, stream);
        }
    }
    else
    {
        modeSpecificActionFloat(inputData, size, result, opMode, allocator, useCuda, stream);
    }

    // We only ever need to run in oneShotQuantizeDequantize once, afterwards just use quantizeDequantize
    if (opMode == DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
    {
        quantInfo->opMode = DlQuantization::TensorQuantizerOpMode::quantizeDequantize;
    }
}


struct QcQuantizeOpCpu : QcQuantizeOp
{
    using QcQuantizeOp::QcQuantizeOp;

    void Compute(const Ort::Custom::Tensor<float>& input, Ort::Custom::Tensor<float>& output)
    {
        computeImpl(input, output, nullptr, false, &cpuAllocator);
    }
};


#ifdef ONNX_CUDA

struct QcQuantizeOpCuda : QcQuantizeOp
{
    using QcQuantizeOp::QcQuantizeOp;

    void Compute(const Ort::Custom::CudaContext& cuda_ctx, const Ort::Custom::Tensor<float>& input,
                 Ort::Custom::Tensor<float>& output)
    {
        cudaStream_t stream = cuda_ctx.cuda_stream;
        if ((quantInfo->opMode == DlQuantization::TensorQuantizerOpMode::updateStats) ||
            (quantInfo->opMode == DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize))
        {
            // updateStats doesn't use cuda stream, must synchronize first to ensure input buffer is populated
            cudaStreamSynchronize(stream);
        }

        computeImpl(input, output, stream, true, &cudaAllocator);
    }
};

#endif


void RegisterOps(Ort::CustomOpDomain& domain)
{
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpu>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpPointer.get());
#ifdef ONNX_CUDA
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCuda>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpPointer.get());
#endif
}
