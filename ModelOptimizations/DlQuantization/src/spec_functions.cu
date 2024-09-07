//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>

#include "spec_functions.hpp"
#include "spec_functions.cuh"
#include "cuda_util.hpp"
#include "math_functions.hpp"
#include "DlQuantization/EncodingRescale.hpp"


namespace DlQuantization
{
template <typename DTYPE>
__global__ void getRescaledOutputAndBiasPerChannelKernel(const DTYPE* bias_in, const int count, DTYPE* bias_out,
                                                  const DTYPE input_scale, const DTYPE* weight_scale, DTYPE acc_scale,
                                                  DTYPE* scaling_params, int32_t bitwidth, DTYPE out_offset,
                                                  DTYPE out_scale, DTYPE max_weight_scale,
                                                  offsetWrapPtr offset_func_ptr)
{
    CUDA_KERNEL_LOOP(i, count)
    {
        getScaleAndBiasPerChannelDevice<DTYPE>(bias_in + i, acc_scale, *(weight_scale + i), input_scale,
                                               max_weight_scale, out_scale, scaling_params + i, out_offset,
                                               bitwidth, bias_out + i, offset_func_ptr);
    }
}

template <typename DTYPE>
__global__ void getRescaledOutputAndBiasPerTensorKernel(const DTYPE* bias_in, const int count, DTYPE* bias_out,
                                                 DTYPE acc_scale, DTYPE* scaling_params, int32_t bitwidth,
                                                 DTYPE out_offset, DTYPE requant_scale,
                                                 offsetWrapPtr offset_func_ptr)
{
    *scaling_params = requant_scale;
    CUDA_KERNEL_LOOP(i, count)
    {
        getScaleAndBiasPerTensorDevice<DTYPE>(bias_in + i, acc_scale, out_offset, requant_scale, bitwidth,
                                              bias_out + i, offset_func_ptr);
    }
}

template <typename DTYPE>
void getRescaledOutputAndBiasImplGpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                             DTYPE* bias_out, DTYPE* scaling_params, bool withOffsetWrap)
{
    std::vector<DTYPE> weightScale = conv_args.weight_scale;
    int weightLen = weightScale.size();
    DTYPE maxWeightScale = *max_element(weightScale.begin(), weightScale.end());
    void* devPtr;
    devPtr = MemoryAllocation_gpu(sizeof(DTYPE) * weightLen);
    DTYPE accScale = conv_args.input_scale * maxWeightScale;
    DTYPE requantScale = accScale / conv_args.out_encoding_delta;
    CudaMemCpy(devPtr, &(weightScale[0]), sizeof(DTYPE) * weightLen, CudaMemcpyDirection::HOST_TO_DEVICE);
    offsetWrapPtr offsetWrap;
    // Copy device function pointer to host side
    if (withOffsetWrap)
    {
        cudaMemcpyFromSymbol(&offsetWrap, withOffsetHost, sizeof(offsetWrapPtr));
    }
    else
    {
        cudaMemcpyFromSymbol(&offsetWrap, withoutOffsetHost, sizeof(offsetWrapPtr));
    }

    if(weightLen == count)
    {
        getRescaledOutputAndBiasPerChannelKernel<DTYPE><<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS>>>(bias_in, count,
                                                                                               bias_out, conv_args.input_scale,
                                                                                               reinterpret_cast<DTYPE*>(devPtr),
                                                                                               accScale, scaling_params,
                                                                                               conv_args.bw,
                                                                                               conv_args.out_encoding_offset,
                                                                                               conv_args.out_encoding_delta,
                                                                                               maxWeightScale, offsetWrap);
    }
    else if(weightLen == 1)
    {
        getRescaledOutputAndBiasPerTensorKernel<DTYPE><<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS>>>(bias_in, count, bias_out,
                                                                                              accScale, scaling_params,
                                                                                              conv_args.bw,
                                                                                              conv_args.out_encoding_offset,
                                                                                              requantScale, offsetWrap);
    }
    else
    {
        throw std::runtime_error("The len of weight_scale should be 1 or equal to the len of bias");
    }
    MemoryFree_gpu(devPtr);
}


// Explicit instantiations
template void getRescaledOutputAndBiasImplGpu(const float* bias_in, const int count, ConvSpecArgs<float> &conv_args,
                                      float* bias_out, float* scaling_params, bool withOffsetWrap);
template void getRescaledOutputAndBiasImplGpu(const double* bias_in, const int count, ConvSpecArgs<double> &conv_args,
                                      double* bias_out, double* scaling_params, bool withOffsetWrap);

} // End of namespace DlQuantization
