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

#include "spec_functions.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <climits>
#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unordered_map>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/EncodingRescale.hpp"

namespace DlQuantization
{

template <typename DTYPE>
void getRescaledOutputAndBias(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                       DTYPE* bias_out, DTYPE* scaling_params, bool use_cuda, bool withOffsetWrap)
{
    DlQuantization::ComputationMode cpuGpuMode;
    setCpuGpuMode(use_cuda, cpuGpuMode);

    getRescaledOutputAndBiasImpl(bias_in, count, conv_args, bias_out, scaling_params, cpuGpuMode,
                          withOffsetWrap);
}

float withOffsetWrapHandler(float offset, float requantScale)
{
    return offset / requantScale;
}

float withoutOffsetWrapHandler(float offset, float requantScale)
{
    return 0.f;
}

/**
 * @brief Generate requant scale and bias
 * i.e. given conv
 *      [(q_input + input_offset) * input_scale] * [q_weight * weight_scale] + bias_in =
 *                                                                             (q_output + output_offset) * output_scale
 * -find
 *      q_output = [(input_scale * weight_scale)/output_scale] * {[(q_input + input_offset) * q_weight] +
 *                 [bias_in/(input_scale * weight_scale)] - [output_offset * output_scale/(input_scale * weight_scale)]}
 * 
 * q_input: unsigned fixed-point input, q_weight: signed fixed-point weight, bias_in: floating-point biases,
 * q_output: unsigned fixed-point output
 * *_scale, *_offset: variables with relative scale and zero_offset(negative)
 * 
 * @return requant scale: (input_scale * weight_scale)/output_scale,
 *         bias: [bias_in/(input_scale * weight_scale)] - [output_offset * output_scale/(input_scale * weight_scale)]
*/

template <typename DTYPE>
void getRescaledOutputAndBiasImplCpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                              DTYPE* bias_out, DTYPE* scaling_params, bool withOffsetWrap)
{
    std::vector<DTYPE> weightScale = conv_args.weight_scale;
    size_t weightLen = weightScale.size();
    DTYPE maxWeightScale = *max_element(weightScale.begin(), weightScale.end());
    DTYPE accScale = maxWeightScale * conv_args.input_scale;

    if (conv_args.bw != 8 && conv_args.bw != 16)
        throw std::runtime_error("currently Quant func only support 8 or 16 bit");

    auto offsetWrapFunc = withoutOffsetWrapHandler;
    if (withOffsetWrap)
    {
        offsetWrapFunc = withOffsetWrapHandler;
    }

    // get perchannel quantization's requant scale and bias
    if(count == weightLen)
    {
        DTYPE accScaleCurr;
        DTYPE normWeightScale;
        for(int i = 0; i < weightLen; ++i)
        {
            accScaleCurr = weightScale[i] * conv_args.input_scale;
            normWeightScale = weightScale[i] / maxWeightScale;
            DTYPE requantScale = accScaleCurr / conv_args.out_encoding_delta;
            *(scaling_params + i) = requantScale;

            DTYPE biasSim = round(*(bias_in + i) / accScaleCurr) * accScaleCurr;

            DTYPE offsetWrapVal = offsetWrapFunc(conv_args.out_encoding_offset, requantScale);
            biasSim = (biasSim / normWeightScale) / accScale - offsetWrapVal;
            // simulate operation, biasSim should be right shift 8 bits when bitwidth is 16.
            biasSim = floor(biasSim * pow(2, 8 - conv_args.bw));
            *(bias_out + i) = biasSim;
        }
    }
    //get pertensor quantization's requant scale and bias
    else if(weightLen == 1)
    {
        DTYPE requantScale = accScale / conv_args.out_encoding_delta;
        *scaling_params = requantScale;
        for(int i = 0; i < count; ++i)
        {
            DTYPE offsetWrapVal = offsetWrapFunc(conv_args.out_encoding_offset, requantScale);
            DTYPE biasSim = round(*(bias_in + i) / accScale - offsetWrapVal);
            biasSim = floor(biasSim * pow(2, 8 - conv_args.bw));

            *(bias_out + i) = biasSim;
        }
    }
    else
    {
        throw std::runtime_error("The len of weight_scale should be 1 or equal to the len of bias");
    }

}

template <typename DTYPE>
void getRescaledOutputAndBiasImpl(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                           DTYPE* bias_out, DTYPE* scaling_params, ComputationMode cpu_gpu_mode, bool withOffsetWrap)
{
    switch (cpu_gpu_mode)
    {
    case COMP_MODE_CPU:
        getRescaledOutputAndBiasImplCpu(bias_in, count, conv_args, bias_out, scaling_params, withOffsetWrap);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        getRescaledOutputAndBiasImplGpu(bias_in, count, conv_args, bias_out, scaling_params, withOffsetWrap);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
        break;
    }
    default:
        throw std::runtime_error("Unknown computation mode.");
        break;
    }


}


// Explicit instantiations
template void getRescaledOutputAndBiasImpl(const float* bias_in, const int count, ConvSpecArgs<float> &conv_args,
                                   float* bias_out, float* scaling_params, ComputationMode cpu_gpu_mode,
                                   bool withOffsetWrap);
template void getRescaledOutputAndBiasImpl(const double* bias_in, const int count, ConvSpecArgs<double> &conv_args,
                                   double* bias_out, double* scaling_params, ComputationMode cpu_gpu_mode,
                                   bool withOffsetWrap);

template void getRescaledOutputAndBias(const float* bias_in, const int count, ConvSpecArgs<float> &conv_args,
                                float* bias_out, float* scaling_params, bool use_cuda, bool withOffsetWrap);

}  // end of DlQuantization
