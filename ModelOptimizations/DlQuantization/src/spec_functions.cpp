//==============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright 2022 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@-COPYRIGHT-END-@@
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
