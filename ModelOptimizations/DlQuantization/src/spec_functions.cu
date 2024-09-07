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
