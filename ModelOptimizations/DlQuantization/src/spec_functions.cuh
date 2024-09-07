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

#include <curand_kernel.h>
#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
__device__ inline float clamp(float val, float min, float max)
{
    return fmaxf(fminf(val, max), min);
}

__device__ inline double clamp(double val, double min, double max)
{
    return fmax(fmin(val, max), min);
}

__device__ inline float round_nearest(float val)
{
    return roundf(val);
}

__device__ inline double round_nearest(double val)
{
    return round(val);
}

__device__ inline float round_downward(float val)
{
    return floorf(val);
}

__device__ inline double round_downward(double val)
{
    return floor(val);
}

__device__ inline float power(float x, float y)
{
    return powf(x, y);
}

__device__ inline double power(double x, double y)
{
    return pow(x, y);
}

__device__ float withoutOffsetWrapDevice(float offset, float requant_scale) {
    return 0;
}

__device__ float withOffsetWrapDevice(float offset, float requant_scale) {
    return offset / requant_scale;
}

typedef float(*offsetWrapPtr)(float, float);

__device__ offsetWrapPtr withoutOffsetHost = withoutOffsetWrapDevice;
__device__ offsetWrapPtr withOffsetHost = withOffsetWrapDevice;

template <typename DTYPE>
__device__ void getScaleAndBiasPerTensorDevice(const DTYPE* bias_in, const DTYPE acc_scale, const DTYPE out_offset,
                                               const DTYPE requant_scale, const int32_t bw, DTYPE* bias_out,
                                               offsetWrapPtr offset_func_ptr)
{
    DTYPE offsetWrapVal = offset_func_ptr(out_offset, requant_scale);
    DTYPE biasSim = round_nearest(*bias_in / acc_scale - offsetWrapVal);
    biasSim = round_downward(biasSim * power(2.0, 8.0 - bw));
    *bias_out = biasSim;
}

template <typename DTYPE>
__device__ void getScaleAndBiasPerChannelDevice(const DTYPE* bias_in, DTYPE acc_scale, DTYPE weight_scale,
                                                const DTYPE input_scale, DTYPE max_weight_scale, DTYPE out_scale,
                                                DTYPE* scaling_param, DTYPE out_offset, int32_t bw, DTYPE* bias_out,
                                                offsetWrapPtr offset_func_ptr)
{
    DTYPE accScaleCurr = weight_scale * input_scale;
    DTYPE normWeightScale = weight_scale / max_weight_scale;
    DTYPE requantScale = accScaleCurr / out_scale;
    *scaling_param = requantScale;
    DTYPE biasSim = round_nearest(*bias_in / accScaleCurr) * accScaleCurr;
    DTYPE offsetWrapVal = offset_func_ptr(out_offset, requantScale);
    biasSim = (biasSim / normWeightScale) / acc_scale - offsetWrapVal;
    biasSim = round_downward(biasSim * power(2.0, 8.0 - bw));
    *bias_out = biasSim;

}


}// End of namespace DlQuantization

