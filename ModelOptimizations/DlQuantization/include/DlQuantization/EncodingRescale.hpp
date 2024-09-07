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

#ifndef ENCODING_RESCALE_HPP_
#define ENCODING_RESCALE_HPP_
#include <cstddef>
#include <iostream>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{

/**
 * @brief Arguments used for simulating on-device convolution
 */
template<typename DTYPE>
struct ConvSpecArgs
{
    // delta of output encoding of convolution
    float out_encoding_delta;
    // offset of output encoding of convolution
    float out_encoding_offset;
    // delta of input encoding of convolution
    float input_scale;
    // quantization bitwidths
    uint8_t bw;
    // weight scales of weight encodings of convolution, if the quantization scheme is perchannel, the length of 
    // weight_scale is equal to the count, if the quantization scheme is pertensor, the length of weight_scale is 1.
    std::vector<DTYPE> weight_scale;   
};

/**
 * @brief returns the exponent and mantissa of x, as a n-bit number
 *
 * Constraint: iexpo must be in range -126..127
 * Input must not be negative, inf, nan, zero, or denormal.
 */
inline std::pair<int32_t, int32_t> getScaleFactor(float x, int mbits)
{
    int32_t inval = *reinterpret_cast<int *>(&x);
    int MBITS = mbits;
    int32_t mask = (1 << MBITS) - 1;
    inval = (inval + (1 << (24 - MBITS - 1))) >> (24 - MBITS);
    int32_t m = ((inval & mask) | (1 << (MBITS - 1)));
    int32_t e = int32_t((inval >> (MBITS - 1)) & 0xFF) - 126;
    if (e < -23)
        e = -9999;
    return {e, m};
}

inline void setCpuGpuMode(bool use_cuda, DlQuantization::ComputationMode& cpu_gpu_mode)
{
    if (use_cuda)
        cpu_gpu_mode = DlQuantization::ComputationMode::COMP_MODE_GPU;
    else
        cpu_gpu_mode = DlQuantization::ComputationMode::COMP_MODE_CPU;
}

template <typename DTYPE>
void getRescaledOutputAndBias(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                       DTYPE* bias_out, DTYPE* scaling_params, bool use_cuda, bool withOffsetWrap);


} // end of namespace DlQuantization
#endif // end of ENCODING_RESCALE_HPP_
