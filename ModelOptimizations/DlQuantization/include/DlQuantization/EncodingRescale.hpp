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
