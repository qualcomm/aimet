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


#ifndef SPEC_FUNCTIONS_HPP_
#define SPEC_FUNCTIONS_HPP_

#include <iostream>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/EncodingRescale.hpp"

namespace DlQuantization
{

template <typename DTYPE>
void getRescaledOutputAndBiasImpl(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                           DTYPE* scaling_params, ComputationMode cpu_gpu_mode, bool withOffsetWrap);

template <typename DTYPE>
void getRescaledOutputAndBiasImplCpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

// GPU implementations ...
#ifdef GPU_QUANTIZATION_ENABLED
template <typename DTYPE>
void getRescaledOutputAndBiasImplGpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &hw_conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

#endif //End of GPU_QUANTIZATION_ENABLED

} // end of namespace DlQuantization

#endif // SPEC_FUNCTIONS_HPP_
