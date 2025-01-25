//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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


#ifndef UTIL_TRIM_FUNCTIONS_HPP_
#define UTIL_TRIM_FUNCTIONS_HPP_

#include <cstdint>
#include <vector>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
inline double randUniformCpu();

template <typename DTYPE>
void quantizeDequantize(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                        RoundingMode rounding_mode, void* stream);


void quantizeDequantizeFp16ForGPU(const float* in, int cnt, float* out, void* stream);


template <typename DTYPE>
void quantizeToFxp(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                   RoundingMode rounding_mode, bool shiftToSigned);

template <typename DTYPE>
void quantizeToFxpPacked(const DTYPE* in, int cnt, const TfEncoding& encoding, uint8_t* out, size_t out_size,
                         ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);

template <typename DTYPE>
void dequantizeFromPackedFxp(const uint8_t* input, int cnt, const TfEncoding& encoding, DTYPE* output,
                             ComputationMode mode_cpu_gpu, bool shiftToSigned);

template <typename DTYPE>
void quantizeDequantizeCpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                           RoundingMode rounding_mode);

template <typename DTYPE>
void quantizeToFxpCpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned);

template <typename DTYPE>
void quantizeToFxpPackedCpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, size_t out_size,
                            RoundingMode rounding_mode, bool shiftToSigned);

// Multi-threading implementation
template <typename DTYPE>
void dequantizeFromPackedFxpCpuMt(const uint8_t* input, int cnt, const TfEncoding& encoding, DTYPE* output,
                                  bool shiftToSigned);

template <typename DTYPE>
void dequantizeFromPackedFxpCpu(const uint8_t* input, int cnt, const TfEncoding& encoding, DTYPE* output,
                                bool shiftToSigned);

double computeDelta(double encodingMin, double encodingMax, double numSteps);
double computeOffset(double encodingMin, double delta);

template <typename DTYPE>
void quantizeDequantizePerChannel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel, DTYPE* out,
                                  DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta, DTYPE* encodingOffset,
                                  ComputationMode modeCpuGpu, RoundingMode roundingMode, void* stream);


template <typename DTYPE>
void quantizeDequantizeBroadcastCpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElement, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides);


// GPU implementations ...
#ifdef GPU_QUANTIZATION_ENABLED

template <typename DTYPE>
void quantizeToFxpGpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned);

template <typename DTYPE>
void quantizeDequantizeGpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                           void* stream);

template <typename DTYPE>
void quantizeDequantizePerChannelGpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode, void* stream);

template <typename DTYPE>
void quantizeDequantizeBroadcastGpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElements, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides, void* stream);

#endif   // GPU_QUANTIZATION_ENABLED

}   // End of namespace DlQuantization

#endif   // UTIL_TRIM_FUNCTIONS_HPP_
