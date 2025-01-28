//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
#ifndef AIMET_PY_TENSOR_QUANTIZER_H
#define AIMET_PY_TENSOR_QUANTIZER_H

#include <memory>
#include <pybind11/numpy.h>

#include "DlQuantization/TensorQuantizer.h"

namespace py = pybind11;

namespace DlQuantization
{
/**
 * This class sublasses a tensor quantizer and overloads two of its functions with pybind
 * alternatives
 */
class PyTensorQuantizer : public TensorQuantizer
{
public:
    /**
     * Constructor
     * @param quantScheme Quantization scheme (e.g. TF-Enhanced)
     * @param roundingMode Rounding mode to use during quantization
     */
    PyTensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode);

    /**
     * Update stats being collected to compute encoding. Overloaded version that accepts a numpy tensor.
     * @param tensor Tensor to update the stats with
     * @param useCuda If true, the tensor is assumed to be in CUDA memory
     */
    void updateStats(py::array_t<float> tensor, bool useCuda);

    /**
     * Convert a tensor from float to quantized int and back to float. Overloaded version that accepts numpy tensors.
     * @param input Input tensor
     * @param output Output tensor
     * @param encodingMin minimum value of encoding range
     * @param encodingMax maximum value of encoding range
     * @param bitwidth bitwidth to be used
     * @param useCuda If true, both the input and output tensors are assumed to be in CUDA memory
     */
    void quantizeDequantize(py::array_t<float> inputTensor, py::array_t<float> outputTensor, double encodingMin,
                            double encodingMax, unsigned int bitwidth, bool useCuda);

    ~PyTensorQuantizer() = default;
};

void pyUpdateStats(BlockTensorQuantizer &self, py::array_t<float> tensor);

}   // namespace DlQuantization

#endif   // AIMET_PY_TENSOR_QUANTIZER_H
