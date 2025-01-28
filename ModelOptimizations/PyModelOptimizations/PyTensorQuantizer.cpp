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
#include "PyTensorQuantizer.hpp"

namespace DlQuantization
{

PyTensorQuantizer::PyTensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
    TensorQuantizer(quantScheme, roundingMode)
{
}

void PyTensorQuantizer::updateStats(py::array_t<float> tensor, bool useCuda)
{
    auto npArr = tensor.mutable_unchecked();

    size_t tensorSize = 1;
    for (int i = 0; i < npArr.ndim(); i++)
        tensorSize *= npArr.shape(i);

    // Get a pointer to the tensor data
    auto tensorPtr = (float*) npArr.mutable_data();

    // Delegate
    TensorQuantizer::updateStats(tensorPtr, tensorSize, useCuda);
}

void PyTensorQuantizer::quantizeDequantize(py::array_t<float> inputTensor, py::array_t<float> outputTensor,
                                           double encodingMin, double encodingMax, unsigned int bitwidth, bool useCuda)
{
    auto inputArr  = inputTensor.mutable_unchecked();
    auto outputArr = outputTensor.mutable_unchecked();

    size_t tensorSize = inputArr.size();

    auto inputTensorPtr  = static_cast<float*>(inputArr.mutable_data());
    auto outputTensorPtr = static_cast<float*>(outputArr.mutable_data());

    // Delegate
    TensorQuantizer::quantizeDequantize(inputTensorPtr, tensorSize, outputTensorPtr, encodingMin, encodingMax, bitwidth,
                                        useCuda);
}


void pyUpdateStats(BlockTensorQuantizer &self, py::array_t<float> tensor)
{
    py::buffer_info buf = tensor.request();
    TensorDims shape(buf.ndim);
    for (size_t i = 0; i < buf.ndim; i++)
    {
        shape[i] = buf.shape[i];
    }
    auto ptr = static_cast<float*>(buf.ptr);
    self.updateStats(ptr, shape, false);

}

}   // namespace DlQuantization
