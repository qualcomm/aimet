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


#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP
#include <cstddef>
#include <cstdint>
#include <vector>

namespace DlQuantization
{

/**
 * Run on the CPU or GPU.
 */
enum ComputationMode
{
    COMP_MODE_CPU,
    COMP_MODE_GPU
};


/**
 * @brief Device memory allocator interface.
 */
class IAllocator
{
public:
    /**
     * @brief Allocate memory to the associated device and return the pointer.
     * @param bytes Bytes to allocate
     */
    virtual void* allocateRaw(size_t bytes) = 0;

    /**
     * @brief Deallocate the memory occupied by the pointer.
     * @param ptr Pointer to deallocate.
     */
    virtual void deleteRaw(void* ptr) = 0;
};


/**
 * @brief The fixed point quantization mode defines what fixed point scheme
 * we use, and how we compute a suitable encoding.
 */
enum QuantizationMode
{
    // TensorFlow quantization with static encoding; zero can be represented.
    QUANTIZATION_TF,

    // An enhanced version of TensorFlow. The encoding is chosen such that the SQNR is maximal.
    QUANTIZATION_TF_ENHANCED,

    // Ranges (min, max) are learnt during training
    QUANTIZATION_RANGE_LEARNING,

    // Percentile calibration. Compute the encoding by adjusting the min/max range of the tensor
    // by clipping percentile of outliers.
    QUANTIZATION_PERCENTILE,

    // Compute the encoding by adjusting the min/max range of the tensor based on the mean square
    // error due to quantization of the tensor.
    QUANTIZATION_MSE,

    // Compute the optimal quantization range (thresholds) for a tensor based on minimizing the
    // Kullback-Leibler divergence.
    QUANTIZATION_ENTROPY,
};

/**
 * @brief TensorFlow-style fixed point format.
 *
 * We use this fixed point format for both TensorFlow quantization and Q-format
 * quantization.
 */
struct TfEncoding
{
    double min;
    double max;
    double delta;
    double offset;
    int bw;
};

/**
 * @brief The fixed point format of activations in a given network layer.
 */
struct TfEncodingLayer
{
    // The encoding of input tensors.
    std::vector<TfEncoding> in;
    // The encoding of output tensors.
    std::vector<TfEncoding> out;
};

/**
 * @brief Layer input or output activations.
 */
enum LayerInOut
{
    LAYER_INPUT,
    LAYER_OUTPUT
};

/**
 * @brief Rounding mode.
 */
enum RoundingMode
{
    ROUND_NEAREST,
    ROUND_STOCHASTIC
};

using TensorDim = int64_t;
using TensorDims = std::vector<TensorDim>;
using Encodings = std::vector<TfEncoding>;

/**
 * @brief Performs quantize-dequantize using numpy-style broadcasting between input and encodings
 *
 * @tparam T Floating point data type of input/output/encodings
 * @param inTensor Pointer to input data
 * @param outTensor Pointer to output data, must be on same device as input
 * @param encodings Vector of TfEncoding pointers
 * @param inputShape Shape of input tensor data
 * @param encodingShape Shape of encoding vector
 * @param mode CPU/GPU computation mode
 * @param stream Pointer cuda stream to use for computation
 */
template <typename T>
void quantizeDequantizeBroadcast(const T* inTensor,
                                 T* outTensor,
                                 const Encodings& encodings,
                                 const TensorDims& inputShape,
                                 const TensorDims& encodingShape,
                                 ComputationMode mode,
                                 void* stream = nullptr);


}   // End of namespace DlQuantization.

#endif   // QUANTIZATION_HPP
