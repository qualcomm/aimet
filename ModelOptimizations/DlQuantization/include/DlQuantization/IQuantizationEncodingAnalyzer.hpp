//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef I_QUANTIZATION_ENCODING_ANALYZER_HPP
#define I_QUANTIZATION_ENCODING_ANALYZER_HPP

#include "Quantization.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace DlQuantization
{
template <typename DTYPE>
class IQuantizationEncodingAnalyzer
{
public:
    virtual ~IQuantizationEncodingAnalyzer() = default;

    /**
     * @brief Given a tensor update running stats for this encoding analyzer
     * @param tensor The tensor to use for updating stats
     * @param tensorSize Number of elements in the tensor
     * @param tensorCpuGpuMode Enum indicating whether the tensor is placed in CPU or GPU memory
     */
    virtual void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode) = 0;

    /**
     * @brief Given a tensor update running stats for this encoding analyzer
     * @param tensor The tensor to use for updating stats
     * @param tensorSize Number of elements in the tensor
     * @param tensorCpuGpuMode Enum indicating whether the tensor is placed in CPU or GPU memory
     * @param allocator Device memory allocator. If nullptr, there is no device memory allocator available.
     */
    virtual void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode,
                             IAllocator* allocator) = 0;

    /**
     * @brief Given a number distribution in CPU memory, compute the TensorFlow
     * encoding with the highest possible SQNR.
     *
     * To do so, we perform a grid search over different deltas and offsets.
     * This grid search optimizes the encoding to reduce the cost of quantization.
     * In this cost function, saturation errors are weighted higher than
     * quantization errors.
     *
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings (with a zero-point of absolute 0)
     * @param useStrictSymmetric If true, and if useSymmetricEncodings is true, calculate encodings exactly centered
     *                           around 0. E.g. if bw==8, then this results in quantized int values (-127:127). If this
     *                           is not set, then quantized int values would be (-128:127) to use the entire range.
     * @param useUnsignedSymmetric If true, and if useSymmetricEncodings is true, check if the entire statistics we have
     *                          collected are for +ve numbers. If yes, use quantized int values (0:255). This is a
     *                          special case, where we have double the resolution for the computed encodings while
     *                          still preserving the zero-point to be absolute 0.
     */
    virtual TfEncoding computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                       bool useUnsignedSymmetric) const = 0;

    /**
     * @brief Returns a histogram that represents a PDF of tensor values seen by this encoding analyzer so far
     *
     * @return Histogram of statistics. The histogram returned is a vector of buckets. Each bucket is a tuple of
     * two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    virtual std::vector<std::tuple<double, double>> getStatsHistogram() const = 0;

    /**
     * @brief Set the percentile value
     *
     * @param percentile Percentile value to be used while adjusting min and max
     */
    virtual void setPercentileValue(float percentile)
    {
        throw std::runtime_error("setPercentileValue is only applicable for percentile encoding analyzers");
    }

    /**
     * @brief Fecth the percentile value
     *
     * @return percentile value of the encoding analyzer.
     */
    virtual float getPercentileValue()
    {
        throw std::runtime_error("getPercentileValue is only applicable for percentile encoding analyzers");
    }
};


template <typename DTYPE>
class IBlockEncodingAnalyzer
{
public:
    virtual ~IBlockEncodingAnalyzer() = default;

    /**
     * @brief Given a tensor, update running stats for this encoding analyzer
     *
     * @param tensor The tensor to use for updating stats
     * @param tensorShape The shape of the input tensor
     * @param tensorCpuGpuMode Enum indicating whether the tensor is placed in CPU or GPU memory
     * @param allocator Memory allocator for the specified device
     * @param stream cuda stream to launch kernels on if applicable
     */
    virtual void updateStats(const DTYPE* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                             IAllocator* allocator = nullptr, void* stream = nullptr) = 0;

    // virtual void resetStats();

    /**
     * @brief Given the observed stats, compute a list of encodings with length determined by _shape
     *
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings (with a zero-point of absolute 0)
     * @param useStrictSymmetric If true, and if useSymmetricEncodings is true, calculate encodings exactly centered
     *                           around 0. E.g. if bw==8, then this results in quantized int values (-127:127). If this
     *                           is not set, then quantized int values would be (-128:127) to use the entire range.
     * @param useUnsignedSymmetric If true, and if useSymmetricEncodings is true, check if the entire statistics we have
     *                          collected are >= 0. If yes, use quantized int values (0:255). This is a
     *                          special case, where we have double the resolution for the computed encodings while
     *                          still preserving the zero-point to be absolute 0.
     */
    virtual std::vector<TfEncoding> computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                                    bool useUnsignedSymmetric) const = 0;

    /**
     * @brief Returns a list of histograms that each represents a PDF of tensor values seen by this encoding analyzer for
     * its associated block
     *
     * @return List of histograms of statistics. Each histogram returned is a vector of buckets. Each bucket is a tuple of
     * two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    virtual std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const = 0;

    /**
     * @brief Set the percentile value
     *
     * @param percentile Percentile value to be used while adjusting min and max
     */
    virtual void setPercentileValue(float percentile)
    {
        throw std::runtime_error("setPercentileValue is only applicable for percentile encoding analyzers");
    }

    /**
     * @brief Fetch the percentile value
     *
     * @return percentile value of the encoding analyzer.
     */
    virtual float getPercentileValue()
    {
        throw std::runtime_error("getPercentileValue is only applicable for percentile encoding analyzers");
    }

    virtual TensorDims getShape();
};


}   // namespace DlQuantization


#endif   // I_QUANTIZATION_ENCODING_ANALYZER_HPP
