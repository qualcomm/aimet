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


#ifndef UTIL_MATH_FUNCTIONS_H_
#define UTIL_MATH_FUNCTIONS_H_

#include <algorithm>
#include <limits>
#include <map>
#include <math.h>
#include <stdexcept>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
/**
 * @brief A probability density distribution.
 *
 * We use this structure to record the dynamic range of a given number
 * distribution.
 */
struct PDF
{
    // The left sides of the buckets.
    std::vector<double> xLeft;
    // The probability for each bucket.
    std::vector<double> pdf;
    // This PDF holds the average data for this many iterations.
    int iterations;
};

// Profiling parameters of a tensor consisting in the global minimum and global
// maximum values and also the histogram obtained during profiling.
struct TensorProfilingParams
{
    double min;
    double max;
    std::vector<double> histogram;
    int iterations;
};

// Number of buckets in histogram which we use to capture the dynamic range.
const int PDF_SIZE = 512;

// The probability density functions for all input and output blobs of a network
// layer.
struct StatsLayerPdf
{
    std::vector<PDF> in;
    std::vector<PDF> out;
};

template <typename DTYPE>
DTYPE GetMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode);

template <typename DTYPE>
DTYPE GetMin(const DTYPE* data, int cnt, ComputationMode cpuGpuMode);

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> GetMinMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode);

// Android compiler doesn't have std::log2, so define it here
double logBase2(double d);

/**
 * @brief Create a probability density function and average it into the
 * data we have so far.
 * @param data The data to create probability densiti function from.
 * @param cnt The number of data points.
 * @param mode_cpu_gpu The 'data' buffer is either in CPU or GPU memory.
 * @param signed_vals If true, we create a histogram of the actual values. If
 * set to false, we create a histogram of the absolute values.
 * @param pdf Compute a probability density function. If this PDF already
 * contains values, update the PDF by averaging it with the new number
 * distribution.
 * @param allocator Device memory allocator. If nullptr, there is no device
 * memory allocator available.
 */
template <typename DTYPE>
void UpdatePdf(const DTYPE* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf,
               IAllocator* allocator);

/**
 * @brief Create a histogram of a given tensor
 * @param data The data to create histogram from.
 * @param cnt The number of data points.
 * @param bucket_size Size of each bucket
 * @param pdf_offset The minimum value that the leftmost bucket can cover
 * @param mode_cpu_gpu The 'data' buffer is either in CPU or GPU memory.
 * @param is_signed If true, we create a histogram of the actual values. If
 * set to false, we create a histogram of the absolute values.
 * @param allocator Device memory allocator. If nullptr, there is no device
 * memory allocator available.
 */
template <typename DTYPE>
void GetHistogram(const DTYPE* data, const int cnt, uint32_t histogram[PDF_SIZE], const DTYPE bucket_size,
                  const DTYPE pdf_offset, const ComputationMode mode_cpu_gpu, const bool is_signed,
                  IAllocator* allocator);

/**
 * @brief Allocate memory.
 * @param modeCpuGpu Allocate memory for CPU or GPU.
 * @param bytes The size of the memory in bytes.
 *
 * The memory is not initialized.
 */
void* MemoryAllocation(ComputationMode modeCpuGpu, size_t bytes);

/**
 * @brief Release previously allocated memory.
 * @param modeCpuGpu Allocate memory for CPU or GPU.
 * @param data Pointer to memory which was previously allocated with
 * MemoryAllocation().
 */
void MemoryFree(ComputationMode modeCpuGpu, void* data);

/**
 * @brief Multiply all inputs by a scale factor.
 * @param cnt The number of floating point values.
 * @param factor The numbers get scaled by this factor.
 */
void ElementwiseMult(ComputationMode modeCpuGpu, const float* in, size_t cnt, float factor, float* out);

/**
 * @brief Perform a matrix multiplication in floating point.
 */
void GemmFloat(ComputationMode modeCpuGpu, bool transposeB, size_t m, size_t n, size_t k, const float* A,
               const float* B, float* C);

// CPU implementations...
template <typename DTYPE>
DTYPE GetMax_cpu(const DTYPE* data, int cnt);

template <typename DTYPE>
DTYPE GetMin_cpu(const DTYPE* data, int cnt);

void ElementwiseMult_cpu(const float* in, size_t cnt, float factor, float* out);

void* MemoryAllocation_cpu(size_t bytes);

void MemoryFree_cpu(void* data);

template <typename DTYPE>
void GetHistogram_cpu(const DTYPE* data, int cnt, uint32_t histogram[PDF_SIZE], const DTYPE bucket_size,
                      const DTYPE pdf_offset, const bool is_signed);

/**
 * @brief Returns a histogram that represents a PDF of tensor values seen so far.
 * @param pdf Probability density function.
 */
std::vector<std::tuple<double, double>> getCollectedHistogram(const PDF& pdf);

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> findOriginalRange(const PDF& pdf);

/**
 * @brief Generate input tensor histogram based on the input tensor and existing
 * histogram.
 * @param data The number distribution for which we create a density function.
 * @param tensorSize The number of data points.
 * @param tpp histogram of float numbers seen so far.
 */
template <typename DTYPE>
void updateTensorHistogram(const DTYPE* data, int tensorSize, ComputationMode mode_cpu_gpu, TensorProfilingParams& tpp);

template <typename DTYPE>
void updateTensorHistogram_cpu(const DTYPE* data, int tensorSize, TensorProfilingParams& tpp);

/**
 * @brief Function to rescale the input histogram srcHist initially computed in the
 * range srcHistMin and srcHistMax to a new range given by destHistMin
 * and destHistMax. The rescaled histogram has the same number of bins as
 * the input histogram.
 */
std::vector<double> rescaleHistogram(const std::vector<double>& srcHist, const double srcHistMin,
                                     const double srcHistMax, const double destHistMin, const double destHistMax);


// GPU implementations...
#ifdef GPU_QUANTIZATION_ENABLED

template <typename DTYPE>
DTYPE GetMax_gpu(const DTYPE* data, int cnt);

template <typename DTYPE>
DTYPE GetMin_gpu(const DTYPE* data, int cnt);

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> GetMinMax_gpu(const DTYPE* data, int cnt);

void ElementwiseMult_gpu(const float* in, size_t cnt, float factor, float* out);

/**
 * @brief Perform matrix-matrix multiplication using cuBLAS.
 * @return True if CUDA call succeeds.
 */
bool GemmFloat_gpu(int M, int N, int K, const float* A, const float* B, float* C, bool transposeB);

void* MemoryAllocation_gpu(size_t bytes);

bool MemoryFree_gpu(void* data);

template <typename DTYPE>
void GetHistogram_gpu(const DTYPE* data, int cnt, uint32_t histogram[PDF_SIZE], const DTYPE bucket_size,
                      const DTYPE pdf_offset, const bool is_signed, IAllocator* allocator);

#endif   // GPU_QUANTIZATION_ENABLED

}   // End of namespace DlQuantization

#endif   // UTIL_MATH_FUNCTIONS_H_
