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


#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <stdlib.h>
#include <algorithm>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#ifdef GPU_QUANTIZATION_ENABLED

#include "cuda_util.hpp"

#endif

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
DTYPE GetMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode)
{
    switch (cpuGpuMode)
    {
    case COMP_MODE_CPU:
        return GetMax_cpu(data, cnt);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        return GetMax_gpu(data, cnt);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        return 0;
        break;
    }
}

template <typename DTYPE>
DTYPE GetMin(const DTYPE* data, int cnt, ComputationMode cpuGpuMode)
{
    switch (cpuGpuMode)
    {
    case COMP_MODE_CPU:
        return GetMin_cpu(data, cnt);
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        return GetMin_gpu(data, cnt);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    default:
        throw runtime_error("Unknown computation mode.");
        return 0;
    }
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> GetMinMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode)
{
    switch (cpuGpuMode)
    {
    case COMP_MODE_CPU:
        return std::make_tuple(GetMin_cpu(data, cnt), GetMax_cpu(data, cnt));
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        return GetMinMax_gpu(data, cnt);
#else
            throw runtime_error("Not compiled for GPU mode.");
#endif
    default:
        throw runtime_error("Unknown computation mode.");
    }
}

double logBase2(double d)
{
    return log(d) / log(2);
}

void* MemoryAllocation(ComputationMode modeCpuGpu, size_t bytes)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        return MemoryAllocation_cpu(bytes);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        return MemoryAllocation_gpu(bytes);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void MemoryFree(ComputationMode modeCpuGpu, void* data)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        MemoryFree_cpu(data);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        MemoryFree_gpu(data);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void ElementwiseMult(ComputationMode modeCpuGpu, const float* in, size_t cnt, float factor, float* out)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        ElementwiseMult_cpu(in, cnt, factor, out);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        ElementwiseMult_gpu(in, cnt, factor, out);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void GemmFloat(ComputationMode modeCpuGpu, bool transposeB, size_t m, size_t n, size_t k, const float* A,
               const float* B, float* C)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        (void) modeCpuGpu;
        (void) transposeB;
        (void) m;
        (void) n;
        (void) k;
        (void) A;
        (void) B;
        (void) C;
        throw runtime_error("CPU mode not implemented yet.");
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        bool result = GemmFloat_gpu(m, n, k, A, B, C, transposeB);
        if (!result)
        {
            throw runtime_error("CUDA GEMM failed.");
        }
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

template <typename DTYPE>
void InitializePdf(PDF& pdf, DTYPE min_val, DTYPE max_val, bool signed_vals)
{
    // Make sure we have a non-zero range.
    if (min_val == max_val)
    {
        max_val = std::max(max_val, min_val + (DTYPE) 0.01);
    }
    // Enlarge the range by factor 3, to be on the safe side.
    DTYPE center = (max_val + min_val) / 2;
    min_val      = std::max(std::numeric_limits<DTYPE>::lowest(), center - 3 * (center - min_val));
    max_val      = std::min(std::numeric_limits<DTYPE>::max(), center + 3 * (max_val - center));
    // Initialize the PDF's buckets.
    double bucket_size;
    if (signed_vals)
    {
        bucket_size = (static_cast<double>(max_val) - static_cast<double>(min_val)) / PDF_SIZE;
    }
    else
    {
        DTYPE max_abs_val = std::max(std::abs(max_val), std::abs(min_val));
        bucket_size       = max_abs_val / PDF_SIZE;
    }
    pdf.xLeft.resize(PDF_SIZE);
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        if (signed_vals)
            pdf.xLeft[i] = floor(min_val / bucket_size + i) * bucket_size;
        else
            pdf.xLeft[i] = i * bucket_size;
    }
    // Initialize the rest of the PDF structure.
    pdf.pdf.resize(PDF_SIZE);
    pdf.iterations = 0;
}

template <typename DTYPE>
void UpdatePdf(const DTYPE* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf,
               IAllocator* allocator)
{
    // Check if we need to initialize the PDF.
    if (0 == pdf.xLeft.size())
    {
        // Define the range over which we want to calculate the PDF.
        DTYPE min_val = GetMin(data, cnt, mode_cpu_gpu);
        DTYPE max_val = GetMax(data, cnt, mode_cpu_gpu);

        if ((min_val == 0) && (max_val == 0))
        {
            // Special case, we don't have a histogram initialized, but we have a zero tensor here
            // No point in trying to initialize the histogram using this
            return;
        }

        InitializePdf(pdf, min_val, max_val, signed_vals);
    }

    // The histogram's range is min_val to max_val.
    DTYPE bucket_size = pdf.xLeft[1] - pdf.xLeft[0];
    DTYPE min_val     = signed_vals ? pdf.xLeft[0] : 0;
    // This offset is used to help map numbers to histogram buckets.
    DTYPE pdf_offset = min_val / bucket_size;

    // Create the histogram of this number distribution.
    uint32_t histogram[PDF_SIZE];
    for (int i = 0; i < PDF_SIZE; i++)
    {
        histogram[i] = 0;
    }

    GetHistogram(data, cnt, histogram, bucket_size, pdf_offset, mode_cpu_gpu, signed_vals, allocator);

    // Average this histogram into the average of all batches.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        // Convert histogram to probability density function.
        double prob = static_cast<double>(histogram[i]) / static_cast<double>(cnt);
        // Average this PDF into the running average.
        pdf.pdf[i] = (pdf.pdf[i] * pdf.iterations + prob) / (pdf.iterations + 1);
    }
    pdf.iterations++;
}

template <typename DTYPE>
void GetHistogram(const DTYPE* data, int cnt, uint32_t histogram[PDF_SIZE], const DTYPE bucket_size,
                  const DTYPE pdf_offset, const ComputationMode mode_cpu_gpu, const bool is_signed,
                  IAllocator* allocator)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        GetHistogram_cpu(data, cnt, histogram, bucket_size, pdf_offset, is_signed);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        if (allocator)
        {
            GetHistogram_gpu(data, cnt, histogram, bucket_size, pdf_offset, is_signed, allocator);
        }
        else
        {
            // Fall back to CPU mode
            DTYPE* data_h = (DTYPE*) malloc(sizeof(DTYPE) * cnt);
            CudaMemCpy(data_h, data, cnt * sizeof(DTYPE), CudaMemcpyDirection::DEVICE_TO_HOST);
            GetHistogram_cpu(data_h, cnt, histogram, bucket_size, pdf_offset, is_signed);
            free(data_h);
        }
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}


// CPU mode implementations

template <typename DTYPE>
DTYPE GetMax_cpu(const DTYPE* data, int cnt)
{
    DTYPE val = -numeric_limits<double>::max();
    for (int i = 0; i < cnt; ++i)
    {
        val = max(val, data[i]);
    }
    return val;
}

template <typename DTYPE>
DTYPE GetMin_cpu(const DTYPE* data, int cnt)
{
    DTYPE val = numeric_limits<double>::max();
    for (int i = 0; i < cnt; ++i)
    {
        val = min(val, data[i]);
    }
    return val;
}

void ElementwiseMult_cpu(const float* in, size_t cnt, float factor, float* out)
{
    for (unsigned int i = 0; i < cnt; ++i)
    {
        out[i] = in[i] * factor;
    }
}

void* MemoryAllocation_cpu(size_t bytes)
{
    return malloc(bytes);
}

void MemoryFree_cpu(void* data)
{
    free(data);
}

template <typename DTYPE>
void GetHistogram_cpu(const DTYPE* data, int cnt, uint32_t histogram[PDF_SIZE], const DTYPE bucket_size,
                      const DTYPE pdf_offset, const bool is_signed)
{
    // Go through all data points and add them to the histogram.
    for (int i = 0; i < cnt; ++i)
    {
        // Map a floating point number to the appropriate bucket.
        int index =
            is_signed ? floor(data[i] / bucket_size - pdf_offset) : floor(std::abs(data[i]) / bucket_size - pdf_offset);

        // Add to histogram, if inside the histogram range.
        if (index >= 0 && index < PDF_SIZE)
        {
            histogram[index] += 1;
        }
    }
}

std::vector<std::tuple<double, double>> getCollectedHistogram(const PDF& pdf)
{
    // Allocate a vector to hold tuples of left edges and pdf for each bucket
    std::vector<std::tuple<double, double>> histogram;
    histogram.reserve(pdf.xLeft.size());

    // Assert that the stats structure is well formed
    assert(pdf.xLeft.size() == pdf.pdf.size());

    unsigned index = 0;
    for (auto entry: pdf.xLeft)
    {
        histogram.push_back(std::make_tuple(entry, pdf.pdf[index]));
        index++;
    }
    return histogram;
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> findOriginalRange(const PDF& pdf)
{
    DTYPE minVal = pdf.xLeft[0];
    DTYPE maxVal = pdf.xLeft[PDF_SIZE - 1];

    // To do so we search for the smallest and largest value from the pdf
    // Search for the lowest bucket which has probability > 0.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        if (pdf.pdf[i] > 0)
        {
            minVal = pdf.xLeft[i];
            break;
        }
    }

    // Search for the highest bucket which has probability > 0.
    for (int i = PDF_SIZE - 1; i > 0; --i)
    {
        if (pdf.pdf[i] > 0)
        {
            maxVal = pdf.xLeft[i];
            break;
        }
    }

    // Make sure we include zero in range.
    minVal = std::min(minVal, (DTYPE) 0);
    maxVal = std::max(maxVal, (DTYPE) 0);

    // Make sure we have a real range.
    maxVal = std::max(maxVal, minVal + (DTYPE) 0.01);

    return std::tuple<DTYPE, DTYPE>(minVal, maxVal);
}

template <typename DTYPE>
void updateTensorHistogram(const DTYPE* data, int tensorSize, ComputationMode mode_cpu_gpu, TensorProfilingParams& tpp)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        updateTensorHistogram_cpu(data, tensorSize, tpp);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        // Fall back to CPU mode.
        DTYPE* data_h = (DTYPE*) malloc(sizeof(DTYPE) * tensorSize);
        CudaMemCpy(data_h, data, tensorSize * sizeof(DTYPE), CudaMemcpyDirection::DEVICE_TO_HOST);
        updateTensorHistogram_cpu(data_h, tensorSize, tpp);
        free(data_h);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

/// Gen a bin number to insert \p value into the histogram which has \p nBins
/// with \p minValue and binWidth in histogram.
static size_t getBin(size_t nBins, float binWidth, float minValue, float value)
{
    size_t result = binWidth == 0 ? 0 : std::min(static_cast<size_t>((value - minValue) / binWidth), nBins - 1);
    return result;
}

template <typename DTYPE>
void updateTensorHistogram_cpu(const DTYPE* data, int tensorSize, TensorProfilingParams& tpp)
{
    double minInput = GetMin(data, tensorSize, COMP_MODE_CPU);
    double maxInput = GetMax(data, tensorSize, COMP_MODE_CPU);

    if ((minInput == 0) && (maxInput == 0))
    {
        // Special case, we don't have a histogram initialized, but we have a zero tensor here
        // No point in trying to initialize the histogram using this
        return;
    }

    // Make sure we have a non-zero range.
    if (minInput == maxInput)
    {
        maxInput = std::max(maxInput, minInput + (DTYPE) 0.01);
    }

    // Check if we need to initialize the Histogram
    if (0 == tpp.histogram.size())
    {
        tpp.histogram = std::vector<double>(PDF_SIZE, 0);
        tpp.min       = minInput;
        tpp.max       = maxInput;
    }

    // Check if we need to rescale histogram.
    if (minInput < tpp.min || maxInput > tpp.max)
    {
        double newMin = std::min(minInput, tpp.min);
        double newMax = std::max(maxInput, tpp.max);

        double destBinWidth = (static_cast<double>(newMax) - newMin) / PDF_SIZE;
        double srcBinWidth  = (static_cast<double>(tpp.max) - tpp.min) / PDF_SIZE;

        std::vector<double> scaledHistogram(PDF_SIZE, 0);

        for (size_t i = 0; i < PDF_SIZE; ++i)
        {
            if (tpp.histogram[i] == 0)
                continue;

            double srcBinBegin = tpp.min + srcBinWidth * i;
            size_t destBin     = (srcBinBegin - newMin) / destBinWidth;
            double destBinEnd  = newMin + destBinWidth * (destBin + 1);

            double srcBinEnd = srcBinBegin + srcBinWidth;

            // Calculate how much we need to redistribute.
            double dstBinCnt =
                std::min(static_cast<double>(round((destBinEnd - srcBinBegin) / srcBinWidth * tpp.histogram[i])),
                         tpp.histogram[i]);

            size_t newBin = getBin(PDF_SIZE, destBinWidth, newMin, srcBinBegin);
            scaledHistogram[newBin] += dstBinCnt;

            if (dstBinCnt < tpp.histogram[i])
            {
                size_t newBin = getBin(PDF_SIZE, destBinWidth, newMin, srcBinBegin + destBinWidth);
                scaledHistogram[newBin] += tpp.histogram[i] - dstBinCnt;
            }
        }

        // Copy scaled histogram back to the existing histogram.
        for (size_t i = 0, e = scaledHistogram.size(); i < e; ++i)
        {
            assert(scaledHistogram[i] >= 0 && "Invalid rescaled histogram value, it must be non-negative.");
            tpp.histogram[i] = scaledHistogram[i];
        }

        // Update global min and max.
        tpp.min = newMin;
        tpp.max = newMax;
    }

    float binWidth = (tpp.max - tpp.min) / PDF_SIZE;
    // Go through all data points and add them to the histogram.
    for (int i = 0; i < tensorSize; ++i)
    {
        size_t newBin = getBin(PDF_SIZE, binWidth, tpp.min, data[i]);
        tpp.histogram[newBin] += 1;
    }
    tpp.iterations++;
}

std::vector<double> rescaleHistogram(const std::vector<double>& srcHist, const double srcHistMin,
                                     const double srcHistMax, const double destHistMin, const double destHistMax)
{
    // If histogram is empty then return.
    if (srcHist.size() == 0)
    {
        return srcHist;
    }

    // Check if we need to rescale the histogram.
    assert(srcHistMin < srcHistMax && "Invalid source histogram min/max range!");
    assert(destHistMin < destHistMax && "Invalid destination histogram min/max range!");
    if ((srcHistMin == destHistMin) && (srcHistMax == destHistMax))
    {
        return srcHist;
    }

    // Number of histogram bins and bin widths.
    const size_t numBins      = srcHist.size();
    const double srcBinWidth  = (static_cast<double>(srcHistMax) - srcHistMin) / numBins;
    const double destBinWidth = (static_cast<double>(destHistMax) - destHistMin) / numBins;

    // Iterate the source bins and distribute into the destination bins.
    std::vector<double> destHist(numBins, 0);
    for (size_t srcBinIdx = 0; srcBinIdx < numBins; srcBinIdx++)
    {
        // Get current source bin value.
        double srcBinVal = srcHist[srcBinIdx];
        if (srcBinVal == 0)
        {
            continue;
        }

        // Get source bin start/stop values for this bin.
        double srcBinStart = srcHistMin + srcBinIdx * srcBinWidth;
        double srcBinStop  = srcHistMin + (srcBinIdx + 1) * srcBinWidth;

        // Get destination bin indices (inclusive) which overlap with the current
        // source bin.
        double dstBinIdxStartF = std::floor((srcBinStart - destHistMin) / destBinWidth);
        double dstBinIdxStopF  = std::ceil((srcBinStop - destHistMin) / destBinWidth);
        size_t dstBinIdxStart  = static_cast<size_t>(std::max(dstBinIdxStartF, 0.0));
        size_t dstBinIdxStop   = static_cast<size_t>(std::max(dstBinIdxStopF, 0.0));

        // Upper saturate the destination bin indices.
        if (dstBinIdxStart >= numBins)
        {
            dstBinIdxStart = numBins - 1;
        }
        if (dstBinIdxStop >= numBins)
        {
            dstBinIdxStop = numBins - 1;
        }

        // Redistribute the source bin into all the destination bins.
        // Only integer values will be distributed.
        double srcBinRem = srcBinVal;
        for (size_t destBinIdx = dstBinIdxStart; destBinIdx <= dstBinIdxStop; destBinIdx++)
        {
            // Get destination bin start/stop values for this bin.
            double destBinStart = destHistMin + destBinIdx * destBinWidth;
            double destBinStop  = destHistMin + (destBinIdx + 1) * destBinWidth;

            // Get source/destination overlap boundaries and ratio.
            double overlapStart = std::max(srcBinStart, destBinStart);
            double overlapStop  = std::min(srcBinStop, destBinStop);
            double overlapRatio = (overlapStop - overlapStart) / srcBinWidth;
            overlapRatio        = overlapRatio >= 0.0f ? overlapRatio : 0.0f;
            overlapRatio        = overlapRatio <= 1.0f ? overlapRatio : 1.0f;

            // Compute distribution value.
            double distVal = std::round(overlapRatio * srcBinVal);
            distVal        = distVal <= srcBinRem ? distVal : srcBinRem;

            // Distribute value.
            destHist[destBinIdx] += distVal;
            srcBinRem -= distVal;
        }
    }

    return destHist;
}

// Explicit instantiations
template double GetMax(const double* data, int cnt, ComputationMode mode_cpu_gpu);

template float GetMax(const float* data, int cnt, ComputationMode mode_cpu_gpu);

template double GetMin(const double* data, int cnt, ComputationMode mode_cpu_gpu);

template float GetMin(const float* data, int cnt, ComputationMode mode_cpu_gpu);

template std::tuple<float, float> GetMinMax(const float* data, int cnt, ComputationMode cpuGpuMode);

template std::tuple<double, double> GetMinMax(const double* data, int cnt, ComputationMode cpuGpuMode);

template void UpdatePdf(const double* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf,
                        IAllocator* allocator);

template void UpdatePdf(const float* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf,
                        IAllocator* allocator);

template std::tuple<double, double> findOriginalRange(const PDF& pdf);

template std::tuple<float, float> findOriginalRange(const PDF& pdf);

template void updateTensorHistogram(const double* data, int tensorSize, ComputationMode mode_cpu_gpu,
                                    TensorProfilingParams& tpp);

template void updateTensorHistogram(const float* data, int tensorSize, ComputationMode mode_cpu_gpu,
                                    TensorProfilingParams& tpp);

}   // End of namespace DlQuantization
