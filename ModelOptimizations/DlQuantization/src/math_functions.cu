//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <limits>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>
#include <cstdint>

#include "cuda_util.hpp"
#include "math_functions.hpp"


namespace DlQuantization
{
template <typename DTYPE>
DTYPE GetMax_gpu(const DTYPE* data, uint64_t cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::lowest(), thrust::maximum<DTYPE>());
}

template <typename DTYPE>
DTYPE GetMin_gpu(const DTYPE* data, uint64_t cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::max(), thrust::minimum<DTYPE>());
}



struct _ApplyScale
{
    _ApplyScale(int scale) : scale(scale)
    {
    }

    __host__ __device__ float operator()(int x) const
    {
        return x * scale;
    }

private:
    int scale;
};


template <typename DTYPE>
std::tuple<std::vector<DTYPE>, std::vector<DTYPE>> GetMinMax_gpu(const DTYPE* data, uint64_t cnt, uint64_t blockSize,
                                                                 IAllocator* allocator, void* stream)
{
    auto computeStream         = static_cast<cudaStream_t>(stream);

    // Faster for per-tensor mode
    if (cnt == blockSize)
    {
        // Per-tensor reduce requires stream to be synchronized first
        cudaStreamSynchronize(computeStream);
        auto minMax = GetMinMax_gpu(data, cnt);
        return std::make_tuple<std::vector<DTYPE>, std::vector<DTYPE> >({std::get<0>(minMax)}, {std::get<1>(minMax)});
    }

    size_t numBlocks           = cnt / blockSize;
    void* dTempStorage         = nullptr;
    size_t tempStorageBytesMin = 0;
    size_t tempStorageBytesMax = 0;
    size_t memSize             = sizeof(DTYPE) * 2 * numBlocks;
    std::vector<DTYPE> minMaxOut(2 * numBlocks);
    DTYPE* dMinMaxOut;
    dMinMaxOut = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(memSize) : MemoryAllocation_gpu(memSize));

    auto offsetIterator = thrust::make_transform_iterator(thrust::make_counting_iterator(0), _ApplyScale(blockSize));

    // When dTempStorage is nullptr, this does not do any device computation, but sets tempStorageBytes to the size of
    // temporary storage necessary for computation.
    // Use the maximum storage needed for min and max calculations (these will likely be identical, this is just to be
    // safe)
    cub::DeviceSegmentedReduce::Min(dTempStorage, tempStorageBytesMin, data, dMinMaxOut, numBlocks, offsetIterator,
                                    offsetIterator + 1);
    cub::DeviceSegmentedReduce::Max(dTempStorage, tempStorageBytesMax, data, dMinMaxOut + numBlocks, numBlocks,
                                    offsetIterator, offsetIterator + 1);
    size_t tempStorageBytes = std::max(tempStorageBytesMin, tempStorageBytesMax);

    // Allocate the temporary device storage
    dTempStorage = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(tempStorageBytes)
                                                 : MemoryAllocation_gpu(tempStorageBytes));

    // Perform the actual min/max reductions
    cub::DeviceSegmentedReduce::Min(dTempStorage, tempStorageBytes, data, dMinMaxOut, numBlocks, offsetIterator,
                                    offsetIterator + 1, computeStream);
    cub::DeviceSegmentedReduce::Max(dTempStorage, tempStorageBytes, data, dMinMaxOut + numBlocks, numBlocks,
                                    offsetIterator, offsetIterator + 1, computeStream);

    // Transfer reduced min/max to CPU
    cudaStreamSynchronize(computeStream);
    cudaMemcpy(minMaxOut.data(), dMinMaxOut, 2 * sizeof(DTYPE) * numBlocks, cudaMemcpyDeviceToHost);

    std::vector<DTYPE> minOut(minMaxOut.begin(), minMaxOut.begin() + numBlocks);
    std::vector<DTYPE> maxOut(minMaxOut.begin() + numBlocks, minMaxOut.end());

    // Free allocated device memory
    allocator ? allocator->deleteRaw(dTempStorage) : (void) MemoryFree_gpu(dTempStorage);
    allocator ? allocator->deleteRaw(dMinMaxOut) : (void) MemoryFree_gpu(dMinMaxOut);

    return std::make_tuple(std::move(minOut), std::move(maxOut));
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> GetMinMax_gpu(const DTYPE* data, uint64_t cnt)
{
    DTYPE minMaxOut[2];
    DTYPE *dMinMaxOut;
    void *dTempStorage = nullptr;
    size_t tempStorageBytesMin = 0;
    size_t tempStorageBytesMax = 0;
    cudaMalloc(&dMinMaxOut, sizeof(DTYPE) * 2);

    // When dTempStorage is nullptr, this does not do any device computation, but sets tempStorageBytes to the size of
    // temporary storage necessary for computation.
    // Use the maximum storage needed for min and max calculations (these will likely be identical, this is just to be safe)
    cub::DeviceReduce::Min(dTempStorage, tempStorageBytesMin, data, dMinMaxOut, cnt);
    cub::DeviceReduce::Max(dTempStorage, tempStorageBytesMax, data, dMinMaxOut + 1, cnt);
    size_t tempStorageBytes = std::max(tempStorageBytesMin, tempStorageBytesMax);

    // Allocate the temporary device storage
    cudaMalloc(&dTempStorage, tempStorageBytes);

    // Perform the actual min/max reductions
    cub::DeviceReduce::Min(dTempStorage, tempStorageBytes, data, dMinMaxOut, cnt);
    cub::DeviceReduce::Max(dTempStorage, tempStorageBytes, data, dMinMaxOut + 1, cnt);

    // Transfer reduce min/max to CPU
    cudaMemcpy(minMaxOut, dMinMaxOut, 2 * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(dTempStorage);
    cudaFree(dMinMaxOut);
    return std::make_tuple(minMaxOut[0], minMaxOut[1]);
}

__global__ void ElementwiseMult_kernel(const float* in, size_t cnt, float factor, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        out[i] = in[i] * factor;
    }
}

void ElementwiseMult_gpu(const float* in, size_t cnt, float factor, float* out)
{
    ElementwiseMult_kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(in, cnt, factor, out);
}

void* MemoryAllocation_gpu(size_t bytes)
{
    void* devPtr;
    auto status = cudaMalloc(&devPtr, bytes);

    if (cudaErrorMemoryAllocation == status) {
        throw std::runtime_error("CUDA OOM");
    }

    if (cudaSuccess != status) {
        throw std::runtime_error("cuda malloc failed");
    }

    return devPtr;
}

bool MemoryFree_gpu(void* data)
{
    return cudaSuccess == cudaFree(data);
}

// Explicit instantiations
template double GetMax_gpu(const double* data, uint64_t cnt);

template float GetMax_gpu(const float* data, uint64_t cnt);

template double GetMin_gpu(const double* data, uint64_t cnt);

template float GetMin_gpu(const float* data, uint64_t cnt);

template std::tuple<std::vector<double>, std::vector<double>> GetMinMax_gpu(const double* data, uint64_t cnt, uint64_t blockSize,
                                                                            IAllocator* allocator, void* stream);

template std::tuple<std::vector<float>, std::vector<float>> GetMinMax_gpu(const float* data, uint64_t cnt, uint64_t blockSize,
                                                                          IAllocator* allocator, void* stream);

template std::tuple<float, float> GetMinMax_gpu(const float* data, uint64_t cnt);

template std::tuple<double, double> GetMinMax_gpu(const double* data, uint64_t cnt);

template <typename DTYPE>
__global__ static void histogramCountKernel(const DTYPE* data,
                                            uint32_t* histogram_per_thread,
                                            const size_t cnt,
                                            const DTYPE bucket_size,
                                            const DTYPE histogram_offset,
                                            const bool is_signed)
{
    // This offset is used to help map numbers to histogram buckets.
    // Go through all data points and add them to the histogram.
    CUDA_KERNEL_LOOP(i, cnt)
    {
        // Map a floating point number to the appropriate bucket.
        int index = is_signed ?
                    floor(data[i] / bucket_size - histogram_offset) :
                    floor(abs(data[i]) / bucket_size - histogram_offset);

        // Add to histogram, if inside the histogram range.
        if (index >= 0 && index < PDF_SIZE)
        {
            int idx = PDF_SIZE * (blockIdx.x * blockDim.x + threadIdx.x) + index;
            histogram_per_thread[idx] += 1;
        }
    }
}


__global__ static void histogramReduceSumKernel(const uint32_t* histogram_per_thread,
                                                uint32_t* histogram,
                                                const size_t cnt)
{
    if (blockIdx.x == 0 && threadIdx.x < PDF_SIZE)
    {
        for (int i = threadIdx.x; i < cnt; i += PDF_SIZE)
        {
            histogram[threadIdx.x] += histogram_per_thread[i];
        }
    }
}


static const int PDF_MAX_BUFF_BYTES = (1 << 25); // 32MB

#define GET_PDF_BUFF_SIZE(tensor_size, DTYPE)\
    sizeof(DTYPE) * CUDA_NUM_BLOCKS(tensor_size) * CUDA_NUM_THREADS * PDF_SIZE < PDF_MAX_BUFF_BYTES ?\
    CUDA_NUM_BLOCKS(tensor_size) :\
    PDF_MAX_BUFF_BYTES / (sizeof(DTYPE) * PDF_SIZE * CUDA_NUM_THREADS)


template <typename DTYPE>
void GetHistogram_gpu(const DTYPE* data,
                      uint64_t cnt,
                      uint32_t histogram[PDF_SIZE],
                      const DTYPE bucket_size,
                      const DTYPE pdf_offset,
                      const bool is_signed,
                      IAllocator* allocator)
{
    // Limit the number of thread blocks for performance based on heuristics
    const size_t CUDA_NUM_BLOCKS_ = GET_PDF_BUFF_SIZE(cnt, DTYPE);
    const size_t buff_size = PDF_SIZE * CUDA_NUM_BLOCKS_ * CUDA_NUM_THREADS;

    uint32_t* histogram_per_thread = (uint32_t*) allocator->allocateRaw(sizeof(uint32_t) * buff_size);

    cudaMemset(histogram_per_thread, 0x00, sizeof(uint32_t) * buff_size);

    // Go through all data points and add them to the histogram.
    histogramCountKernel<<<CUDA_NUM_BLOCKS_, CUDA_NUM_THREADS>>>(data,
                                                                 histogram_per_thread,
                                                                 cnt,
                                                                 bucket_size,
                                                                 pdf_offset,
                                                                 is_signed);

    uint32_t* histogram_gpu = (uint32_t*) allocator->allocateRaw(sizeof(uint32_t) * PDF_SIZE);
    cudaMemset(histogram_gpu, 0x00, sizeof(uint32_t) * PDF_SIZE);

    histogramReduceSumKernel<<<1, PDF_SIZE>>>(histogram_per_thread, histogram_gpu, buff_size);

    cudaMemcpy(histogram,
               histogram_gpu,
               sizeof(uint32_t) * PDF_SIZE,
               cudaMemcpyDefault);

    allocator->deleteRaw(histogram_gpu);
    allocator->deleteRaw(histogram_per_thread);
}

template void GetHistogram_gpu(const float* data,
                               uint64_t cnt,
                               uint32_t histogram[PDF_SIZE],
                               const float bucket_size,
                               const float pdf_offset,
                               const bool is_signed,
                               IAllocator* allocator);
template void GetHistogram_gpu(const double* data,
                               uint64_t cnt,
                               uint32_t histogram[PDF_SIZE],
                               const double bucket_size,
                               const double pdf_offset,
                               const bool is_signed,
                               IAllocator* allocator);

}   // End of namespace DlQuantization
