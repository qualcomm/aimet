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

#ifndef TEST_QUANTIZATION_LIB_HPP
#define TEST_QUANTIZATION_LIB_HPP

#include "gtest/gtest.h"

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#ifdef GPU_QUANTIZATION_ENABLED

#include "cuda_util.hpp"

#endif   // GPU_QUANTIZATION_ENABLED

// Definition of test template
typedef ::testing::Types<float, double> TestDtypes;

template <typename DataType>
struct CpuDevice
{
    typedef DataType dataType;
    static const DlQuantization::ComputationMode modeCpuGpu = DlQuantization::COMP_MODE_CPU;
};

#ifndef GPU_QUANTIZATION_ENABLED

typedef ::testing::Types<CpuDevice<float>, CpuDevice<double> > TestDataTypesAndDevices;

typedef ::testing::Types<CpuDevice<float> > TestDeviceTypes;

#else

template <typename DataType>
struct GpuDevice
{
    typedef DataType dataType;
    static const DlQuantization::ComputationMode modeCpuGpu = DlQuantization::COMP_MODE_GPU;
};

typedef ::testing::Types<CpuDevice<float>, CpuDevice<double>, GpuDevice<float>, GpuDevice<double> >
    TestDataTypesAndDevices;

typedef ::testing::Types<CpuDevice<float>, GpuDevice<float> >
    TestDeviceTypes;

#endif

/**
 * @brief This method should be called at runtime by gtests that require a GPU
 * with CUDA support.
 */
template <typename DataType>
bool CheckRunTest()
{
#ifndef GPU_QUANTIZATION_ENABLED
    return true;
#else
    if (DlQuantization::COMP_MODE_CPU == DataType::modeCpuGpu)
    {
        return true;
    }
    else
    {
        return DlQuantization::CudaSupportedHelper();
    }
#endif
}

// Move data between CPU and GPU memory
template <typename TestParams>
class Blob
{
    typedef typename TestParams::dataType DataType;

public:
    // Blob is initialized with data from CPU
    Blob(DataType* cpuDataPtr, uint64_t cnt)
    {
        _cpuDataPtr = (DataType*) malloc(cnt * sizeof(DataType));
        memcpy(_cpuDataPtr, cpuDataPtr, cnt * sizeof(DataType));
        _count       = cnt;
        _gpuDataPtr  = nullptr;
        _isDataOnGpu = false;
#ifdef GPU_QUANTIZATION_ENABLED
        if (DlQuantization::COMP_MODE_GPU == TestParams::modeCpuGpu)
        {
            _gpuDataPtr =
                (DataType*) DlQuantization::MemoryAllocation(DlQuantization::COMP_MODE_GPU, _count * sizeof(DataType));
        }
#endif
    }

    // Free all memory
    ~Blob()
    {
        free(_cpuDataPtr);
#ifdef GPU_QUANTIZATION_ENABLED
        if (DlQuantization::COMP_MODE_GPU == TestParams::modeCpuGpu)
        {
            DlQuantization::MemoryFree(DlQuantization::COMP_MODE_GPU, _gpuDataPtr);
        }
#endif
    }

    // Move data to CPU memory, is necessary, and return data pointer.
    DataType* getDataPtrOnCpu()
    {
#ifdef GPU_QUANTIZATION_ENABLED
        if (_isDataOnGpu)
        {
            DlQuantization::CudaMemCpy(_cpuDataPtr, _gpuDataPtr, _count * sizeof(DataType),
                                       DlQuantization::CudaMemcpyDirection::DEVICE_TO_HOST);
            _isDataOnGpu = false;
        }
#endif
        return _cpuDataPtr;
    }

    // Move data to test device, and return data pointer.
    // In CPU mode, the CPU is test device. In GPU mode, the GPU is test device.
    DataType* getDataPtrOnDevice()
    {
#ifdef GPU_QUANTIZATION_ENABLED
        // Case device == CPU
        if (DlQuantization::COMP_MODE_CPU == TestParams::modeCpuGpu)
        {
            return _cpuDataPtr;
            // Case device == GPU
        }
        else
        {
            if (!_isDataOnGpu)
            {
                DlQuantization::CudaMemCpy(_gpuDataPtr, _cpuDataPtr, _count * sizeof(DataType),
                                           DlQuantization::CudaMemcpyDirection::HOST_TO_DEVICE);
                memset(_cpuDataPtr, 0, _count * sizeof(DataType));
                _isDataOnGpu = true;
            }
            return _gpuDataPtr;
        }
#else
        return _cpuDataPtr;
#endif
    }

private:
    DataType* _cpuDataPtr;
    DataType* _gpuDataPtr;
    // The number of data points to hold in memory.
    int _count;
    bool _isDataOnGpu;
};

using namespace DlQuantization;

// Assume a non-skewed distribution.
// Round offset to fixed point, and adjust min and max accordingly.
static TfEncoding getTfEncoding(double min, double max, int bw)
{
    TfEncoding encoding;
    int steps       = pow(2, bw) - 1;
    encoding.delta  = (max - min) / (double) steps;
    encoding.offset = round(min / encoding.delta);
    encoding.min    = encoding.delta * encoding.offset;
    encoding.max    = encoding.delta * steps + encoding.min;
    encoding.bw     = bw;
    return encoding;
}

static TfEncoding getTfSymmetricEncoding(double max, int bw)
{
    TfEncoding encoding;
    int halfSteps                 = pow(2, bw) - 2;   // To make it symmetric
    unsigned int numPositiveSteps = std::floor(halfSteps / 2);
    encoding.delta                = max / numPositiveSteps;
    encoding.offset               = -std::ceil(halfSteps / 2);
    encoding.min                  = encoding.offset * encoding.delta;
    encoding.max                  = encoding.delta * numPositiveSteps;
    encoding.bw                   = bw;
    return encoding;
}

static bool compareEncodings(TfEncoding e0, TfEncoding e1)
{
    bool result = true;

    // Hacky to use the gtest internal function of testing double value equaity
#define TEST_DOUBLE_EQ(v0, v1) bool(::testing::internal::CmpHelperFloatingPointEQ<double>(#v0, #v1, v0, v1))

    result = result && TEST_DOUBLE_EQ(e0.min, e1.min);
    result = result && TEST_DOUBLE_EQ(e0.max, e1.max);
    result = result && TEST_DOUBLE_EQ(e0.delta, e1.delta);
    result = result && TEST_DOUBLE_EQ(e0.offset, e1.offset);
#undef TEST_DOUBLE_EQ

    result = result && (e0.bw == e1.bw);
    return result;
}

static void compareEncodings(TfEncoding e0, TfEncoding e1, double err)
{
    EXPECT_NEAR(e0.min, e1.min, err);
    EXPECT_NEAR(e0.max, e1.max, err);
    EXPECT_NEAR(e0.delta, e1.delta, err);
    EXPECT_NEAR(e0.offset, e1.offset, err);
}

template <typename T>
bool compareTensors(const T* v1, const T* v2, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i)
    {
        if (v1[i] != v2[i])
        {
            return false;
        }
    }
    return true;
}
#endif   // TEST_QUANTIZATION_LIB_HPP
