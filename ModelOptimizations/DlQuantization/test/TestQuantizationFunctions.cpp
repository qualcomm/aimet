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


#include <cstdint>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "gtest/gtest.h"

#include <math_functions.hpp>

#ifdef GPU_QUANTIZATION_ENABLED
#include "cuda_runtime_api.h"
#endif


void launchBlockQdqKernel(float* in, float* out, std::vector<DlQuantization::TfEncoding>& encodings, const DlQuantization::TensorDims& inputShape, const DlQuantization::TensorDims& encodingShape,
                          int64_t numElements, bool useCuda)
{
    void* inputBuffer;
    void* outputBuffer;

    if (useCuda)
    {
#ifdef GPU_QUANTIZATION_ENABLED
        cudaMalloc(&inputBuffer, sizeof(float) * numElements);
        cudaMalloc(&outputBuffer, sizeof(float) * numElements);
        // copy input to gpu
        cudaMemcpy(inputBuffer, in, numElements * sizeof(float), cudaMemcpyHostToDevice);

        DlQuantization::quantizeDequantizeBroadcast((float*) inputBuffer, (float*) outputBuffer, encodings, inputShape, encodingShape, DlQuantization::COMP_MODE_GPU);

        // copy output to cpu
        cudaMemcpy(out, outputBuffer, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        // free gpu memory
        cudaFree(outputBuffer);
        cudaFree(inputBuffer);
#endif
    }
    else
    {
        DlQuantization::quantizeDequantizeBroadcast(in, out, encodings, inputShape, encodingShape, DlQuantization::COMP_MODE_CPU);
    }

}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeBroadcast) {
    int numel = 16;
    DlQuantization::TensorDims inputShape = {2, 2, 2, 2};
    DlQuantization::TensorDims encodingShape = {2, 1, 1, 2};
    const std::vector<int64_t> inputStrides = {8, 4, 2, 1};
    const std::vector<int64_t> encodingStrides = {2, 0, 0, 1};
    const std::vector<float> encodingMax = {63.5, 127.0, 254.0, 508.0};
    const std::vector<float> encodingMin = {-64.0, -128.0, -256.0, -512.0};
    const std::vector<float> encodingDelta = {0.5, 1.0, 2.0, 4.0};
    const std::vector<float> encodingOffset = {-128, -128, -128, -128};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());

    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }
    float out[numel];


    float input[4 * 2 * 2] = {
        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,       -3.1, -3.1,

        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,        -3.1, -3.1,
    };

    float expected[4 * 2 * 2] = {
        -64.0, -125.0,     48.5, 48.0,
         63.5, 68.0,       -3.0, -3.0,

        -126.0, -124.0,    48.0, 48.0,
         68.0, 68.0,        -4.0, -4.0

    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif

    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernel(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
            out[i] = 0; // Clear output
        }
    }

}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeBroadcast2) {
    int numel = 24;
    DlQuantization::TensorDims inputShape = {2, 3, 4};
    DlQuantization::TensorDims encodingShape = {2, 3, 1};
    const std::vector<int64_t> inputStrides = {12, 4, 1};
    const std::vector<int64_t> encodingStrides = {3, 1, 0};
    const std::vector<float> encodingDelta = {0.25, 1.0, 0.5, 2.0, 0.25, 10.0};
    const std::vector<float> encodingOffset = {0, 0, 0, -1, -10, 0};
    const std::vector<float> encodingMin = {0, 0, 0, -2, -2.5, 0};
    const std::vector<float> encodingMax = {255. * 0.25, 255.0, 127.5, 508., 245. * 0.25, 2550.};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());
    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }
    float out[numel];


    float input[numel] = {
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
    };

    float expected[numel] = {
        0.25, 10.5, 0, 63.75,  // scale = .25
        0., 10., 0., 255.,  // scale = 1
        0., 10.5, 0., 127.5,  // scale = 0.5
        0., 10., -2., 508.,  // scale = 2. offset=-1
        0.25, 10.5, -2.5, 61.25,  // scale = .25
        0., 10., 0, 2550.,  // scale = 10
    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif


    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernel(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
            out[i] = 0; // Clear output
        }
    }

}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeBroadcast3) {
    int numel = 24;
    DlQuantization::TensorDims inputShape = {4, 2, 3};
    DlQuantization::TensorDims encodingShape = {2, 3};
    const std::vector<int64_t> inputStrides = {6, 3, 1};
    const std::vector<int64_t> encodingStrides = {0, 3, 1};
    const std::vector<float> encodingDelta = {0.25, 1.0, 0.5, 2.0, 0.25, 10.0};
    const std::vector<float> encodingOffset = {0, 0, 0, -1, -10, 0};
    const std::vector<float> encodingMin = {0, 0, 0, -2, -2.5, 0};
    const std::vector<float> encodingMax = {255. * 0.25, 255.0, 127.5, 508., 245. * 0.25, 2550.};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());

    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }
    float out[numel];


    float input[numel] = {
        0.126, 0.126, 0.126, 0.126, 0.126, 0.126,
        10.4,  10.4,  10.4,  10.4,  10.4,  10.4,
        -12.3, -12.3, -12.3, -12.3, -12.3, -12.3,
        10000, 10000, 10000, 10000, 10000, 10000,
    };

    float expected[numel] = {
    //  0.25     1.0      0.5      2.0      .25      10.0
        0.25,    0.,      0.,      0.,      0.25,    0.,
        10.5,    10.,     10.5,    10.,     10.5,    10.,
        -0.,     0.,      0.,      -2,      -2.5,    0.,
        63.75,   255.,    127.5,   508,     61.25,   2550,
    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif


    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernel(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
            out[i] = 0; // Clear output
        }
    }

}
