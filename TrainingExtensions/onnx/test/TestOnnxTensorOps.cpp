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

#include "DlQuantization/Quantization.hpp"
#include "QuantizeDequantizeUtils.hpp"
#include "cuda_runtime_api.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>


class TestOnnxTensorOps : public ::testing::Test
{
};


void launchPermuteEncodingKernel(float* in, float* out, const BroadcastShapeInfo& shapeInfo, bool useCuda)
{
    size_t numElements = shapeInfo.numElements;
    void* inputBuffer;
    void* outputBuffer;
    if (useCuda)
    {
        cudaMalloc(&inputBuffer, sizeof(float) * numElements);
        cudaMalloc(&outputBuffer, sizeof(float) * numElements);
        cudaMemcpy(inputBuffer, in, numElements * sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        inputBuffer = in;
        outputBuffer = out;
    }

    copyToContiguousBlockLayout((float*) inputBuffer, (float*) outputBuffer, shapeInfo, useCuda);

    if (useCuda)
    {
        cudaMemcpy(out, outputBuffer, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(outputBuffer);
        cudaFree(inputBuffer);
    }

}

TEST(TestOnnxTensorOps, TensorBlockPermute) {
    const int numElements = 16;

    float in[numElements] = {1.0f, 2.0f,     3.0f, 4.0f,
                             5.0f, 6.0f,     7.0f, 8.0f,
                             9.0f, 10.0f,    11.0f, 12.0f,
                             13.0f, 14.0f,   15.0f, 16.0f};

    float out[numElements];

    std::vector<int64_t> inputShape = {4, 2, 2};
    std::vector<int64_t> encodingShape = {2, 1, 2};
    std::vector<int64_t> inputStrides = {8, 4, 2, 1};
    std::vector<int64_t> encodingStrides = {2, 0, 0, 1};


    float expected[numElements] = {1.0f,  3.0f,  5.0f,  7.0f,
                                   2.0f,  4.0f,  6.0f,  8.0f,
                                   9.0f, 11.0f, 13.0f, 15.0f,
                                   10.0f, 12.0f, 14.0f, 16.0f};

    bool useCuda[2] = {false, true};

    for (int c = 0; c < 2; c++)
    {
        BroadcastShapeInfo shapeInfo{inputShape, 2, 0, 2};
        // Launch the kernel
        launchPermuteEncodingKernel(in, out, shapeInfo, useCuda[c]);
        for (int i = 0; i < numElements; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
        }
    }

}


TEST(TestOnnxTensorOps, TensorBlockPermute2) {
    const int numElements = 64;
    const int outDims = 4;
    const int numElementsPerEncoding = 8;


    float inp[4][2][2][4];   // becomes [4][2][2][4]
    float* in = &inp[0][0][0][0];
    float enc[4][1][2][1];  // becomes [4][1][2][1]
    std::vector<int64_t> encodingStrides = {2, 0, 1, 0};
    std::vector<int64_t> inputStrides = {16, 8, 4, 1};

    std::vector<int64_t> inputShape = {4, 2, 8};

    BroadcastShapeInfo shapeInfo{inputShape, 0, 2, 4};

    float out[4 * 2 * 8];

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                for (int m = 0; m < 4; m++)
                {
                    inp[i][j][k][m] = static_cast<float>(k) + 2.0f * static_cast<float>(i);
                }
            }
        }
    }

    float expected[numElements];
    for (int i = 0; i < numElements; i++)
    {
        expected[i] = static_cast<float>(i / numElementsPerEncoding);
    }

    bool useCuda[2] = {false, true};

    for (int c = 0; c < 2; c++)
    {
        // Launch the kernel
        launchPermuteEncodingKernel(in, out, shapeInfo, useCuda[c]);

        // Check the results
        for (int i = 0; i < numElements; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
        }
    }
}

TEST(TestOnnxTensorOps, TensorBlockPermute3) {
    const int numElements = 16;

    float in[numElements] = {1.0f, 2.0f,     3.0f, 4.0f,
                             5.0f, 6.0f,     7.0f, 8.0f,
                             9.0f, 10.0f,    11.0f, 12.0f,
                             13.0f, 14.0f,   15.0f, 16.0f};

    float out[numElements];


    std::vector<int64_t> inputShape = {4, 2, 2};
    std::vector<int64_t> encodingShape = {2, 1};
    std::vector<int64_t> inputStrides = {4, 2, 1};
    std::vector<int64_t> encodingStrides = {0, 1, 0};

    BroadcastShapeInfo shapeInfo{inputShape, 1, -1, 0};

    // Check the results
    float expected[numElements] = {1.0f, 2.0f,   5.0f, 6.0f,
                                   9.0f, 10.0f,  13.0f, 14.0f,
                                   3.0f, 4.0f,   7.0f, 8.0f,
                                   11.0f, 12.0f, 15.0f, 16.0f};

    bool useCuda[2] = {false, true};

    for (int c = 0; c < 2; c++)
    {
        // Launch the kernel
        launchPermuteEncodingKernel(in, out, shapeInfo, useCuda[c]);

        for (int i = 0; i < numElements; i++)
        {
            EXPECT_EQ(out[i], expected[i]);
        }
    }
}


TEST(TestOnnxTensorOps, TestBroadcastShapeInfo) {
    const std::vector<int64_t> inputShape{8, 12, 5, 2};
    const int numElements = 8 * 12 * 5 * 2;
    const int channelAxis = 0;
    const int blockAxis = 1;
    const int blockSize = 3;

    BroadcastShapeInfo shapeInfo = BroadcastShapeInfo(inputShape,
                                                      channelAxis,
                                                      blockAxis,
                                                      blockSize
                                                      );


    EXPECT_EQ(shapeInfo.numEncodings, 8 * 4);

    const std::vector<int64_t> expectedTensorStrides = {120, 30, 10, 2, 1};
    auto tensorStrides = shapeInfo.tensorStrides;
    EXPECT_EQ(tensorStrides.size(), expectedTensorStrides.size());
    for (int i = 0; i < expectedTensorStrides.size(); i++)
    {
        EXPECT_EQ(expectedTensorStrides[i], tensorStrides[i]);
    }

    const std::vector<int64_t> expectedEncodingStrides = {4, 1, 0, 0, 0};
    auto encodingStrides = shapeInfo.encodingStrides;
    EXPECT_EQ(encodingStrides.size(), expectedEncodingStrides.size());
    for (int i = 0; i < expectedEncodingStrides.size(); i++)
    {
        EXPECT_EQ(expectedEncodingStrides[i], encodingStrides[i]);
    }

    EXPECT_TRUE(shapeInfo.hasContiguousBlocks());


}

TEST(TestOnnxTensorOps, TestBroadcastShapeInfo2) {
    const std::vector<int64_t> inputShape{10, 4, 10};
    const int numElements = 400;
    const int channelAxis = 1;
    const int blockAxis = 0;
    const int blockSize = 2;

    BroadcastShapeInfo shapeInfo = BroadcastShapeInfo(inputShape,
                                                      channelAxis,
                                                      blockAxis,
                                                      blockSize
                                                      );


    EXPECT_EQ(shapeInfo.numEncodings, 5 * 4);

    const std::vector<int64_t> expectedTensorStrides = {80, 40, 10, 1};
    auto tensorStrides = shapeInfo.tensorStrides;
    EXPECT_EQ(tensorStrides.size(), expectedTensorStrides.size());
    for (int i = 0; i < expectedTensorStrides.size(); i++)
    {
        EXPECT_EQ(expectedTensorStrides[i], tensorStrides[i]);
    }

    const std::vector<int64_t> expectedEncodingStrides = {4, 0, 1, 0};
    auto encodingStrides = shapeInfo.encodingStrides;
    EXPECT_EQ(encodingStrides.size(), expectedEncodingStrides.size());
    for (int i = 0; i < expectedEncodingStrides.size(); i++)
    {
        EXPECT_EQ(expectedEncodingStrides[i], encodingStrides[i]);
    }

    EXPECT_FALSE(shapeInfo.hasContiguousBlocks());


}

TEST(TestOnnxTensorOps, TestBroadcastShapeInfo3) {
    const std::vector<int64_t> inputShape = {4, 2, 2};
    const std::vector<int64_t> encodingShape = {2, 1, 2};

    const int numElements = 16;
    const int channelAxis = 2;
    const int blockAxis = 0;
    const int blockSize = 2;

    const std::vector<int64_t> expectedTensorStrides = {8, 4, 2, 1};
    const std::vector<int64_t> expectedEncodingStrides = {2, 0, 0, 1};

    BroadcastShapeInfo shapeInfo = BroadcastShapeInfo(inputShape,
                                                      channelAxis,
                                                      blockAxis,
                                                      blockSize
                                                      );


    EXPECT_EQ(shapeInfo.numEncodings, 4);

    auto tensorStrides = shapeInfo.tensorStrides;
    EXPECT_EQ(tensorStrides.size(), expectedTensorStrides.size());
    for (int i = 0; i < expectedTensorStrides.size(); i++)
    {
        EXPECT_EQ(expectedTensorStrides[i], tensorStrides[i]);
    }

    auto encodingStrides = shapeInfo.encodingStrides;
    EXPECT_EQ(encodingStrides.size(), expectedEncodingStrides.size());
    for (int i = 0; i < expectedEncodingStrides.size(); i++)
    {
        EXPECT_EQ(expectedEncodingStrides[i], encodingStrides[i]);
    }

    EXPECT_FALSE(shapeInfo.hasContiguousBlocks());

}

