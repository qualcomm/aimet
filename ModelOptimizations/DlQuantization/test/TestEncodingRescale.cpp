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

#include <gtest/gtest.h>
#include <vector>

#include "DlQuantization/EncodingRescale.hpp"

using namespace DlQuantization;

class TestEncodingRescale : public ::testing::Test
{
protected:
    std::vector<float> perChannelWeightScale;
    std::vector<float> perTensorWeightScale;
    std::vector<float> bias;
    std::vector<float> requantScale;
    std::vector<float> biasSim;
    ConvSpecArgs<float> convArgs;

    void SetUp()
    {
        if (perChannelWeightScale.size() == 0)
        {
            perChannelWeightScale.insert(perChannelWeightScale.end(), {-0.5f, -0.25f, 0.25, 0.5, 0.75}); 
        }
        if (perTensorWeightScale.size() == 0)
        {
            perTensorWeightScale.insert(perTensorWeightScale.end(), {0.75});
        }
        if (bias.size() == 0)
        {
            bias.insert(bias.end(), {-0.1f, -0.05f, 0.0f, 0.05f, 0.1f});
        }
        if (biasSim.size() == 0)
        {
            biasSim.resize(bias.size());
        }
        convArgs = {.out_encoding_delta = 0.0002f,
                      .out_encoding_offset = -128,
                      .input_scale = 0.0001f};
    }
};

TEST_F(TestEncodingRescale, SanityTestAct8BwPerChannelQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    std::vector<float> expectedRequantScale = {-0.25f, -0.125f, 0.125f, 0.25f, 0.375f};
    std::vector<float> expectedBiasSim = {1488, 976, 1024, 1512, 1674};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perChannelWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerTensorQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    std::vector<float> expectedRequantScale = {0.375f};
    std::vector<float> expectedBiasSim = {-992, -325, 341, 1008, 1675};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perTensorWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerChannelQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {5, 3, 4, 5, 6};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerTensorQuantSimOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {-4, -2, 1, 3, 6};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerChannelQuantSimNoOffsetWrap)
{
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    std::vector<float> expectedRequantScale = {-0.25f, -0.125f, 0.125f, 0.25f, 0.375f};
    std::vector<float> expectedBiasSim = {2000, 2000, 0, 1000, 1333};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perChannelWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerTensorQuantSimNoOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    std::vector<float> expectedRequantScale = {0.375f};
    std::vector<float> expectedBiasSim = {-1333, -667, 0, 667, 1333};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perTensorWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerChannelQuantSimNoOffsetWrap)
{
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {7, 7, 0, 3, 5};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerTensorQuantSimNoOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {-6, -3, 0, 2, 5};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

