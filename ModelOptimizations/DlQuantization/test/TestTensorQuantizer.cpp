//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
#include <random>

#include "DlQuantization/TensorQuantizer.h"
#include "test_quantization_lib.hpp"

using namespace DlQuantization;

class TestTensorQuantizer : public ::testing::Test
{
protected:
    std::vector<float> data1, data2, data3, data4;
    std::vector<uint32_t> shape1, shape2, shape3;
    std::unique_ptr<TensorQuantizer> enhancedTensorQuant;
    std::unique_ptr<TensorQuantizer> tfTensorQuant;

    void SetUp()
    {
        if (data1.size() == 0)
        {
            data1.resize(24);
            std::iota(std::begin(data1), std::end(data1), 0);
            shape1 = {2, 3, 2, 2};
        }

        if (data2.size() == 0)
        {
            data2.resize(60);
            float t = -15;
            for (uint32_t i = 0; i < data2.size(); ++i)
            {
                data2[i] = t;
                t += 0.5;
            }
            shape2 = {1, 4, 5, 3};
        }

        if (data3.size() == 0)
        {
            shape3 = {2, 5, 4, 1};
            data3.resize(40);
            std::mt19937 eng;
            std::normal_distribution<float> dist;
            for (auto& d: data3)
            {
                d = dist(eng);
            }
            std::iota(std::begin(data1), std::end(data1), 0);
        }

        if (data4.size() == 0)
        {
            float mean   = 2;
            float stddev = 2;
            std::normal_distribution<float> distribution(mean, stddev);
            std::mt19937 generator(1);

            unsigned int tensorCount = 6000;
            data4.resize(tensorCount);

            for (unsigned int i = 0; i < tensorCount; i++)
            {
                data4[i] = distribution(generator);
            }
        }

        enhancedTensorQuant.reset(new TensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST));
        tfTensorQuant.reset(new TensorQuantizer(QuantizationMode::QUANTIZATION_TF, ROUND_NEAREST));
    }
};

TEST_F(TestTensorQuantizer, SanityTestCpu)
{
    enhancedTensorQuant->setStrictSymmetric(false);
    enhancedTensorQuant->setUnsignedSymmetric(false);
    enhancedTensorQuant->updateStats(data4.data(), data4.size(), false);
    EXPECT_FALSE(enhancedTensorQuant->isEncodingValid);
    TfEncoding encoding = enhancedTensorQuant->computeEncoding(8, false);
    EXPECT_TRUE(enhancedTensorQuant->isEncodingValid);

    std::vector<float> inputTensor(data4.size(), 5);
    std::vector<float> quantizedTensor(data4.size());

    enhancedTensorQuant->quantizeDequantize(inputTensor.data(), inputTensor.size(), quantizedTensor.data(),
                                            encoding.min, encoding.max, 8, false);

    std::cout << "Encoding min=" << encoding.min << ", max=" << encoding.max << std::endl;
    double MAX = 8.88412;
    double MIN = -6.52711;
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE);

    std::cout << "input-data=" << inputTensor.data()[0] << ", quantized-data=" << quantizedTensor.data()[0]
              << std::endl;

    EXPECT_NE(inputTensor.data()[0], quantizedTensor.data()[0]);
    EXPECT_NEAR(quantizedTensor.data()[0], 5.0162, HISTOGRAM_BUCKET_SIZE);
}

TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataAsymmetricTFEnhanced)
{
    auto paramTensor = this->data4;
    auto tensorCount = paramTensor.size();
    TfEncoding encoding {};
    enhancedTensorQuant->computeEncodingFromData(8, paramTensor.data(), tensorCount, encoding,
                                                 ComputationMode::COMP_MODE_CPU, false, false, false);
    double MAX = 8.88412;
    double MIN = -6.52711;
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE);
}

TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataSymmetricTF)
{
    auto paramTensor = this->data4;
    auto tensorCount = paramTensor.size();
    TfEncoding encoding {};
    tfTensorQuant->computeEncodingFromData(8, paramTensor.data(), tensorCount, encoding, ComputationMode::COMP_MODE_CPU,
                                           true, false, true);

    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingDeltaOffsetSymmetricTF)
{
    auto paramTensor = this->data2;


    TfEncoding encoding {0, 0, 0, 0, 8};
    auto expected_encoding = getTfSymmetricEncoding(15.0, 8);

    // set up partial to mimic user provided scale and offset
    encoding.delta  = expected_encoding.delta;
    encoding.offset = expected_encoding.offset;

    tfTensorQuant->computePartialEncoding(8, encoding, true, false, true);


    // Expect the min and max values to be unchanged
    EXPECT_NEAR(encoding.max, expected_encoding.max, 0.001);
    EXPECT_NEAR(encoding.min, expected_encoding.min, 0.001);

    EXPECT_FLOAT_EQ(encoding.delta, expected_encoding.delta);
    EXPECT_FLOAT_EQ(encoding.offset, expected_encoding.offset);

    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingDeltaOffsetAsymmetricTF)
{
    auto paramTensor = this->data1;


    TfEncoding encoding {0, 0, 0, 0, 8};
    auto expected_encoding = getTfEncoding(0.0, 24.0, 8);

    // set up partial to mimic user provided scale and offset
    encoding.delta  = expected_encoding.delta;
    encoding.offset = expected_encoding.offset;

    tfTensorQuant->computePartialEncoding(8, encoding, false, false, false);


    // Expect the min and max values to be unchanged
    EXPECT_NEAR(encoding.max, expected_encoding.max, 0.001);
    EXPECT_NEAR(encoding.min, expected_encoding.min, 0.001);

    EXPECT_FLOAT_EQ(encoding.delta, expected_encoding.delta);
    EXPECT_FLOAT_EQ(encoding.offset, expected_encoding.offset);

    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingMinMaxSymmetricTF)
{
    auto paramTensor = this->data2;


    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    TfEncoding encoding {-expected_max, expected_max, 0, 0, 8};
    tfTensorQuant->computePartialEncoding(8, encoding, true, false, true);


    // Expect the min and max values to be unchanged
    EXPECT_EQ(encoding.max, expected_max);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SANITY_GeneratePerChannelParams)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> encodings;
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splitParams;

    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF, ROUND_NEAREST);
    tensorQuantizer.generatePerChannelEncodings(data1.data(), shape1, axis, encodings, bw, splitParams, splitShape,
                                                false);

    ASSERT_EQ(encodings.size(), shape1[axis]);
    ASSERT_EQ(splitParams.size(), shape1[axis]);

    std::vector<uint32_t> expectedOutputShape = {2, 1, 2, 2};
    ASSERT_EQ(splitShape, expectedOutputShape);

    std::vector<std::vector<float>> expectedOutputData(3);
    expectedOutputData[0] = {0, 1, 2, 3, 12, 13, 14, 15};
    expectedOutputData[1] = {4, 5, 6, 7, 16, 17, 18, 19};
    expectedOutputData[2] = {8, 9, 10, 11, 20, 21, 22, 23};

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(0, 15, 8);
    expectedEncodings[1] = getTfEncoding(0, 19, 8);
    expectedEncodings[2] = getTfEncoding(0, 23, 8);

    ASSERT_EQ(splitParams.size(), expectedOutputData.size());
    ASSERT_EQ(splitParams[0], expectedOutputData[0]);
    ASSERT_EQ(splitParams[1], expectedOutputData[1]);
    ASSERT_EQ(splitParams[2], expectedOutputData[2]);

    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }
}


// 1. Quantize some per channel data using QuantizePerChannelParamsPacked()
TEST_F(TestTensorQuantizer, SANITY_QuantizePerChannelTensorPackedAsymmetric)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> expectedEncodings(4);
    expectedEncodings[0]                = getTfEncoding(-15, 0, 8);
    expectedEncodings[1]                = getTfEncoding(-7.5, 0, 8);
    expectedEncodings[2]                = getTfEncoding(0, 7, 8);
    expectedEncodings[3]                = getTfEncoding(0, 14.5, 8);
    std::vector<uint8_t> expectedParams = {0,   9,   17,  26,  34,  43,  51,  60,  68,  77,  85,  94,  102, 111, 119,
                                           0,   17,  34,  51,  68,  85,  102, 119, 136, 153, 170, 187, 204, 221, 238,
                                           0,   18,  36,  55,  73,  91,  109, 128, 146, 164, 182, 200, 219, 237, 255,
                                           132, 141, 149, 158, 167, 176, 185, 193, 202, 211, 220, 229, 237, 246, 255};


    std::vector<TfEncoding> encodings;
    std::vector<uint8_t> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(this->data2.data(), this->shape2, axis, params_quantized, encodings,
                                                  bw, DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
        printEncoding(encodings[i]);
    }

    EXPECT_TRUE(compareTensors(params_quantized.data(), expectedParams.data(), this->data2.size()));
}

// 1. Quantize some per channel data using QuantizePerChannelParamsPacked()
TEST_F(TestTensorQuantizer, SANITY_QuantizePerChannelTensorPackedSymmetric)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> expectedEncodings;
    expectedEncodings.emplace_back(getTfSymmetricEncoding(15, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(7.5, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(7, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(14.5, 8));
    std::vector<int8_t> expectedParams = {
        -127, -123, -119, -114, -110, -106, -102, -97, -93, -89, -85, -80, -76, -72, -68, -127, -119, -110, -102, -93,
        -85,  -76,  -68,  -59,  -51,  -42,  -34,  -25, -17, -8,  0,   9,   18,  27,  36,  45,   54,   64,   73,   82,
        91,   100,  109,  118,  127,  66,   70,   74,  79,  83,  88,  92,  96,  101, 105, 109,  114,  118,  123,  127};


    std::vector<TfEncoding> encodings;
    std::vector<uint8_t> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(this->data2.data(), this->shape2, axis, params_quantized, encodings,
                                                  bw, DlQuantization::RoundingMode::ROUND_NEAREST, false, true);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    EXPECT_TRUE(compareTensors((int8_t*) params_quantized.data(), expectedParams.data(), this->data2.size()));
}

// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensor)
{
    int bw       = 8;
    int32_t axis = 3;


    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(-15, 13.5, 8);
    expectedEncodings[1] = getTfEncoding(-14.5, 14, 8);
    expectedEncodings[2] = getTfEncoding(-14, 14.5, 8);

    std::vector<float> expectedParams = {
        -14.9765, -14.5294, -13.9706, -13.5235, -12.9647, -12.5176, -11.9588, -11.5118, -10.9529, -10.5059,
        -9.94706, -9.5,     -9.05294, -8.49412, -8.04706, -7.48824, -7.04118, -6.48235, -6.03529, -5.47647,
        -5.02941, -4.47059, -4.02353, -3.46471, -3.01765, -2.45882, -2.01176, -1.45294, -1.00588, -0.447059,
        0,        0.447059, 1.00588,  1.45294,  2.01176,  2.45882,  3.01765,  3.46471,  4.02353,  4.47059,
        5.02941,  5.47647,  6.03529,  6.48235,  7.04118,  7.48824,  8.04706,  8.49412,  9.05294,  9.5,
        9.94706,  10.5059,  10.9529,  11.5118,  11.9588,  12.5176,  12.9647,  13.5235,  13.9706,  14.5294};


    std::vector<TfEncoding> encodings;
    std::vector<float> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizeDequantizePerChannelTensor(this->data2.data(), this->shape2, axis, params_quantized.data(),
                                                      encodings, bw, DlQuantization::RoundingMode::ROUND_NEAREST, false,
                                                      false);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}

// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensorSymmetric)
{
    int bw       = 8;
    int32_t axis = 2;


    std::vector<TfEncoding> expectedEncodings(5);
    expectedEncodings[0] = getTfSymmetricEncoding(15, 8);
    expectedEncodings[1] = getTfSymmetricEncoding(13.5, 8);
    expectedEncodings[2] = getTfSymmetricEncoding(12, 8);
    expectedEncodings[3] = getTfSymmetricEncoding(13, 8);
    expectedEncodings[4] = getTfSymmetricEncoding(14.5, 8);

    std::vector<float> expectedParams = {
        -15,      -14.5276, -14.0551, -13.5,    -12.9685, -12.5433, -12,      -11.5276, -10.9606, -10.5433,
        -10.0315, -9.51968, -9.01968, -8.44882, -7.99213, -7.44094, -6.9685,  -6.49606, -5.95276, -5.52756,
        -4.99606, -4.53543, -3.9685,  -3.49606, -2.9685,  -2.45669, -2.04724, -1.48425, -1.02756, -0.456693,
        0,        0.472441, 0.944882, 1.48819,  2.01969,  2.55118,  3.02362,  3.49606,  3.9685,   4.50394,
        5.01575,  5.52756,  6.05118,  6.50787,  6.96457,  7.55906,  8.0315,   8.50394,  9.03543,  9.46063,
        9.99213,  10.4882,  10.9606,  11.5276,  11.9764,  12.4882,  13,       13.4724,  14.0433,  14.5};


    std::vector<TfEncoding> encodings;
    std::vector<float> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizeDequantizePerChannelTensor(this->data2.data(), this->shape2, axis, params_quantized.data(),
                                                      encodings, bw, DlQuantization::RoundingMode::ROUND_NEAREST, false,
                                                      true);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.0001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}


// test parameter quantization for bw=8, number range: -50...80
// also test encoding
TEST_F(TestTensorQuantizer, SANITY_QuantizeTensorPackedAsymmetric)
{
    // Test data
    float data[]            = {-40, -1, 0, 1, 2, -50, 80};
    uint8_t data_expected[] = {20, 96, 98, 100, 102, 0, 255};
    int cnt                 = 7;
    int bw                  = 8;

    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> output(cnt);
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->updateStats(data, cnt, false);
    encoding = tfTensorQuant->computeEncoding(bw, false);
    tfTensorQuant->quantizeTensorPacked(data, cnt, output, encoding.min, encoding.max, bw,
                                        DlQuantization::ROUND_NEAREST, false, false);
    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(output[i], data_expected[i]);
    }
    // Check encoding
    TfEncoding encoding_expected = getTfEncoding(-50, 80, bw);
    EXPECT_TRUE(compareEncodings(encoding_expected, encoding));
}


// 1. Quantize some data using QuantizeParamsPacked()
// 2. Dequantize the result using DeQuantize()
// The final result will be close to the original data
// Test both the vector and pointer API versions of QuantizeParamsPacked() and
// DeQuantize().
TEST_F(TestTensorQuantizer, SANITY_Dequantize)
{
    // Unquantized data
    float data_unquantized[] = {-40, -1, 0, 1, 2, -50, 80};
    // Expected quantized data
    uint8_t data_quantized_expected[] = {20, 96, 98, 100, 102, 0, 255};
    int cnt                           = 7;
    int bw                            = 8;
    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> data_quantized(cnt);

    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->updateStats(data_unquantized, cnt, false);
    encoding = tfTensorQuant->computeEncoding(bw, false);
    tfTensorQuant->quantizeTensorPacked(data_unquantized, cnt, data_quantized, encoding.min, encoding.max, bw,
                                        DlQuantization::ROUND_NEAREST, false, false);

    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(data_quantized[i], data_quantized_expected[i]);
    }
    // De-quantize values
    std::vector<float> data_dequantized(cnt);
    // Test the pointer API.
    tfTensorQuant->dequantize(data_quantized.data(), cnt, encoding.min, encoding.max, bw, data_dequantized.data(),
                              false);

    // Check de-quantized values
    float data_dequantized_expected[] = {-39.7647f, -1.01961f, 0, 1.01961f, 2.03922f, -49.9608f, 80.0392f};
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(data_dequantized[i], data_dequantized_expected[i], 1e-4);
    }
}


// 1. Quantize some data using QuantizeParamsPacked()
// 2. Dequantize the result using DeQuantize()
// The final result will be close to the original data
// Test both the vector and pointer API versions of QuantizeParamsPacked() and
// DeQuantize().
TEST_F(TestTensorQuantizer, SANITY_DequantizePerChannel)
{
    // Unquantized data
    float data_unquantized[] = {-40, -1, 0, 1, 2, -50, 80, 1, 30, 10, 25, 3, 2, -50, 70, 1};
    // Expected quantized data
    uint8_t data_quantized_expected[] = {20, 96, 98, 100, 102, 0, 255, 100, 170, 127, 159, 112, 110, 0, 255, 108};
    int cnt                           = 16;
    int bw                            = 8;
    size_t axis                       = 2;

    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> data_quantized(cnt);
    std::vector<uint32_t> inputShape {1, 1, 2, 8};

    std::vector<TfEncoding> encodings;
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(data_unquantized, inputShape, axis, data_quantized, encodings, bw,
                                                  DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(data_quantized[i], data_quantized_expected[i]);
    }
    // De-quantize values
    std::vector<float> data_dequantized(cnt);
    // Test the pointer API.
    tfTensorQuant->dequantizePerChannelTensor(data_quantized.data(), inputShape, axis, encodings, bw,
                                              data_dequantized.data(), false);

    // Check de-quantized values
    float data_dequantized_expected[] = {-39.7647f, -1.01961f, 0,        1.01961f, 2.03922f, -49.9608f,
                                         80.0392f,  1.01961f,  30.1176f, 9.88235f, 24.9412f, 2.82353f,
                                         1.88235f,  -49.8824f, 70.1176f, 0.941176f};

    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(data_dequantized[i], data_dequantized_expected[i], 1e-4);
    }
}

#ifdef GPU_QUANTIZATION_ENABLED

TEST_F(TestTensorQuantizer, SanityTestGpu)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }
    Blob<GpuDevice<float>> statsTensorBlob(statsTensor.data(), tensorCount);

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), statsTensor.size(), true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    Blob<GpuDevice<float>> inputTensorBlob(inputTensor.data(), tensorCount);

    std::vector<float> quantizedTensor(tensorCount, 0);
    Blob<GpuDevice<float>> quantTensorBlob(quantizedTensor.data(), tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensorBlob.getDataPtrOnDevice(), inputTensor.size(),
                                       quantTensorBlob.getDataPtrOnDevice(), encoding.min, encoding.max, 8, true);

    std::cout << "Encoding min=" << encoding.min << ", max=" << encoding.max << std::endl;
    double MAX = 8.88412;
    double MIN = -6.52711;
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE);

    std::cout << "input-data=" << inputTensorBlob.getDataPtrOnCpu()[0]
              << ", quantized-data=" << quantTensorBlob.getDataPtrOnCpu()[0] << std::endl;
    EXPECT_NE(inputTensorBlob.getDataPtrOnCpu()[0], quantTensorBlob.getDataPtrOnCpu()[0]);
    EXPECT_NEAR(quantTensorBlob.getDataPtrOnCpu()[0], 5.0162, HISTOGRAM_BUCKET_SIZE);
}

#endif
