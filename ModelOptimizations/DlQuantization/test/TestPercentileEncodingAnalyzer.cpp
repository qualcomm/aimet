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

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "test_quantization_lib.hpp"
#include <DlQuantization/TensorQuantizer.h>
#include <PercentileEncodingAnalyzer.h>

using namespace DlQuantization;

template <typename TypeParam>
class TestPercentileEncodingAnalyzer : public ::testing::Test
{
};
// Test on CPU and GPU with float and double
TYPED_TEST_CASE(TestPercentileEncodingAnalyzer, TestDataTypesAndDevices);

TYPED_TEST(TestPercentileEncodingAnalyzer, Asymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer;
    analyzer.setPercentileValue(99.9999);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    // Use larger tensor size to avoid ending up with calibrated min, max same as
    // original min, max
    unsigned int tensorCount = 100000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        min       = std::min(min, double(tensor[i]));
        max       = std::max(max, double(tensor[i]));
    }

    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, false, false, false);

    std::cout << "Absolute Min: " << min << std::endl;
    std::cout << "Absolute Max: " << max << std::endl;

    std::cout << encoding.min << std::endl;
    std::cout << encoding.max << std::endl;
    std::cout << encoding.delta << std::endl;
    std::cout << encoding.offset << std::endl;

    // We know we have a normal distribution. We expect the encoding to cover
    // at least 2 standard deviations, and at most 6.
    std::cout << "mean - 6 * stddev: " << mean - 6 * stddev << "\n";
    std::cout << "mean - 2 * stddev: " << mean - 2 * stddev << "\n";
    std::cout << "mean + 2 * stddev: " << mean + 2 * stddev << "\n";
    std::cout << "mean + 6 * stddev: " << mean + 6 * stddev << "\n";
    EXPECT_GT(encoding.min, mean - 6 * stddev);
    EXPECT_LT(encoding.min, mean - 2 * stddev);
    EXPECT_GT(encoding.max, mean + 2 * stddev);
    EXPECT_LT(encoding.max, mean + 6 * stddev);
}

TYPED_TEST(TestPercentileEncodingAnalyzer, Symmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer;
    analyzer.setPercentileValue(99.9);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, false);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    std::cout << "Absolute Min: " << absoluteMin << std::endl;
    std::cout << "Absolute Max: " << absoluteMax << std::endl;
    std::cout << encoding.min << std::endl;
    std::cout << encoding.max << std::endl;
    std::cout << encoding.delta << std::endl;
    std::cout << encoding.offset << std::endl;

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    EXPECT_GT(encoding.min, absoluteMin);
    EXPECT_LT(encoding.max, absoluteMax);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.offset, -128);
    EXPECT_EQ(encoding.bw, 8);
}

TYPED_TEST(TestPercentileEncodingAnalyzer, StrictSymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer;
    analyzer.setPercentileValue(99.9);

    float mean   = -2;
    float stddev = 1;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, true, false);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    std::cout << "Absolute Min: " << absoluteMin << std::endl;
    std::cout << "Absolute Max: " << absoluteMax << std::endl;
    std::cout << encoding.min << std::endl;
    std::cout << encoding.max << std::endl;
    std::cout << encoding.delta << std::endl;
    std::cout << encoding.offset << std::endl;

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    EXPECT_GT(encoding.min, absoluteMin);
    EXPECT_LT(encoding.max, absoluteMax);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.offset, -127);
    EXPECT_EQ(encoding.bw, 8);
    EXPECT_EQ(encoding.min, -encoding.max);
}

TYPED_TEST(TestPercentileEncodingAnalyzer, SymmetricUnsigned)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer;
    analyzer.setPercentileValue(99.9999);

    float mean   = -2;
    float stddev = 1;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        tensor[i] = std::max(tensor[i], (dataType) 0.0);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, true);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    std::cout << "Absolute Min: " << absoluteMin << std::endl;
    std::cout << "Absolute Max: " << absoluteMax << std::endl;
    std::cout << "Encoding Min: " << encoding.min << std::endl;
    std::cout << "Encoding Max: " << encoding.max << std::endl;
    std::cout << "Encoding Delta: " << encoding.delta << std::endl;
    std::cout << "Encoding Offset: " << encoding.offset << std::endl;

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    EXPECT_EQ(encoding.min, 0);
    EXPECT_NEAR(encoding.max, absoluteMax, 0.015);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.bw, 8);
}

TYPED_TEST(TestPercentileEncodingAnalyzer, Symmetric_Percentile100)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer;
    analyzer.setPercentileValue(100);

    float mean   = -1;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, false);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    std::cout << "Absolute Min: " << absoluteMin << std::endl;
    std::cout << "Absolute Max: " << absoluteMax << std::endl;
    std::cout << encoding.min << std::endl;
    std::cout << encoding.max << std::endl;
    std::cout << encoding.delta << std::endl;
    std::cout << encoding.offset << std::endl;

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (absoluteMax - absoluteMin) / PDF_SIZE;
    EXPECT_LT(encoding.max, absoluteMax + PDF_SIZE);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.offset, -128);
    EXPECT_EQ(encoding.bw, 8);
}

TYPED_TEST(TestPercentileEncodingAnalyzer, AllSameValuesAsymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer1;
    analyzer1.setPercentileValue(99.9999);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor1(tensorCount, 4);
    Blob<TypeParam> tensorBlob1(tensor1.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer1.updateStats(tensorBlob1.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding1 = analyzer1.computeEncoding(8, false, false, false);
    EXPECT_LE(encoding1.min, 0);   // 0 is included
    EXPECT_GE(encoding1.max, 3.99006);


    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer2;
    analyzer2.setPercentileValue(99.9999);

    std::vector<dataType> tensor2(tensorCount, -5);
    Blob<TypeParam> tensorBlob2(tensor2.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer2.updateStats(tensorBlob2.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding2 = analyzer2.computeEncoding(8, false, false, false);
    EXPECT_LE(encoding2.min, -4.99992);
    EXPECT_GE(encoding2.max, 0);   // 0 is included
}

TYPED_TEST(TestPercentileEncodingAnalyzer, AllZeroesAsymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate PercentileEncodingAnalyzer
    DlQuantization::PercentileEncodingAnalyzer<dataType> analyzer1;
    analyzer1.setPercentileValue(99.9999);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor1(tensorCount, 0);
    Blob<TypeParam> tensorBlob1(tensor1.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer1.updateStats(tensorBlob1.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding1 = analyzer1.computeEncoding(8, false, false, false);

    EXPECT_NEAR(encoding1.min, -1.00392, 0.0001);
    EXPECT_NEAR(encoding1.max, 0.996078, 0.0001);
    EXPECT_EQ(encoding1.offset, -128);
    EXPECT_EQ(encoding1.bw, 8);
}
