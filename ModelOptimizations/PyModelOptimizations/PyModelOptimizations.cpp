//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020 - 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "pybind11/complex.h"
#include <DlEqualization/CrossLayerScalingForPython.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DlCompression/ISVD.hpp"
#include "DlEqualization/BatchNormFoldForPython.h"
#include "DlEqualization/BiasCorrectionForPython.h"
#include "DlEqualization/CrossLayerScaling.h"
#include "DlEqualization/HighBiasFoldForPython.h"
#include "DlEqualization/def.h"
#include "DlQuantization/EncodingAnalyzerForPython.h"
#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"
#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "DlQuantization/TensorQuantizationSimForPython.h"
#include "DlQuantization/TensorQuantizerOpFacade.h"
#include "DlQuantization/EncodingRescale.hpp"
#include "PyTensorQuantizer.hpp"

namespace py = pybind11;

using namespace DlQuantization;
using namespace DlCompression;
using namespace AimetEqualization;

PYBIND11_MODULE(_libpymo, m)
{
    py::options options;
    options.show_function_signatures();
    options.show_user_defined_docstrings();

    // Quantization python bindings
    py::enum_<ComputationMode>(m, "ComputationMode")
        .value("COMP_MODE_CPU", ComputationMode::COMP_MODE_CPU)
        .value("COMP_MODE_GPU", ComputationMode::COMP_MODE_GPU)
        .export_values();

    py::enum_<QuantizationMode>(m, "QuantizationMode")
        .value("QUANTIZATION_TF", QuantizationMode::QUANTIZATION_TF)
        .value("QUANTIZATION_TF_ENHANCED", QuantizationMode::QUANTIZATION_TF_ENHANCED)
        .value("QUANTIZATION_RANGE_LEARNING", QuantizationMode::QUANTIZATION_RANGE_LEARNING)
        .value("QUANTIZATION_PERCENTILE", QuantizationMode::QUANTIZATION_PERCENTILE)
        .value("QUANTIZATION_MSE", QuantizationMode::QUANTIZATION_MSE)
        .value("QUANTIZATION_ENTROPY", QuantizationMode::QUANTIZATION_ENTROPY)
        .export_values();

    py::enum_<LayerInOut>(m, "LayerInOut")
        .value("LAYER_INPUT", LayerInOut::LAYER_INPUT)
        .value("LAYER_OUTPUT", LayerInOut::LAYER_OUTPUT)
        .export_values();

    py::enum_<RoundingMode>(m, "RoundingMode")
        .value("ROUND_NEAREST", RoundingMode::ROUND_NEAREST)
        .value("ROUND_STOCHASTIC", RoundingMode::ROUND_STOCHASTIC)
        .export_values();

    py::class_<DlQuantization::TfEncoding>(m, "TfEncoding")
        .def(py::init<>())
        .def_readwrite("min", &DlQuantization::TfEncoding::min)
        .def_readwrite("max", &DlQuantization::TfEncoding::max)
        .def_readwrite("delta", &DlQuantization::TfEncoding::delta)
        .def_readwrite("offset", &DlQuantization::TfEncoding::offset)
        .def_readwrite("bw", &DlQuantization::TfEncoding::bw);

    // Factory func
    py::class_<IQuantizer<float>>(m, "Quantizer");
    m.def("GetQuantizationInstance", &GetQuantizerInstance<float>);

    py::class_<IQuantizationEncodingAnalyzer<float>>(m, "QuantizationEncodingAnalyzer");
    m.def("GetQuantizationEncodingAnalyzerInstance", &getEncodingAnalyzerInstance<float>);

    // Compression python bindings
    py::enum_<COMPRESS_LAYER_TYPE>(m, "COMPRESS_LAYER_TYPE")
        .value("LAYER_TYPE_OTHER", COMPRESS_LAYER_TYPE::LAYER_TYPE_OTHER)
        .value("LAYER_TYPE_CONV", COMPRESS_LAYER_TYPE::LAYER_TYPE_CONV)
        .value("LAYER_TYPE_FC", COMPRESS_LAYER_TYPE::LAYER_TYPE_FC)
        .export_values();

    py::enum_<NETWORK_COST_METRIC>(m, "NETWORK_COST_METRIC")
        .value("COST_TYPE_MEMORY", NETWORK_COST_METRIC::COST_TYPE_MEMORY)
        .value("COST_TYPE_MAC", NETWORK_COST_METRIC::COST_TYPE_MAC)
        .export_values();

    py::enum_<SVD_COMPRESS_TYPE>(m, "SVD_COMPRESS_TYPE")
        .value("TYPE_NONE", SVD_COMPRESS_TYPE::TYPE_NONE)
        .value("TYPE_SINGLE", SVD_COMPRESS_TYPE::TYPE_SINGLE)
        .value("TYPE_SUCCESSIVE", SVD_COMPRESS_TYPE::TYPE_SUCCESSIVE)
        .export_values();

    py::class_<DlCompression::LayerAttributes<float>>(m, "LayerAttributes")
        .def(py::init<>())
        .def_readwrite("shape", &LayerAttributes<float>::shape)
        .def_readwrite("blobs", &LayerAttributes<float>::blobs)
        .def_readwrite("activation_dims", &LayerAttributes<float>::activation_dims)
        .def_readwrite("candidateRanks", &LayerAttributes<float>::candidateRanks)
        .def_readwrite("bestRanks", &LayerAttributes<float>::bestRanks)
        .def_readwrite("mode", &LayerAttributes<float>::mode)
        .def_readwrite("layerType", &LayerAttributes<float>::layerType)
        .def_readwrite("inputChannelMean", &LayerAttributes<float>::inputChannelMean)
        .def_readwrite("compressionRate", &LayerAttributes<float>::compressionRate);

    py::class_<DlQuantization::EncodingAnalyzerForPython>(m, "EncodingAnalyzerForPython")
        .def(py::init<DlQuantization::QuantizationMode>())
        .def("updateStats", &DlQuantization::EncodingAnalyzerForPython::updateStats)
        .def("computeEncoding", &DlQuantization::EncodingAnalyzerForPython::computeEncoding);

    py::class_<DlQuantization::TensorQuantizationSimForPython>(m, "TensorQuantizationSimForPython")
        .def(py::init<>())
        .def("quantizeDequantize",
             (py::array_t<float>(TensorQuantizationSimForPython::*)(py::array_t<float>, DlQuantization::TfEncoding&,
                                                                    DlQuantization::RoundingMode, unsigned int, bool)) &
                 DlQuantization::TensorQuantizationSimForPython::quantizeDequantize)
        .def("quantizeDequantize",
             (py::array_t<float>(TensorQuantizationSimForPython::*)(py::array_t<float>, DlQuantization::TfEncoding&,
                                                                    DlQuantization::RoundingMode, bool)) &
                 DlQuantization::TensorQuantizationSimForPython::quantizeDequantize);

    py::enum_<DlQuantization::TensorQuantizerOpMode>(m, "TensorQuantizerOpMode")
        .value("updateStats", DlQuantization::TensorQuantizerOpMode::updateStats)
        .value("oneShotQuantizeDequantize", DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
        .value("quantizeDequantize", DlQuantization::TensorQuantizerOpMode::quantizeDequantize)
        .value("passThrough", DlQuantization::TensorQuantizerOpMode::passThrough);

    py::class_<DlQuantization::PyTensorQuantizer>(m, "TensorQuantizer")
        .def(py::init<DlQuantization::QuantizationMode, DlQuantization::RoundingMode>())
        .def("updateStats",
             (void(PyTensorQuantizer::*)(py::array_t<float>, bool)) & DlQuantization::PyTensorQuantizer::updateStats)
        .def("computeEncoding", &DlQuantization::PyTensorQuantizer::computeEncoding)
        .def("quantizeDequantize",
             (void(PyTensorQuantizer::*)(py::array_t<float>, py::array_t<float>, double, double, unsigned int, bool)) &
                 DlQuantization::PyTensorQuantizer::quantizeDequantize)
        .def("resetEncodingStats", &DlQuantization::PyTensorQuantizer::resetEncodingStats)
        .def("setQuantScheme", &DlQuantization::PyTensorQuantizer::setQuantScheme)
        .def("getQuantScheme", &DlQuantization::PyTensorQuantizer::getQuantScheme)
        .def("setStrictSymmetric", &DlQuantization::PyTensorQuantizer::setStrictSymmetric)
        .def("getStrictSymmetric", &DlQuantization::PyTensorQuantizer::getStrictSymmetric)
        .def("setUnsignedSymmetric", &DlQuantization::PyTensorQuantizer::setUnsignedSymmetric)
        .def("getUnsignedSymmetric", &DlQuantization::PyTensorQuantizer::getUnsignedSymmetric)
        .def("getStatsHistogram", &DlQuantization::PyTensorQuantizer::getStatsHistogram)
        .def("setPercentileValue", &DlQuantization::PyTensorQuantizer::setPercentileValue)
        .def("getPercentileValue", &DlQuantization::PyTensorQuantizer::getPercentileValue)
        .def("computePartialEncoding", &DlQuantization::PyTensorQuantizer::computePartialEncoding)
        .def_readwrite("roundingMode", &DlQuantization::PyTensorQuantizer::roundingMode)
        .def_readwrite("isEncodingValid", &DlQuantization::PyTensorQuantizer::isEncodingValid);

    m.def("PtrToInt64", [](void* ptr) { return (uint64_t) ptr; });

    py::class_<ISVD<float>>(m, "Svd")
        //        .def(py::init<const std::string &>())
        .def("SetCandidateRanks", &ISVD<float>::SetCandidateRanks)
        .def("GetCandidateRanks", &ISVD<float>::GetCandidateRanks)
        .def("PrintCandidateRanks", &ISVD<float>::PrintCandidateRanks)
        .def("GetLayerType", &ISVD<float>::GetLayerType)
        .def("GetLayerNames", &ISVD<float>::GetLayerNames)
        .def("GetCompressionType", (SVD_COMPRESS_TYPE(ISVD<float>::*)(COMPRESS_LAYER_TYPE, const std::string&) const) &
                                       ISVD<float>::GetCompressionType)
        .def("GetCompressionType",
             (SVD_COMPRESS_TYPE(ISVD<float>::*)(const std::string&) const) & ISVD<float>::GetCompressionType)
        .def("SetCostMetric", &ISVD<float>::SetCostMetric)
        .def("StoreLayerAttributes", &ISVD<float>::StoreLayerAttributes)
        .def("GetLayerAttributes", &ISVD<float>::GetLayerAttributes, py::return_value_policy::reference)
        .def("ComputeNetworkCost", &ISVD<float>::ComputeNetworkCost)
        .def("GetCompressionScore", &ISVD<float>::GetCompressionScore)
        .def("SplitLayerWeights",
             (std::vector<std::vector<float>> &
              (ISVD<float>::*) (const std::string&, std::vector<std::vector<float>>& splitWeights,
                                const std::vector<unsigned int>&, const std::vector<unsigned int>&) ) &
                 ISVD<float>::SplitLayerWeights)
        .def("SplitLayerBiases",
             (std::vector<std::vector<float>> &
              (ISVD<float>::*) (const std::string&, std::vector<std::vector<float>>& splitBiases,
                                const std::vector<unsigned int>&, const std::vector<unsigned int>&) ) &
                 ISVD<float>::SplitLayerBiases)
        .def("StoreBestRanks", (void(ISVD<float>::*)(const int)) & ISVD<float>::StoreBestRanks)
        .def("StoreBestRanks", (void(ISVD<float>::*)(const std::string&, const std::vector<unsigned int>&)) &
                                   ISVD<float>::StoreBestRanks);

    // Factory func
    m.def("GetSVDInstance", &GetSVDInstance<float>);

    py::class_<AimetEqualization::CrossLayerScalingForPython>(m, "CrossLayerScaling");
    m.def("scaleLayerParams", &AimetEqualization::CrossLayerScalingForPython::scaleLayerParams);
    m.def("scaleDepthWiseSeparableLayer", &AimetEqualization::CrossLayerScalingForPython::scaleDepthWiseSeparableLayer);

    py::class_<AimetEqualization::CrossLayerScaling::RescalingParamsVectors>(m, "RescalingParamsVectors")
        .def(py::init<>())
        .def_readwrite("scalingMatrix12",
                       &AimetEqualization::CrossLayerScaling::RescalingParamsVectors::scalingMatrix12)
        .def_readwrite("scalingMatrix23",
                       &AimetEqualization::CrossLayerScaling::RescalingParamsVectors::scalingMatrix23);

    py::class_<AimetEqualization::EqualizationParamsForPython>(m, "EqualizationParams")
        .def(py::init<>())
        .def_readwrite("weightShape", &AimetEqualization::EqualizationParamsForPython::weightShape)
        .def_readwrite("weight", &AimetEqualization::EqualizationParamsForPython::weight)
        .def_readwrite("bias", &AimetEqualization::EqualizationParamsForPython::bias)
        .def_readwrite("isBiasNone", &AimetEqualization::EqualizationParamsForPython::isBiasNone);

    py::class_<AimetEqualization::BatchNormFoldForPython>(m, "BatchNormFold");
    m.def("fold", &AimetEqualization::BatchNormFoldForPython::fold);

    py::class_<AimetEqualization::BNParamsForPython>(m, "BNParams")
        .def(py::init<>())
        .def_readwrite("beta", &AimetEqualization::BNParamsForPython::beta)
        .def_readwrite("gamma", &AimetEqualization::BNParamsForPython::gamma)
        .def_readwrite("runningMean", &AimetEqualization::BNParamsForPython::runningMean)
        .def_readwrite("runningVar", &AimetEqualization::BNParamsForPython::runningVar);

    py::class_<AimetEqualization::TensorParamsForPython>(m, "TensorParams")
        .def(py::init<>())
        .def_readwrite("shape", &AimetEqualization::TensorParamsForPython::shape)
        .def_readwrite("data", &AimetEqualization::TensorParamsForPython::data);

    py::class_<AimetEqualization::HighBiasFoldForPython>(m, "HighBiasFold");
    m.def("updateBias", &AimetEqualization::HighBiasFoldForPython::updateBias);

    py::class_<AimetEqualization::LayerParamsForPython>(m, "LayerParams")
        .def(py::init<>())
        .def_readwrite("bias", &AimetEqualization::LayerParamsForPython::bias)
        .def_readwrite("weight", &AimetEqualization::LayerParamsForPython::weight)
        .def_readwrite("weightShape", &AimetEqualization::LayerParamsForPython::weightShape)
        .def_readwrite("activationIsRelu", &AimetEqualization::LayerParamsForPython::activationIsRelu);

    py::class_<AimetEqualization::BNParamsHighBiasFoldForPython>(m, "BNParamsHighBiasFold")
        .def(py::init<>())
        .def_readwrite("beta", &AimetEqualization::BNParamsHighBiasFoldForPython::beta)
        .def_readwrite("gamma", &AimetEqualization::BNParamsHighBiasFoldForPython::gamma);

    py::class_<AimetEqualization::TensorParamForPython>(m, "TensorParamBiasCorrection")
        .def(py::init<>())
        .def_readwrite("shape", &AimetEqualization::TensorParamForPython::shape)
        .def_readwrite("data", &AimetEqualization::TensorParamForPython::data);

    py::class_<AimetEqualization::BiasCorrectionForPython>(m, "BiasCorrection")
        .def(py::init<>())
        .def("correctBias", &AimetEqualization::BiasCorrectionForPython::correctBias)
        .def("storeQuantizedPreActivationOutput",
             &AimetEqualization::BiasCorrectionForPython::storeQuantizedPreActivationOutput)
        .def("storePreActivationOutput", &AimetEqualization::BiasCorrectionForPython::storePreActivationOutput);

    py::class_<AimetEqualization::BnBasedBiasCorrectionForPython>(m, "BnBasedBiasCorrection")
        .def(py::init<>())
        .def("correctBias", &AimetEqualization::BnBasedBiasCorrectionForPython::correctBias);

    py::class_<AimetEqualization::BnParamsBiasCorrForPython>(m, "BnParamsBiasCorr")
        .def(py::init<>())
        .def_readwrite("beta", &AimetEqualization::BnParamsBiasCorrForPython::beta)
        .def_readwrite("gamma", &AimetEqualization::BnParamsBiasCorrForPython::gamma);

    py::enum_<AimetEqualization::ActivationType>(m, "ActivationType")
        .value("relu", AimetEqualization::ActivationType::relu)
        .value("relu6", AimetEqualization::ActivationType::relu6)
        .value("noActivation", AimetEqualization::ActivationType::noActivation);

    m.def("getScaleFactor", &getScaleFactor);
    m.def("getRescaledOutputAndBias", &getRescaledOutputAndBias<float>);
}
