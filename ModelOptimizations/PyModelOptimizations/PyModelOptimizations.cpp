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
#include "DlQuantization/ParserModule.h"
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

    py::class_<ModelOpDefParser>(m, "ModelOpDefParser")
        .def(py::init<std::string, std::string>())
        .def_readwrite("op_list", &ModelOpDefParser::m_opList)
        .def_readonly("master_path", &ModelOpDefParser::m_masterPath)
        .def_readonly("backend_path", &ModelOpDefParser::m_backendPath)
        .def("get_supported_ops_in_backend", &ModelOpDefParser::getSupportedOpsInBackend,
             R"(Getting supported ops in backend XML)")
        .def("get_size", &ModelOpDefParser::getSize, py::arg("op_name"),
             R"(Extracting input, output and parameter sizes)")
        .def("get_filters_index", &ModelOpDefParser::getFiltersIndex, py::arg("op_name"),
             R"(Getting the index mapping to filter in inputs)")
        .def("get_input_datatype", &ModelOpDefParser::getInputDataType, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting input datatypes)")
        .def("get_output_datatype", &ModelOpDefParser::getOutputDataType, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting output datatypes)")
        .def("get_param_datatype", &ModelOpDefParser::getParamDataType, py::arg("op_name"), py::arg("attrib_name"),
             R"(Extracting parameter datatypes)")
        .def("get_input_rank", &ModelOpDefParser::getInputRank, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting input rank)")
        .def("get_output_rank", &ModelOpDefParser::getOutputRank, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting output rank)")
        .def("get_param_rank", &ModelOpDefParser::getParamRank, py::arg("op_name"), py::arg("attrib_name"),
             R"(Extracting parameter rank)")
        .def("get_input_multiflag", &ModelOpDefParser::getInputMultiFlag, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting input multi-flag)")
        .def("get_output_multiflag", &ModelOpDefParser::getOutputMultiFlag, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting output multi-flag)")
        .def("get_size_list", &ModelOpDefParser::getSizeList, py::arg("op_name"),
             R"(Extracting input, output and parameter sizes)")
        .def("get_input_datatype_list", &ModelOpDefParser::getInputDataTypeList, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting input datatype params)")
        .def("get_output_datatype_list", &ModelOpDefParser::getOutputDataTypeList, py::arg("op_name"), py::arg("attrib_num"),
             R"(Extracting output datatypes list)");

    py::enum_<QnnDatatype_t>(m, "QnnDatatype")
        .value("QNN_DATATYPE_INT_8", QnnDatatype_t::QNN_DATATYPE_INT_8)
        .value("QNN_DATATYPE_INT_16", QnnDatatype_t::QNN_DATATYPE_INT_16)
        .value("QNN_DATATYPE_INT_32", QnnDatatype_t::QNN_DATATYPE_INT_32)
        .value("QNN_DATATYPE_INT_64", QnnDatatype_t::QNN_DATATYPE_INT_64)

        .value("QNN_DATATYPE_UINT_8", QnnDatatype_t::QNN_DATATYPE_UINT_8)
        .value("QNN_DATATYPE_UINT_16", QnnDatatype_t::QNN_DATATYPE_UINT_16)
        .value("QNN_DATATYPE_UINT_32", QnnDatatype_t::QNN_DATATYPE_UINT_32)
        .value("QNN_DATATYPE_UINT_64", QnnDatatype_t::QNN_DATATYPE_UINT_64)

        .value("QNN_DATATYPE_FLOAT_16", QnnDatatype_t::QNN_DATATYPE_FLOAT_16)
        .value("QNN_DATATYPE_FLOAT_32", QnnDatatype_t::QNN_DATATYPE_FLOAT_32)

        .value("QNN_DATATYPE_SFIXED_POINT_8", QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_8)
        .value("QNN_DATATYPE_SFIXED_POINT_16", QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_16)
        .value("QNN_DATATYPE_SFIXED_POINT_32", QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_32)

        .value("QNN_DATATYPE_UFIXED_POINT_8", QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_8)
        .value("QNN_DATATYPE_UFIXED_POINT_16", QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_16)
        .value("QNN_DATATYPE_UFIXED_POINT_32", QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_32)

        .value("QNN_DATATYPE_BOOL_8", QnnDatatype_t::QNN_DATATYPE_BOOL_8)
        .value("QNN_DATATYPE_BACKEND_SPECIFIC", QnnDatatype_t::QNN_DATATYPE_BACKEND_SPECIFIC)
        .value("QNN_DATATYPE_UNDEFINED", QnnDatatype_t::QNN_DATATYPE_UNDEFINED);

    py::enum_<QnnRank_t>(m, "QnnRank")
        .value("QNN_SCALAR", QnnRank_t::QNN_SCALAR)
        .value("QNN_RANK_1", QnnRank_t::QNN_RANK_1)
        .value("QNN_RANK_2", QnnRank_t::QNN_RANK_2)
        .value("QNN_RANK_3", QnnRank_t::QNN_RANK_3)
        .value("QNN_RANK_4", QnnRank_t::QNN_RANK_4)
        .value("QNN_RANK_5", QnnRank_t::QNN_RANK_5)
        .value("QNN_RANK_N", QnnRank_t::QNN_RANK_N)
        .value("QNN_RANK_INVALID", QnnRank_t::QNN_RANK_INVALID);

    m.def("str_to_dtype", &strToDtype, py::arg("dtype"), R"(A function to convert string to QNN_DATATYPE)");

    m.def("str_to_rank", &strToRank, py::arg("rank"), R"(A function to convert string to QNN_RANK)");

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
