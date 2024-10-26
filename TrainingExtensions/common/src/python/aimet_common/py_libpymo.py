# /usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Creating an alias for function/classes/methods to use AIMET without MO library  """

import enum

IMPORT_ERROR: ImportError = ImportError()

def _error_message():
    return f'libpymo import failed with the following error:\n\n{IMPORT_ERROR}\n\n' \
           'Please check that libpymo has been built and is compatible with your ' \
           'current environment.' \

libpymo_classes = [
    'ModelOpDefParser',
    'TfEncoding',
    'Quantizer',
    'QuantizationEncodingAnalyzer',
    'LayerAttributes',
    'EncodingAnalyzerForPython',
    'TensorQuantizationSimForPython',
    'TensorQuantizer',
    'Svd',
    'CrossLayerScaling',
    'RescalingParamsVectors',
    'EqualizationParams',
    'BatchNormFold',
    'BNParams',
    'TensorParams',
    'HighBiasFold',
    'LayerParams',
    'BNParamsHighBiasFold',
    'TensorParamBiasCorrection',
    'BiasCorrection',
    'BnBasedBiasCorrection',
    'BnParamsBiasCorr',
]

libpymo_functions = [
    'str_to_dtype',
    'str_to_rank',
    'GetQuantizationInstance',
    'GetQuantizationEncodingAnalyzerInstance',
    'PtrToInt64',
    'GetSVDInstance',
    'scaleLayerParams',
    'scaleDepthWiseSeparableLayer',
    'fold',
    'updateBias',
    'getScaleFactor',
    'getRescaledOutputAndBias'
]


def create_unavailable_class(class_name: str):
    """
    Create unavailable class to lazily throw error when user tries to use the class.
    """
    class _MetaUnavailableClass(type):
        @classmethod
        def __getattr__(mcs, name):
            raise RuntimeError(f"Unable to access attribute {name} of class {class_name}: {_error_message()}")


    class _UnavailableClass(metaclass=_MetaUnavailableClass):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"Unable to initialize class {class_name}: {_error_message()}")

        def __getattr__(self, name):
            raise RuntimeError(f"Unable to access attribute {name} of class {class_name}: {_error_message()}")


    return type(class_name, (_UnavailableClass,), {})


for libpymo_class in libpymo_classes:
    globals()[libpymo_class] = create_unavailable_class(libpymo_class)


def create_unavailable_function(method_name: str):
    """
    Create unavailable function to lazily throw error when user tries to use the function.
    """
    def unavailable_function(*args, **kwargs):
        raise RuntimeError(f"Unable to run function {method_name}: {_error_message()}")

    return unavailable_function


for libpymo_function in libpymo_functions:
    globals()[libpymo_function] = create_unavailable_function(libpymo_function)


class COMPRESS_LAYER_TYPE(enum.Enum):
    """
    COMPRESS_LAYER_TYPE
    """
    LAYER_TYPE_OTHER = 0
    LAYER_TYPE_CONV = 1
    LAYER_TYPE_FC = 2


class ComputationMode(enum.Enum):
    """
    ComputationMode
    """
    COMP_MODE_CPU = 0
    COMP_MODE_GPU = 1


class LayerInOut(enum.Enum):
    """
    LayerInOut
    """
    LAYER_INPUT = 0
    LAYER_OUTPUT = 1


class NETWORK_COST_METRIC(enum.Enum):
    """
    NETWORK_COST_METRIC
    """
    COST_TYPE_MEMORY = 0
    COST_TYPE_MAC = 1


class QuantizationMode(enum.Enum):
    """
    QuantizationMode
    """
    QUANTIZATION_TF = 0
    QUANTIZATION_TF_ENHANCED = 1
    QUANTIZATION_RANGE_LEARNING = 2
    QUANTIZATION_PERCENTILE = 3
    QUANTIZATION_MSE = 4
    QUANTIZATION_ENTROPY = 5


class RoundingMode(enum.Enum):
    """
    RoundingMode
    """
    ROUND_NEAREST = 0
    ROUND_STOCHASTIC = 1


class SVD_COMPRESS_TYPE(enum.Enum):
    """
    SVD_COMPRESS_TYPE
    """
    TYPE_NONE = 0
    TYPE_SINGLE = 1
    TYPE_SUCCESSIVE = 2


class QnnDatatype(enum.Enum):
    """
    QnnDatatype
    """
    QNN_DATATYPE_INT_8 = 0
    QNN_DATATYPE_INT_16 = 1
    QNN_DATATYPE_INT_32 = 2
    QNN_DATATYPE_INT_64 = 3
    QNN_DATATYPE_UINT_8 = 4
    QNN_DATATYPE_UINT_16 = 5
    QNN_DATATYPE_UINT_32 = 6
    QNN_DATATYPE_UINT_64 = 7
    QNN_DATATYPE_FLOAT_16 = 8
    QNN_DATATYPE_FLOAT_32 = 9
    QNN_DATATYPE_SFIXED_POINT_8 = 10
    QNN_DATATYPE_SFIXED_POINT_16 = 11
    QNN_DATATYPE_SFIXED_POINT_32 = 12
    QNN_DATATYPE_UFIXED_POINT_8 = 13
    QNN_DATATYPE_UFIXED_POINT_16 = 14
    QNN_DATATYPE_UFIXED_POINT_32 = 15
    QNN_DATATYPE_BOOL_8 = 16
    QNN_DATATYPE_BACKEND_SPECIFIC = 17
    QNN_DATATYPE_UNDEFINED = 18


class TensorQuantizerOpMode(enum.Enum):
    """
    TensorQuantizerOpMode
    """
    updateStats = 0
    oneShotQuantizeDequantize = 1
    quantizeDequantize = 2
    passThrough = 3


class QnnRank(enum.Enum):
    """
    QnnRank
    """
    QNN_SCALAR = 0
    QNN_RANK_1 = 1
    QNN_RANK_2 = 2
    QNN_RANK_3 = 3
    QNN_RANK_4 = 4
    QNN_RANK_5 = 5
    QNN_RANK_N = 6
    QNN_RANK_INVALID = 7


class ActivationType(enum.Enum):
    """
    ActivationType
    """
    noActivation = 0
    relu = 1
    relu6 = 2


libpymo_enums = [
    QnnDatatype,
    QnnRank,
    ComputationMode,
    QuantizationMode,
    LayerInOut,
    RoundingMode,
    COMPRESS_LAYER_TYPE,
    NETWORK_COST_METRIC,
    SVD_COMPRESS_TYPE,
    TensorQuantizerOpMode,
    ActivationType,
]


for libpymo_enum in libpymo_enums:
    globals().update(libpymo_enum.__members__)
