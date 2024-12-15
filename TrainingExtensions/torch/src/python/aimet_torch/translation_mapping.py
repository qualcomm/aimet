#@@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Mapping information for AIMET and backend"""
# pylint: disable=import-error, no-name-in-module
import torch
from aimet_common.libpymo import QnnDatatype
from aimet_common.defs import QuantizationDataType
from aimet_torch._base.nn.modules import custom

backend_datatype_to_aimet_map = {QnnDatatype.QNN_DATATYPE_INT_8: {'bitwidth': 8,
                                                                  'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_16: {'bitwidth': 16,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_32: {'bitwidth': 32,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_64: {'bitwidth': 64,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_8: {'bitwidth': 8,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_16: {'bitwidth': 16,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_32: {'bitwidth': 32,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_64: {'bitwidth': 64,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_FLOAT_16: {'bitwidth': 16,
                                                                     'dtype': QuantizationDataType.float},
                                 QnnDatatype.QNN_DATATYPE_FLOAT_32: {'bitwidth': 32,
                                                                     'dtype': QuantizationDataType.float},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_8: {'bitwidth': 8,
                                                                           'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_16: {'bitwidth': 16,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_32: {'bitwidth': 32,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_8: {'bitwidth': 8,
                                                                           'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_16: {'bitwidth': 16,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_32: {'bitwidth': 32,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_BOOL_8: {'bitwidth': 8,
                                                                   'dtype': QuantizationDataType.int}
                                 }
aimet_op_to_backend_op_name_map = {
    torch.nn.Conv1d: "Conv1d",
    torch.nn.Conv2d: "Conv2d",
    torch.nn.Conv3d: "Conv3d",
    torch.nn.ConvTranspose1d: "TransposeConv1d",
    torch.nn.ConvTranspose2d: "TransposeConv2d",
    torch.nn.ConvTranspose3d: "TransposeConv3d",
    torch.nn.ReLU: "Relu",
    torch.nn.Tanh: "Tanh",
    torch.nn.Sigmoid: "Sigmoid",
    torch.nn.ELU: "Elu",
    torch.nn.ReLU6: "Relu6",
    torch.nn.Hardtanh: "ReluMinMax",
    torch.nn.Hardswish: "HardSwish",
    custom.Add: "ElementWiseAdd",
    custom.Subtract: "ElementWiseSubtract",
    custom.Multiply: "ElementWiseMultiply",
    custom.Divide: "ElementWiseDivide",
    custom.Minimum: "ElementWiseMinimum",
    custom.Maximum: "ElementWiseMaximum",
    custom.Pow: "ElementWisePower",
    custom.Remainder: "ElementWiseMod",
    custom.Fmod: "ElementWiseFmod",
    custom.Exponential: "ElementWiseExp",
    custom.Log: "ElementWiseLog",
    custom.Sqrt: "ElementWiseRsqrt",
    custom.Abs: "ElementWiseAbs",
    custom.Neg: "ElementWiseNeg",
    custom.Erf: "Gelu",
    custom.Round: "ElementWiseRound",
    custom.Where: "ElementWiseSelect",
    custom.Equal: "ElementWiseEqual",
    custom.Greater: "ElementWiseGreater",
    custom.Less: "ElementWiseLess",
    custom.GreaterEqual: "ElementWiseGreaterEqual",
    custom.LessEqual: "ElementWiseLessEqual",
    custom.LogicalOr: "ElementWiseOr",
    custom.LogicalAnd: "ElementWiseAnd",
    custom.LogicalNot: "ElementWiseNot",
    custom.Mean: "ReduceMean",
    custom.Sum: "ReduceSum",
    custom.Prod: "ReduceProd",
    custom.ElementwiseCeil: "ElementWiseCeil",
    custom.ElementwiseFloor: "ElementWiseFloor",
    custom.Split: "Split",
    custom.Concat: "Concat",
    torch.nn.MaxPool2d: "PoolMax2d",
    custom.MaxPool2d: "PoolMax2d",
    torch.nn.MaxPool3d: "PoolMax3d",
    torch.nn.AvgPool2d: "PoolAvg2d",
    custom.AvgPool2d: "PoolAvg2d",
    torch.nn.AvgPool3d: "PoolAvg3d",
    torch.nn.LPPool2d: "L2Pool2d",
    custom.Reshape: "Reshape",
    custom.Permute: "Transpose",
    torch.nn.Upsample: "Resize",
    torch.nn.Linear: "FullyConnected",
    torch.nn.Softmax: "Softmax",
    torch.nn.LogSoftmax: "LogSoftmax",
    torch.nn.LayerNorm: "LayerNorm",
    torch.nn.Softplus: "ElementWiseSoftplus",
    torch.nn.PReLU: "Prelu",
    custom.CustomGather: "Gather",
    torch.nn.InstanceNorm1d: "InstanceNorm",
    torch.nn.InstanceNorm2d: "InstanceNorm",
    torch.nn.InstanceNorm3d: "InstanceNorm",
    custom.MatMul: "MatMul",
    custom.CumSum: "CumulativeSum",
    custom.Argmin: "Argmin",
    custom.Argmax: "Argmax",
    custom.Sin: "ElementWiseSin",
    custom.Cos: "ElementWiseCos",
    custom.Asin: "ElementWiseAsin",
    custom.Atan: "ElementWiseAtan",
    custom.Normalize: "L2Norm",
    custom.Gather: "Gather",
    torch.nn.ChannelShuffle: "ChannelShuffle",
    custom.ChannelShuffle: "ChannelShuffle",
    custom.Pad: "Pad",
    custom.ElementwiseUnarySign: "ElementWiseUnary",
    torch.nn.PixelShuffle: "DepthToSpace",
    custom.DepthToSpaceDCRMode: "DepthToSpace",
    torch.nn.PixelUnshuffle: "SpaceToDepth",
    custom.Min: "ReduceMin",
    custom.Max: "ReduceMax",
    custom.NonZero: "NonZero",
    custom.TopK: "TopK",
    custom.Shape: "Shape",
    custom.Tile: "Tile",
    torch.nn.LocalResponseNorm: "Lrn",
    torch.nn.LSTM: "Lstm",
    custom.ScatterND: "ScatterNd",
    custom.RoiAlign: "RoiAlign",
    custom.NonMaxSuppression: "NonMaxSuppression",
    custom.GatherNd: "GatherNd",
    torch.nn.BatchNorm1d: "Batchnorm",
    torch.nn.BatchNorm2d: "Batchnorm",
    torch.nn.BatchNorm3d: "Batchnorm",
    custom.OneHot: "OneHot",
    custom.ScatterElements: "ScatterElements",
    torch.nn.LeakyReLU: "Prelu",
    torch.nn.GRU: "Gru",
    custom.IndexSelect: "Gather",
    torch.nn.Embedding: "Gather",
    custom.Expand: "ElementWiseMultiply",
    custom.FloorDivide: "ElementWiseFloorDiv",
    torch.nn.GELU: "Gelu",
    custom.Cast: "Cast",
    custom.StridedSlice: "StridedSlice",
    torch.nn.GroupNorm: "GroupNorm",
    custom.GroupNorm: "GroupNorm",
}


op_to_weight_index_map = {'Conv1d' : 1,
                          'Conv2d': 1,
                          'Conv3d': 1,
                          'DepthWiseConv1d' : 1,
                          'DepthWiseConv2d': 1,
                          'TransposeConv1d' : 1,
                          'TransposeConv2d': 1,
                          'TransposeConv3d': 1,
                          'Batchnorm': 1,
                          'FullyConnected': 1,
                          'LayerNorm': 1,
                          'InstanceNorm': 1,
                          'GroupNorm': 1,
                          'MatMul': 1
                          }
