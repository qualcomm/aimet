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
""" Quantized definitions for custom modules of AIMET """

import copy
from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from aimet_torch._base.nn.modules.custom import * # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from ..true_quant import (
    QuantizationMixin,
    _DispatchMixin,
    _quantize_if_applicable,
    _quantize_dequantize_if_applicable,
)


def _binary_quant_init(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([None, None])


def _ternary_quant_init(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([None, None, None])


@QuantizationMixin.implements(Sin)
class QuantizedSin(_DispatchMixin, QuantizationMixin, Sin):
    """ Quantized Sin """
    _builtin_torch_fn = torch.sin


@QuantizationMixin.implements(Cos)
class QuantizedCos(_DispatchMixin, QuantizationMixin, Cos):
    """ Quantized Cos """
    _builtin_torch_fn = torch.cos


@QuantizationMixin.implements(AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, AvgPool2d):
    """ Quantized AvgPool2d """
    _builtin_torch_fn = F.avg_pool2d


@QuantizationMixin.implements(Reshape)
class QuantizedReshape(_DispatchMixin, QuantizationMixin, Reshape):
    """ Quantized Reshape """
    _builtin_torch_fn = torch.reshape


@QuantizationMixin.implements(RSqrt)
class QuantizedRSqrt(_DispatchMixin, QuantizationMixin, RSqrt):
    """ Quantized RSqrt """
    _builtin_torch_fn = torch.rsqrt


@QuantizationMixin.implements(MatMul)
class QuantizedMatMul(_DispatchMixin, QuantizationMixin, MatMul):
    """ Quantized MatMul """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.matmul


@QuantizationMixin.implements(Add)
class QuantizedAdd(_DispatchMixin, QuantizationMixin, Add):
    """ Quantized Add """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.add


@QuantizationMixin.implements(Multiply)
class QuantizedMultiply(_DispatchMixin, QuantizationMixin, Multiply):
    """ Quantized Multiply """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.mul


@QuantizationMixin.implements(Subtract)
class QuantizedSubtract(_DispatchMixin, QuantizationMixin, Subtract):
    """ Quantized Subtract """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.sub


@QuantizationMixin.implements(Divide)
class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    """ Quantized Divide """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.div


@QuantizationMixin.implements(Concat)
class QuantizedConcat(_DispatchMixin, QuantizationMixin, Concat):
    """ Quantized Concat """
    _builtin_torch_fn = torch.cat

    # pylint: disable=attribute-defined-outside-init
    def __quant_init__(self):
        super().__quant_init__()
        self._num_inputs = 1

    def export_input_encodings(self, encoding_version: str):
        """
        Extends super().export to repeat input quantizer's encodings :attr:`self._num_inputs` times
        """
        input_encodings = super().export_input_encodings(encoding_version)
        # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
        return [copy.deepcopy(encoding) for encoding in input_encodings * self._num_inputs]

    def import_input_encodings(self,
                               encodings,
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Extends super().import_input_encodings to set `self._num_inputs` based on length of encodings.
        """
        self._num_inputs = len(encodings)
        super().import_input_encodings(encodings,
                                       strict=strict,
                                       partial=partial,
                                       requires_grad=requires_grad,
                                       allow_overwrite=allow_overwrite)

    def forward(self, *x): # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.Concat.
        """
        self._num_inputs = len(x)
        return super().forward(*x)

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_dequantize_if_applicable(x, input_qtzr) for x in tensors)
            output = fn(tensors, dim=dim, out=out)
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return cat

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_if_applicable(x, input_qtzr) for x in tensors)
            output_encodings = self.output_quantizers[0].get_encodings() if self.output_quantizers[0] else None
            return fn(tensors, dim=dim, out=out, output_encodings=output_encodings)

        return cat


# @QuantizationMixin.implements(FloorDivide)
# class QuantizedFloorDivide(_DispatchMixin, QuantizationMixin, FloorDivide):
#     """ Quantized FloorDivide """
#     _builtin_torch_fn = torch.floor_divide
#
#
@QuantizationMixin.implements(Norm)
class QuantizedNorm(_DispatchMixin, QuantizationMixin, Norm):
    """ Quantized Norm """
    _builtin_torch_fn = torch.norm


@QuantizationMixin.implements(Exponential)
class QuantizedExponential(_DispatchMixin, QuantizationMixin, Exponential):
    """ Quantized Exponential """
    _builtin_torch_fn = torch.exp


@QuantizationMixin.implements(Erf)
class QuantizedErf(_DispatchMixin, QuantizationMixin, Erf):
    """ Quantized Erf """
    _builtin_torch_fn = torch.erf


@QuantizationMixin.implements(Sqrt)
class QuantizedSqrt(_DispatchMixin, QuantizationMixin, Sqrt):
    """ Quantized Sqrt """
    _builtin_torch_fn = torch.sqrt
#
#
# @QuantizationMixin.implements(Maximum)
# class QuantizedMaximum(_DispatchMixin, QuantizationMixin, Maximum):
#     """ Quantized Maximum """
#     _builtin_torch_fn = torch.maximum
#
#
# @QuantizationMixin.implements(Max)
# class QuantizedMax(_DispatchMixin, QuantizationMixin, Max):
#     """ Quantized Max """
#     _builtin_torch_fn = torch.max
#
# @QuantizationMixin.implements(AMax)
# class QuantizedAMax(_DispatchMixin, QuantizationMixin, AMax):
#     """ Quantized AMax """
#     _builtin_torch_fn = torch.amax
#
#
# @QuantizationMixin.implements(Minimum)
# class QuantizedMinimum(_DispatchMixin, QuantizationMixin, Minimum):
#     """ Quantized Minimum """
#     _builtin_torch_fn = torch.minimum
#
#
# @QuantizationMixin.implements(Min)
# class QuantizedMin(_DispatchMixin, QuantizationMixin, Min):
#     """ Quantized Min """
#     _builtin_torch_fn = torch.min
#
# @QuantizationMixin.implements(AMin)
# class QuantizedAMin(_DispatchMixin, QuantizationMixin, AMin):
#     """ Quantized AMin """
#     _builtin_torch_fn = torch.amin
#
#
# @QuantizationMixin.implements(Where)
# class QuantizedWhere(_DispatchMixin, QuantizationMixin, Where):
#     """ Quantized Where """
#     _builtin_torch_fn = torch.where
#
#
# @QuantizationMixin.implements(Greater)
# class QuantizedGreater(_DispatchMixin, QuantizationMixin, Greater):
#     """ Quantized Greater """
#     _builtin_torch_fn = torch.gt
#
#
# @QuantizationMixin.implements(Less)
# class QuantizedLess(_DispatchMixin, QuantizationMixin, Less):
#     """ Quantized Less """
#     _builtin_torch_fn = torch.lt
#
#
# @QuantizationMixin.implements(GreaterEqual)
# class QuantizedGreaterEqual(_DispatchMixin, QuantizationMixin, GreaterEqual):
#     """ Quantized GreaterEqual """
#     _builtin_torch_fn = torch.ge
#
#
# @QuantizationMixin.implements(LessEqual)
# class QuantizedLessEqual(_DispatchMixin, QuantizationMixin, LessEqual):
#     """ Quantized LessEqual """
#     _builtin_torch_fn = torch.le
#
#
# @QuantizationMixin.implements(NotEqual)
# class QuantizedNotEqual(_DispatchMixin, QuantizationMixin, NotEqual):
#     """ Quantized NotEqual """
#     _builtin_torch_fn = torch.ne
#
#
# @QuantizationMixin.implements(Equal)
# class QuantizedEqual(_DispatchMixin, QuantizationMixin, Equal):
#     """ Quantized Equal """
#     _builtin_torch_fn = torch.eq


@QuantizationMixin.implements(Bmm)
class QuantizedBmm(_DispatchMixin, QuantizationMixin, Bmm):
    """ Quantized Bmm """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.bmm


@QuantizationMixin.implements(CumSum)
class QuantizedCumSum(_DispatchMixin, QuantizationMixin, CumSum):
    """ Quantized CumSum """
    _builtin_torch_fn = torch.cumsum


# @QuantizationMixin.implements(MaskedFill)
# class QuantizedMaskedFill(_DispatchMixin, QuantizationMixin, MaskedFill):
#     """ Quantized MaskedFill """
#     _builtin_torch_fn = torch.Tensor.masked_fill_
#
#
# @QuantizationMixin.implements(Mean)
# class QuantizedMean(_DispatchMixin, QuantizationMixin, Mean):
#     """ Quantized Mean """
#     _builtin_torch_fn = torch.mean
#
#
# @QuantizationMixin.implements(Sum)
# class QuantizedSum(_DispatchMixin, QuantizationMixin, Sum):
#     """ Quantized Sum """
#     _builtin_torch_fn = torch.sum
#
#
# @QuantizationMixin.implements(Prod)
# class QuantizedProd(_DispatchMixin, QuantizationMixin, Prod):
#     """ Quantized Prod """
#     _builtin_torch_fn = torch.prod
#
#
@QuantizationMixin.implements(Log)
class QuantizedLog(_DispatchMixin, QuantizationMixin, Log):
    """ Quantized Log """
    _builtin_torch_fn = torch.log


@QuantizationMixin.implements(Abs)
class QuantizedAbs(_DispatchMixin, QuantizationMixin, Abs):
    """ Quantized Abs """
    _builtin_torch_fn = torch.abs


@QuantizationMixin.implements(Neg)
class QuantizedNeg(_DispatchMixin, QuantizationMixin, Neg):
    """ Quantized Neg """
    _builtin_torch_fn = torch.neg
#
#
# @QuantizationMixin.implements(Argmin)
# class QuantizedArgmin(_DispatchMixin, QuantizationMixin, Argmin):
#     """ Quantized Argmin """
#     _builtin_torch_fn = torch.argmin
#
#
# @QuantizationMixin.implements(Argmax)
# class QuantizedArgmax(_DispatchMixin, QuantizationMixin, Argmax):
#     """ Quantized Argmax """
#     _builtin_torch_fn = torch.argmax
#
#
# @QuantizationMixin.implements(ElementwiseCeil)
# class QuantizedElementwiseCeil(_DispatchMixin, QuantizationMixin, ElementwiseCeil):
#     """ Quantized ElementwiseCeil """
#     _builtin_torch_fn = torch.ceil
#
#
# @QuantizationMixin.implements(ElementwiseFloor)
# class QuantizedElementwiseFloor(_DispatchMixin, QuantizationMixin, ElementwiseFloor):
#     """ Quantized ElementwiseFloor """
#     _builtin_torch_fn = torch.floor
#
#
# @QuantizationMixin.implements(Asin)
# class QuantizedAsin(_DispatchMixin, QuantizationMixin, Asin):
#     """ Quantized Asin """
#     _builtin_torch_fn = torch.asin
#
#
# @QuantizationMixin.implements(Atan)
# class QuantizedAtan(_DispatchMixin, QuantizationMixin, Atan):
#     """ Quantized Atan """
#     _builtin_torch_fn = torch.atan
#
#
# @QuantizationMixin.implements(Round)
# class QuantizedRound(_DispatchMixin, QuantizationMixin, Round):
#     """ Quantized Round """
#     _builtin_torch_fn = torch.round
#
#
# @QuantizationMixin.implements(Gather)
# class QuantizedGather(_DispatchMixin, QuantizationMixin, Gather):
#     """ Quantized Gather """
#     _builtin_torch_fn = torch.gather
#
#
# @QuantizationMixin.implements(LogicalOr)
# class QuantizedLogicalOr(_DispatchMixin, QuantizationMixin, LogicalOr):
#     """ Quantized LogicalOr """
#     _builtin_torch_fn = torch.logical_or
#
#
# @QuantizationMixin.implements(LogicalAnd)
# class QuantizedLogicalAnd(_DispatchMixin, QuantizationMixin, LogicalAnd):
#     """ Quantized LogicalAnd """
#     _builtin_torch_fn = torch.logical_and
#
#
# @QuantizationMixin.implements(LogicalNot)
# class QuantizedLogicalNot(_DispatchMixin, QuantizationMixin, LogicalNot):
#     """ Quantized LogicalNot """
#     _builtin_torch_fn = torch.logical_not
#
#
# @QuantizationMixin.implements(Split)
# class QuantizedSplit(_DispatchMixin, QuantizationMixin, Split):
#     """ Quantized Split """
#     _builtin_torch_fn = torch.split
#
#
# @QuantizationMixin.implements(Permute)
# class QuantizedPermute(_DispatchMixin, QuantizationMixin, Permute):
#     """ Quantized Permute """
#     _builtin_torch_fn = torch.permute
#
#
# @QuantizationMixin.implements(Remainder)
# class QuantizedRemainder(_DispatchMixin, QuantizationMixin, Remainder):
#     """ Quantized Remainder """
#     _builtin_torch_fn = torch.remainder
#
#
# @QuantizationMixin.implements(IndexSelect)
# class QuantizedIndexSelect(_DispatchMixin, QuantizationMixin, IndexSelect):
#     """ Quantized IndexSelect """
#     _builtin_torch_fn = torch.index_select
#
#
# @QuantizationMixin.implements(Fmod)
# class QuantizedFmod(_DispatchMixin, QuantizationMixin, Fmod):
#     """ Quantized Fmod """
#     _builtin_torch_fn = torch.fmod
#
#
# @QuantizationMixin.implements(NonZero)
# class QuantizedNonZero(_DispatchMixin, QuantizationMixin, NonZero):
#     """ Quantized NonZero """
#     _builtin_torch_fn = torch.nonzero
#
#
# @QuantizationMixin.implements(TopK)
# class QuantizedTopK(_DispatchMixin, QuantizationMixin, TopK):
#     """ Quantized TopK """
#     _builtin_torch_fn = torch.topk
#
#
# @QuantizationMixin.implements(Shape)
# class QuantizedShape(_DispatchMixin, QuantizationMixin, Shape):
#     """ Quantized Shape """
#     _builtin_torch_fn = torch.Tensor.size
#
#
# @QuantizationMixin.implements(Tile)
# class QuantizedTile(_DispatchMixin, QuantizationMixin, Tile):
#     """ Quantized Tile """
#     _builtin_torch_fn = torch.tile

# @QuantizationMixin.implements(ElementwiseUnarySign)
# class QuantizedElementwiseUnarySign(_DispatchMixin, QuantizationMixin, ElementwiseUnarySign):
#     """ Quantized ElementwiseUnarySign """
#     _builtin_torch_fn = torch.sign


@QuantizationMixin.implements(Baddbmm)
class QuantizedBaddbmm(_DispatchMixin, QuantizationMixin, Baddbmm):
    """ Quantized Baddbmm """
    __quant_init__ = _ternary_quant_init
    _builtin_torch_fn = torch.baddbmm


@QuantizationMixin.implements(Addmm)
class QuantizedAddmm(_DispatchMixin, QuantizationMixin, Addmm):
    """ Quantized Addmm """
    __quant_init__ = _ternary_quant_init
    _builtin_torch_fn = torch.addmm


@QuantizationMixin.implements(RmsNorm)
class QuantizedRmsNorm(QuantizationMixin, RmsNorm):
    """Custom module for RmsNorm"""
    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RmsNorm
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


# @QuantizationMixin.implements(Square)
# class QuantizedSquare(_DispatchMixin, QuantizationMixin, Square):
#     """ Quantized Square """
#     _builtin_torch_fn = torch.square
#
#
# @QuantizationMixin.implements(Select)
# class QuantizedSelect(_DispatchMixin, QuantizationMixin, Select):
#     """ Quantized Select """
#     _builtin_torch_fn = torch.select
#
#
#
# # modules for functional operations defined under torch.nn.functional package
# @QuantizationMixin.implements(Interpolate)
# class QuantizedInterpolate(_DispatchMixin, QuantizationMixin, Interpolate):
#     """ Quantized Interpolate """
#     _builtin_torch_fn = torch.nn.functional.interpolate
#
#
# @QuantizationMixin.implements(MaxPool2d)
# class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, MaxPool2d):
#     """ Quantized MaxPool2d """
#     _builtin_torch_fn = torch.nn.functional.max_pool2d
#
#
# @QuantizationMixin.implements(AdaptiveAvgPool2d)
# class QuantizedAdaptiveAvgPool2d(_DispatchMixin, QuantizationMixin, AdaptiveAvgPool2d):
#     """ Quantized AdaptiveAvgPool2d """
#     _builtin_torch_fn = torch.nn.functional.adaptive_avg_pool2d
#
#
@QuantizationMixin.implements(BatchNorm)
class QuantizedBatchNorm(_DispatchMixin, QuantizationMixin, BatchNorm):
    """ Quantized BatchNorm """
    _builtin_torch_fn = torch.nn.functional.batch_norm

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None, None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if self.input_quantizers[1] is not None or self.input_quantizers[2] is not None:
                    raise RuntimeError(f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode")

            input = _quantize_dequantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_dequantize_if_applicable(running_mean, self.input_quantizers[1])
            running_var = _quantize_dequantize_if_applicable(running_var, self.input_quantizers[2])
            weight = _quantize_dequantize_if_applicable(weight, self.input_quantizers[3])
            bias = _quantize_dequantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(input, running_mean.detach(), running_var.detach(),
                        weight, bias, training, momentum, eps)

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper

    def _custom_kernel_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if self.input_quantizers[1] is not None or self.input_quantizers[2] is not None:
                    raise RuntimeError(f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode")

            input = _quantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_if_applicable(running_mean, self.input_quantizers[1])
            running_var = _quantize_if_applicable(running_var, self.input_quantizers[2])
            weight = _quantize_if_applicable(weight, self.input_quantizers[3])
            bias = _quantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(input, running_mean.detach(), running_var.detach(),
                        weight, bias, training, momentum, eps)
            return _quantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper
#
#
@QuantizationMixin.implements(GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, GroupNorm):
    """ Quantized GroupNorm """
    _builtin_torch_fn = torch.nn.functional.group_norm
#
#
@QuantizationMixin.implements(Normalize)
class QuantizedNormalize(_DispatchMixin, QuantizationMixin, Normalize):
    """ Quantized Normalize """
    _builtin_torch_fn = torch.nn.functional.normalize
#
#
# @QuantizationMixin.implements(Pad)
# class QuantizedPad(_DispatchMixin, QuantizationMixin, Pad):
#     """ Quantized Pad """
#     _builtin_torch_fn = torch.nn.functional.pad
#
#
# @QuantizationMixin.implements(GridSample)
# class QuantizedGridSample(_DispatchMixin, QuantizationMixin, GridSample):
#     """ Quantized GridSample """
#     _builtin_torch_fn = torch.nn.functional.grid_sample
#
#
# @QuantizationMixin.implements(DynamicConv2d)
# class QuantizedDynamicConv2d(QuantizationMixin, DynamicConv2d):
#     """ Quantized DynamicConv2d """
#
#
# @QuantizationMixin.implements(Pow)
# class QuantizedPow(QuantizationMixin, Pow):
#     """ Quantized Pow """
#
#
# @QuantizationMixin.implements(CustomSiLU)
# class QuantizedCustomSiLU(QuantizationMixin, CustomSiLU):
#     """ Quantized CustomSiLU """
#
#
# @QuantizationMixin.implements(StridedSlice)
# class QuantizedStridedSlice(QuantizationMixin, StridedSlice):
#     """ Quantized StridedSlice """
#
#
# @QuantizationMixin.implements(ChannelShuffle)
# class QuantizedChannelShuffle(QuantizationMixin, ChannelShuffle):
#     """ Quantized ChannelShuffle """
#
#
# @QuantizationMixin.implements(Cast)
# class QuantizedCast(QuantizationMixin, Cast):
#     """ Quantized Cast """
#
#
# @QuantizationMixin.implements(CustomGather)
# class QuantizedCustomGather(QuantizationMixin, CustomGather):
#     """ Quantized CustomGather """
#
#
# @QuantizationMixin.implements(DepthToSpaceCRDMode)
# class QuantizedDepthToSpaceCRDMode(QuantizationMixin, DepthToSpaceCRDMode):
#     """ Quantized DepthToSpaceCRDMode """
#
#
# @QuantizationMixin.implements(DepthToSpaceDCRMode)
# class QuantizedDepthToSpaceDCRMode(QuantizationMixin, DepthToSpaceDCRMode):
#     """ Quantized DepthToSpaceDCRMode """
#
#
# @QuantizationMixin.implements(CustomSparseConv3DLayer)
# class QuantizedCustomSparseConv3DLayer(QuantizationMixin, CustomSparseConv3DLayer):
#     """ Quantized CustomSparseConv3DLayer """
#
#
# @QuantizationMixin.implements(SparseTensorWrapper)
# class QuantizedSparseTensorWrapper(QuantizationMixin, SparseTensorWrapper):
#     """ Quantized SparseTensorWrapper """
#
#
# @QuantizationMixin.implements(ScatterDense)
# class QuantizedScatterDense(QuantizationMixin, ScatterDense):
#     """ Quantized ScatterDense """
#
#
# @QuantizationMixin.implements(ScatterND)
# class QuantizedScatterND(QuantizationMixin, ScatterND):
#     """ Quantized ScatterND """
#
#
# @QuantizationMixin.implements(RoiAlign)
# class QuantizedRoiAlign(QuantizationMixin, RoiAlign):
#     """ Quantized RoiAlign """
#
#
# @QuantizationMixin.implements(NonMaxSuppression)
# class QuantizedNonMaxSuppression(QuantizationMixin, NonMaxSuppression):
#     """ Quantized NonMaxSuppression """
#
#
# @QuantizationMixin.implements(GatherNd)
# class QuantizedGatherNd(QuantizationMixin, GatherNd):
#     """ Quantized GatherNd """
#
#
# @QuantizationMixin.implements(ScatterElements)
# class QuantizedScatterElements(QuantizationMixin, ScatterElements):
#     """ Quantized ScatterElements """
#
#
# @QuantizationMixin.implements(OneHot)
# class QuantizedOneHot(QuantizationMixin, OneHot):
#     """ Quantized OneHot """
#
#
# @QuantizationMixin.implements(Expand)
# class QuantizedExpand(QuantizationMixin, Expand):
#     """ Quantized Expand """
#
#
# @QuantizationMixin.implements(DynamicLinear)
# class QuantizedDynamicLinear(QuantizationMixin, DynamicLinear):
#     """ Quantized DynamicLinear """
