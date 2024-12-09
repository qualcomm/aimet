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
# pylint: disable=too-many-lines, wrong-import-order
"""Fake-quantized modules (deprecated)"""

from packaging import version
from collections import OrderedDict
from typing import Type, Optional, Tuple
import abc
import warnings

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.adaptive import _ASMoutput
from torch.nn.utils.rnn import PackedSequence
from torch.utils._pytree import tree_map

from ..base import BaseQuantizationMixin # pylint: disable=import-error
from ..modules import custom # pylint: disable=import-error


class FakeQuantMeta(abc.ABCMeta):
    """Sets :meth:`forward` to :meth:`quantized_forward` if only :meth:`quantized_forward` is defined
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        if "quantized_forward" in namespace and "forward" not in namespace:
            warnings.warn("Support for defining `quantized_forward` in place of `forward` method will be deprecated, "
                          "please use `forward` instead.",
                          DeprecationWarning, stacklevel=2)
            namespace["forward"] = namespace["quantized_forward"]
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class FakeQuantizationMixin(BaseQuantizationMixin, metaclass=FakeQuantMeta): # pylint: disable=abstract-method
    """Mixin that implements fake-quantization on top of regular pytorch modules.

    Specifically, a fake-quantized module will quantize input, output, and parameter tensors with
    its held :class:`QuantizerBase` objects during the :meth:`forward` method and use the inherited :class:torch.nn.Module`
    forward method to compute the layer operation. If all input, output, and parameter quantizers are ``None``, a
    fake-quantized module will behave exactly the same as its parent :class:`torch.nn.Module`.

    A fake-quantized module can be initialized from scratch using the same syntax as the parent module, or can be
    formed from an existing module using the :meth:`from_module` method.

    Attributes:
        input_quantizers (nn.ModuleList): :class:`ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's input tensors
        output_quantizers (nn.ModuleList): :class:`ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's output tensors
        param_quantizers (nn.ModuleDict): :class:`ModuleDict` mapping parameter names to associated :class:`QuantizerBase`
            objects

    Examples:

        >>> qlinear = FakeQuantizedLinear(in_features=10, out_features=20, bias=False)
        >>> print(qlinear)
        FakeQuantizedLinear(
          in_features=10, out_features=20, bias=False
          (param_quantizers): ModuleDict(
            (weight): None
          )
          (input_quantizers): ModuleList(
            (0): None
          )
          (output_quantizers): ModuleList(
            (0): None
          )
        )


        >>> linear = torch.nn.Linear(in_features=10, out_features=20, bias=True)
        >>> qlinear = FakeQuantizationMixin.from_module(linear)
        >>> print(qlinear)
        FakeQuantizedLinear(
          in_features=10, out_features=20, bias=True
          (param_quantizers): ModuleDict(
            (weight): None
            (bias): None
          )
          (input_quantizers): ModuleList(
            (0): None
          )
          (output_quantizers): ModuleList(
            (0): None
          )
        )
        >>> qlinear.weight is linear.weight
        True

    """

    cls_to_qcls = OrderedDict() # ouantized class -> original class
    qcls_to_cls = OrderedDict() # original class -> quantized class

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Computes a fake-quantized version of the parent module's forward method.

        The :meth:`forward` method should perform the following logic in order:

            1) Apply existing input quantizers to input tensors
            2) Apply existing param quantizers to the layer's parameters
            3) Call the inherited :class:`torch.nn.Module` forward method with quantized inputs and parameters
            4) Apply existing output quantizers to the outputs of the forward method

        If all input, output, and parameter quantizers are ``None``, this method will behave exactly the same as
        its parent module's forward pass.
        """
        return super().forward(*args, **kwargs)

    @classmethod
    def wrap(cls, module_cls: Type[nn.Module]) -> Type[nn.Module]:
        """
        Wrap a regular module class into a fake-quantized module class
        """
        if not issubclass(module_cls, nn.Module):
            raise ValueError("Expected module_cls to be a subclass of torch.nn.Module. "
                             f"Got {module_cls}.")
        if module_cls in cls.cls_to_qcls:
            return cls.cls_to_qcls[module_cls]

        quantized_cls_name = f"FakeQuantized{module_cls.__name__}"
        base_classes = (cls, module_cls)
        quantized_cls = type(quantized_cls_name, base_classes, {'__module__': __name__})
        return cls.implements(module_cls)(quantized_cls)


class _FakeQuantizedUnaryOpMixin(FakeQuantizationMixin): # pylint: disable=abstract-method
    def forward(self, *args, **kwargs) -> Tensor: # pylint: disable=missing-function-docstring
        x, *others = args

        if isinstance(x, Tensor) and x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            output = super().forward(x, *others, **kwargs)

        if isinstance(output, Tensor) and output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output

class _FakeQuantizedBinaryOpMixin(FakeQuantizationMixin): # pylint: disable=abstract-method
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])

    def forward(self, *args, **kwargs) -> Tensor: # pylint: disable=missing-function-docstring
        x, y, *others = args

        if isinstance(x, Tensor) and x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        if isinstance(y, Tensor) and y.is_floating_point() and self.input_quantizers[1]:
            y = self.input_quantizers[1](y)

        with self._patch_quantized_parameters():
            output = super().forward(x, y, *others, **kwargs)

        if isinstance(output, Tensor) and output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output

class _FakeQuantizedTernaryOpMixin(FakeQuantizationMixin): # pylint: disable=abstract-method
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None, None])

    def forward(self, *args, **kwargs) -> Tensor: # pylint: disable=missing-function-docstring
        x, y, z, *others = args

        if isinstance(x, Tensor) and x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        if isinstance(y, Tensor) and y.is_floating_point() and self.input_quantizers[1]:
            y = self.input_quantizers[1](y)

        if isinstance(z, Tensor) and z.is_floating_point() and self.input_quantizers[2]:
            z = self.input_quantizers[2](z)

        with self._patch_quantized_parameters():
            output = super().forward(x, y, z, *others, **kwargs)

        if isinstance(output, Tensor) and output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output



########################
### torch.nn.Modules ###
########################

# Below are the lists of modules with regular code patterns
# that takes tensors as the first N arguments and returns single tensor as output
_TORCH_NN_UNARY_MODULES = [
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.AlphaDropout,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.CELU,
    nn.ChannelShuffle,
    nn.ConstantPad1d,
    nn.ConstantPad2d,
    nn.ConstantPad3d,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.CrossMapLRN2d,
    nn.CTCLoss,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.ELU,
    nn.FeatureAlphaDropout,
    nn.Flatten,
    nn.Fold,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
    nn.GELU,
    nn.GLU,
    nn.GroupNorm,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Hardtanh,
    nn.HingeEmbeddingLoss,
    nn.Identity,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LPPool1d,
    nn.LPPool2d,
    nn.LayerNorm,
    nn.LeakyReLU,
    nn.Linear,
    nn.LocalResponseNorm,
    nn.LogSigmoid,
    nn.LogSoftmax,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.MaxUnpool1d,
    nn.MaxUnpool2d,
    nn.MaxUnpool3d,
    nn.Mish,
    nn.MultiLabelMarginLoss,
    nn.MultiLabelSoftMarginLoss,
    nn.MultiMarginLoss,
    nn.NLLLoss,
    nn.NLLLoss2d,
    nn.PReLU,
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    nn.RReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ReplicationPad3d,
    nn.SELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Softmax,
    nn.Softmax2d,
    nn.Softmin,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.SoftMarginLoss,
    nn.SyncBatchNorm,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
    nn.Unflatten,
    nn.Unfold,
    nn.Upsample,
    nn.UpsamplingBilinear2d,
    nn.UpsamplingNearest2d,
    nn.ZeroPad2d,
]

if version.parse(torch.__version__) >= version.parse("1.10.0"):
    _TORCH_NN_UNARY_MODULES.extend([
        nn.ReflectionPad3d,
    ])

if version.parse(torch.__version__) >= version.parse("1.12.0"):
    _TORCH_NN_UNARY_MODULES.extend([
        nn.Dropout1d,
    ])

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    _TORCH_NN_UNARY_MODULES.extend([
        nn.CircularPad1d,
        nn.CircularPad2d,
        nn.CircularPad3d,
        nn.ZeroPad1d,
        nn.ZeroPad3d,
    ])

_TORCH_NN_BINARY_MODULES = [
    nn.BCELoss,
    nn.BCEWithLogitsLoss,
    nn.Bilinear,
    nn.CosineEmbeddingLoss,
    nn.CosineSimilarity,
    nn.CrossEntropyLoss,
    nn.HuberLoss,
    nn.KLDivLoss,
    nn.L1Loss,
    nn.MSELoss,
    nn.MarginRankingLoss,
    nn.PairwiseDistance,
    nn.PoissonNLLLoss,
    nn.SmoothL1Loss,
]
_TORCH_NN_TERNARY_MODULES = [
    nn.GaussianNLLLoss,
    nn.TripletMarginLoss,
    nn.TripletMarginWithDistanceLoss,
]


def _register_global_variable(var_name, obj):
    if var_name in globals():
        raise RuntimeError(f'Variable name "{var_name}" already exists in the global namespace.')
    globals().update({var_name: obj})


# Auto-generate quantized module definitions for regular-patterned modules
for _module_cls in _TORCH_NN_UNARY_MODULES:
    _quantized_cls = _FakeQuantizedUnaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)

for _module_cls in _TORCH_NN_BINARY_MODULES:
    _quantized_cls = _FakeQuantizedBinaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)

for _module_cls in _TORCH_NN_TERNARY_MODULES:
    _quantized_cls = _FakeQuantizedTernaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)


@FakeQuantizationMixin.implements(nn.Embedding)
class FakeQuantizedEmbedding(FakeQuantizationMixin, nn.Embedding):
    """
    Quantized class definition for nn.Embedding.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([]) # nn.Embedding takes no float input
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, input: Tensor) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.Embedding.
        """
        # pylint: disable=redefined-builtin

        with self._patch_quantized_parameters():
            output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(nn.EmbeddingBag)
class FakeQuantizedEmbeddingBag(FakeQuantizationMixin, nn.EmbeddingBag):
    """
    Quantized class definition for nn.EmbeddingBag.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, # pylint: disable=arguments-differ
                input: Tensor,
                offsets: Optional[Tensor] = None,
                per_sample_weights: Optional[Tensor] = None) -> Tensor:
        """
        Quantized forward impl for nn.EmbeddingBag.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0] and per_sample_weights is not None:
            per_sample_weights = self.input_quantizers[0](per_sample_weights)

        with self._patch_quantized_parameters():
            output = super().forward(input, offsets, per_sample_weights)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


class _FakeQuantizedRNNBaseMixin(FakeQuantizationMixin):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def forward(self, input, hx: Optional[Tensor] = None): # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.GRU and nn.RNN.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            if isinstance(input, PackedSequence):
                data, *others = input
                quantized_data = self.input_quantizers[0](data)
                input = PackedSequence(quantized_data, *others)
            else:
                input = self.input_quantizers[0](input)

        if hx is not None and self.input_quantizers[1]:
            hx = self.input_quantizers[1](hx)

        with self._patch_quantized_parameters():
            output, hidden = super().forward(input, hx)

        if self.output_quantizers[0]:
            if isinstance(output, PackedSequence):
                data, *others = output
                quantized_data = self.output_quantizers[0](data)
                output = PackedSequence(quantized_data, *others)
            else:
                output = self.output_quantizers[0](output)

        if self.output_quantizers[1]:
            hidden = self.output_quantizers[1](hidden)

        return output, hidden

FakeQuantizedGRU = _FakeQuantizedRNNBaseMixin.wrap(nn.GRU)
FakeQuantizedRNN = _FakeQuantizedRNNBaseMixin.wrap(nn.RNN)


class _FakeQuantizedRNNCellBaseMixin(FakeQuantizationMixin):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.GRUCell and nn.RNNCell.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        if hx is not None and self.input_quantizers[1]:
            hx = self.input_quantizers[1](hx)

        with self._patch_quantized_parameters():
            output = super().forward(input, hx)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output

FakeQuantizedGRUCell = _FakeQuantizedRNNCellBaseMixin.wrap(nn.GRUCell)
FakeQuantizedRNNCell = _FakeQuantizedRNNCellBaseMixin.wrap(nn.RNNCell)


@FakeQuantizationMixin.implements(nn.LSTM)
class FakeQuantizedLSTM(FakeQuantizationMixin, nn.LSTM):
    """
    Quantized class definition for nn.LSTM.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None, None])

    def forward(self, input, hx: Optional[Tuple[Tensor, Tensor]] = None): # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.LSTM.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            if isinstance(input, PackedSequence):
                data, *others = input
                quantized_data = self.input_quantizers[0](data)
                input = PackedSequence(quantized_data, *others)
            else:
                input = self.input_quantizers[0](input)

        if hx is not None:
            h, c = hx
            h_quantizer, c_quantizer = self.input_quantizers[1:]

            if h_quantizer:
                h = h_quantizer(h)
            if c_quantizer:
                c = c_quantizer(c)
            hx = (h, c)

        with self._patch_quantized_parameters():
            output, hidden = super().forward(input, hx)

        if self.output_quantizers[0]:
            if isinstance(output, PackedSequence):
                data, *others = output
                quantized_data = self.output_quantizers[0](data)
                output = PackedSequence(quantized_data, *others)
            else:
                output = self.output_quantizers[0](output)

        h_n, c_n = hidden
        h_quantizer, c_quantizer = self.output_quantizers[1:]

        if h_quantizer:
            h_n = h_quantizer(h_n)
        if c_quantizer:
            c_n = c_quantizer(c_n)
        hidden = (h_n, c_n)

        return output, hidden


@FakeQuantizationMixin.implements(nn.LSTMCell)
class FakeQuantizedLSTMCell(FakeQuantizationMixin, nn.LSTMCell):
    """
    Quantized class definition for nn.LSTMCell.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None): # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.LSTMCell.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        if hx is not None:
            h, c = hx
            h_quantizer, c_quantizer = self.input_quantizers[1:]
            if h_quantizer:
                h = h_quantizer(h)
            if c_quantizer:
                c = c_quantizer(c)
            hx = (h, c)

        with self._patch_quantized_parameters():
            hx, cx = super().forward(input, hx)

        if self.output_quantizers[0]:
            hx = self.output_quantizers[0](hx)

        if self.output_quantizers[1]:
            cx = self.output_quantizers[1](cx)

        return hx, cx


@FakeQuantizationMixin.implements(nn.AdaptiveLogSoftmaxWithLoss)
class FakeQuantizedAdaptiveLogSoftmaxWithLoss(FakeQuantizationMixin, nn.AdaptiveLogSoftmaxWithLoss):
    """
    Quantized class definition for nn.AdaptiveLogSoftmaxWithLoss.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def forward(self, input_: Tensor, target_: Tensor) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for nn.AdaptiveLogSoftmaxWithLoss.
        """
        if  self.input_quantizers[0]:
            input_ = self.input_quantizers[0](input_)

        if self.input_quantizers[1]:
            target_ = self.input_quantizers[1](target_)

        with self._patch_quantized_parameters():
            outputs = super().forward(input_, target_)

        output, loss = outputs

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        if self.output_quantizers[1]:
            loss = self.output_quantizers[1](loss)

        return _ASMoutput(output, loss)



# Quantized definitions of the following nn.Modules are intentionally omitted:
#  * nn.MultiheadAttention
#  * nn.Transformer
#  * nn.TransformerDecoder
#  * nn.TransformerDecoderLayer
#  * nn.TransformerEncoder
#  * nn.TransformerEncoderLayer





###########################
### AIMET V1 custom ops ###
###########################

# These class names are already occupied by torch.nn.Modules.
# To avoid name collision, we add prefix "Aimet" to the variable names as an ad-hoc workaraound.
FakeQuantizedAimetChannelShuffle = _FakeQuantizedUnaryOpMixin.wrap(custom.ChannelShuffle)
FakeQuantizedAimetMaxPool2d = _FakeQuantizedUnaryOpMixin.wrap(custom.MaxPool2d)
FakeQuantizedAimetAdaptiveAvgPool2d = _FakeQuantizedUnaryOpMixin.wrap(custom.AdaptiveAvgPool2d)
FakeQuantizedAimetAvgPool2d = _FakeQuantizedUnaryOpMixin.wrap(custom.AvgPool2d)

_AIMET_V1_UNARY_MODULES = [
    custom.AMax,
    custom.AMin,
    custom.Cast,
    custom.DepthToSpaceCRDMode,
    custom.DepthToSpaceDCRMode,
    custom.OneHot,
    custom.Exponential,
    custom.Erf,
    custom.Sqrt,
    custom.Log,
    custom.Abs,
    custom.Neg,
    custom.ElementwiseCeil,
    custom.ElementwiseFloor,
    custom.Sin,
    custom.Cos,
    custom.Asin,
    custom.Atan,
    custom.Round,
    custom.LogicalNot,
    custom.NonZero,
    custom.ElementwiseUnarySign,
    custom.RSqrt,
    custom.Square,
    custom.Mean,
    custom.Sum,
    custom.Prod,
    custom.Argmin,
    custom.Argmax,
    custom.Gather,
    custom.Reshape,
    custom.RoiAlign,
    custom.Permute,
    custom.IndexSelect,
    custom.TopK,
    custom.Tile,
    custom.Norm,
    custom.CumSum,
    custom.Interpolate,
    custom.Normalize,
    custom.Pad,
    custom.Shape,
    custom.Expand,
    custom.StridedSlice,
    custom.RmsNorm
]
_AIMET_V1_BINARY_MODULES = [
    custom.MatMul,
    custom.Add,
    custom.Multiply,
    custom.Subtract,
    custom.Divide,
    custom.FloorDivide,
    custom.Greater,
    custom.Less,
    custom.GreaterEqual,
    custom.LessEqual,
    custom.NotEqual,
    custom.Equal,
    custom.Remainder,
    custom.Fmod,
    custom.Pow,
    custom.CustomSiLU,
    custom.Maximum,
    custom.Max,
    custom.Minimum,
    custom.Min,
    custom.Bmm,
    custom.LogicalOr,
    custom.LogicalAnd,
    custom.CustomGather,
    custom.GatherNd,
]
_AIMET_V1_TERNARY_MODULES = [
    custom.Baddbmm,
    custom.Addmm,
    custom.ScatterND,
    custom.DynamicConv2d,
    custom.DynamicLinear,
    custom.ScatterElements,
]

# Auto-generate quantized module definitions for regular-patterned modules
for _module_cls in _AIMET_V1_UNARY_MODULES:
    _quantized_cls = _FakeQuantizedUnaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)

for _module_cls in _AIMET_V1_BINARY_MODULES:
    _quantized_cls = _FakeQuantizedBinaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)

for _module_cls in _AIMET_V1_TERNARY_MODULES:
    _quantized_cls = _FakeQuantizedTernaryOpMixin.wrap(_module_cls)
    _register_global_variable(_quantized_cls.__name__, _quantized_cls)



@FakeQuantizationMixin.implements(custom.BatchNorm)
class FakeQuantizedBatchNorm(FakeQuantizationMixin, custom.BatchNorm): # pylint: disable=abstract-method
    """
    Quantized class definition for custom.BatchNorm.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None, None])

    def forward(self, # pylint: disable=too-many-arguments, arguments-differ
                          input: Tensor,
                          running_mean: Optional[Tensor],
                          running_var: Optional[Tensor],
                          weight: Optional[Tensor] = None,
                          bias: Optional[Tensor] = None,
                          training: bool = False,
                          momentum: float = 0.1,
                          eps: float = 1e-5) -> Tensor:
        """
        Quantized forward impl for custom.BatchNorm.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        if running_mean is not None and self.input_quantizers[1]:
            running_mean = self.input_quantizers[1](running_mean)

        if running_var is not None and self.input_quantizers[2]:
            running_var = self.input_quantizers[2](running_var)

        if weight is not None and self.input_quantizers[3]:
            weight = self.input_quantizers[3](weight)

        if bias is not None and self.input_quantizers[4]:
            bias = self.input_quantizers[4](bias)

        output = super().forward(input, running_mean.detach(), running_var.detach(),
                                     weight, bias, training, momentum, eps)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(custom.GroupNorm)
class FakeQuantizedAimetGroupNorm(FakeQuantizationMixin, custom.GroupNorm): # pylint: disable=abstract-method
    """
    Quantized class definition for custom.GroupNorm.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None])

    def forward(self, # pylint: disable=arguments-differ
                          input: Tensor,
                          num_groups: int,
                          weight: Optional[Tensor] = None,
                          bias: Optional[Tensor] = None,
                          eps: float = 1e-5) -> Tensor:
        """
        Quantized forward impl for custom.GroupNorm.
        """
        # pylint: disable=redefined-builtin

        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        if weight is not None and self.input_quantizers[2]:
            weight = self.input_quantizers[2](weight)

        if bias is not None and self.input_quantizers[3]:
            bias = self.input_quantizers[3](bias)

        output = super().forward(input, num_groups, weight, bias, eps)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(custom.NonMaxSuppression)
class FakeQuantizedNonMaxSuppression(FakeQuantizationMixin, custom.NonMaxSuppression):
    """
    Quantized class definition for custom.NonMaxSuppression.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, *args) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.NonMaxSuppression.
        """
        boxes, scores = args # boxes are integer tensors

        if self.input_quantizers[0]:
            # Use same input quantizer for all the score tensors
            scores = tree_map(self.input_quantizers[0], scores)

        output = super().forward(boxes, scores)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(custom.Split)
class FakeQuantizedSplit(FakeQuantizationMixin, custom.Split):
    """
    Quantized class definition for custom.Split.
    """
    def forward(self, *args, **kwargs): # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.Split.
        """
        x, *others = args

        if x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        outputs = super().forward(x, *others, **kwargs)

        if self.output_quantizers[0]:
            # Use same output quantizer for all the output tensors
            quantize_fn = lambda out: self.output_quantizers[0](out) if out.is_floating_point() else out
            outputs = tree_map(quantize_fn, outputs)

        return outputs


@FakeQuantizationMixin.implements(custom.Concat)
class FakeQuantizedConcat(FakeQuantizationMixin, custom.Concat):
    """
    Quantized class definition for custom.Concat.
    """
    _num_inputs: int

    def __quant_init__(self):
        super().__quant_init__()
        self._num_inputs = 1

    def export_input_encodings(self, encoding_version: str):
        """
        Extends super().export to repeat input quantizer's encodings :attr:`self._num_inputs` times
        """
        input_encodings = super().export_input_encodings(encoding_version)
        return input_encodings * self._num_inputs

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

        if self.input_quantizers[0]:
            # Use same input quantizer for all the input tensors
            quantize_fn = lambda inp: self.input_quantizers[0](inp) if inp.is_floating_point() else inp
            x = tree_map(quantize_fn, x)

        output = super().forward(*x)

        if output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(custom.Where)
class FakeQuantizedWhere(FakeQuantizationMixin, custom.Where): # pylint: disable=abstract-method
    """
    Quantized class definition for custom.Where.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, condition: Tensor, input, other, **kwargs) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.MaskedFill.
        """
        # pylint: disable=redefined-builtin

        if isinstance(input, Tensor) and input.is_floating_point() and self.input_quantizers[1]:
            input = self.input_quantizers[1](input)

        if isinstance(other, Tensor) and other.is_floating_point() and self.input_quantizers[2]:
            other = self.input_quantizers[2](other)

        output = super().forward(condition, input, other, **kwargs)

        if output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@FakeQuantizationMixin.implements(custom.MaskedFill)
class FakeQuantizedMaskedFill(FakeQuantizationMixin, custom.MaskedFill): # pylint: disable=abstract-method
    """
    Quantized class definition for custom.MaskedFill.
    """
    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, mask: Tensor, value) -> Tensor: # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.MaskedFill.
        """
        if isinstance(value, Tensor) and value.is_floating_point() and self.input_quantizers[1]:
            value = self.input_quantizers[1](value)

        output = super().forward(mask, value)

        if output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output
