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
""" QuantizationSimModel interface """
from abc import ABC, abstractmethod
from typing import (
    List,
    Union,
    Dict,
    Optional,
    runtime_checkable,
    Protocol,
    Mapping,
    TYPE_CHECKING,
    Tuple,
    Iterable,
)
import torch

from aimet_common.defs import QuantScheme

if TYPE_CHECKING:
    from aimet_torch.v2.quantization.base.encoding import EncodingBase


class QuantParams:
    """
    Data type to hold quantization related params.
    """

    def __init__(self,
                 weight_bw: int = 8,
                 act_bw: int = 8,
                 round_mode: str = 'nearest',
                 quant_scheme: Union[QuantScheme, str] = QuantScheme.post_training_tf_enhanced,
                 config_file: str = None):
        """
        Constructor

        :param weight_bw: Weight bitwidth (4-31) to use for quantizing layer weights. Default = 8
        :param act_bw: Activation bitwidth(4-31) to use for quantizing layer activations. Default = 8
        :param round_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param config_file: Path to Configuration file for model quantizers
        """

        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.round_mode = round_mode
        self.quant_scheme = quant_scheme
        self.config_file = config_file


@runtime_checkable
class _QuantizerProtocol(Protocol):
    def get_encodings(self) -> Optional["EncodingBase"]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """

    def set_encodings(self, encoding: "EncodingBase"):
        """
        Set the quantizer's encodings
        """


@runtime_checkable
class _QuantizedModuleProtocol(Protocol):
    """
    Defines the minimum interface requirements for exporting encodings from a module.
    """
    input_quantizers: List[_QuantizerProtocol]
    output_quantizers: List[_QuantizerProtocol]
    param_quantizers: Dict[str, _QuantizerProtocol]

    def export_input_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """

    def export_output_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """

    def export_param_encodings(self) -> Dict[str, List[Dict]]:
        """
        Returns a dict of {param name: param encodings}, with each encoding represented as a List of Dicts
        """

    def import_input_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import input encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_output_encodings(self,
                                encodings: Mapping[str, Mapping],
                                strict: bool,
                                partial: bool,
                                requires_grad: Optional[bool],
                                allow_overwrite: bool):
        """
        Import output encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_param_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import parameter encodings represented in below format:
        {
            'param_name_0': [dict, dict, ...],
            'param_name_1': [dict, dict, ...],
            ...
        }
        """

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns the floating point version of quantized module
        """


ExportableQuantModule = _QuantizedModuleProtocol


class _QuantizationSimModelInterface(ABC):
    model: torch.nn.Module

    @abstractmethod
    def compute_encodings(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def export(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def load_encodings(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def named_qmodules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        """Generator that yields all quantized modules in the model and their names
        """

    def qmodules(self) -> Iterable[torch.nn.Module]:
        """Generator that yields all quantized modules in the model
        """
        yield from (module for _, module in self.named_qmodules())
