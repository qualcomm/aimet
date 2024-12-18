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
""" Wrapper and quantizer builder class for supporting both v1 and v2 blocks """

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type
import torch

from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_common.utils import AimetLogger, log_with_error_and_assert_if_false
from aimet_torch.utils import is_leaf_module


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint: disable=import-outside-toplevel

class LazyQuantizeWrapper(torch.nn.Module, ABC): # pylint: disable=too-many-instance-attributes
    """
    Wrapper builder class for supporting both v1 and v2 blocks
    """
    _lazy_qtzr_cls: Type['LazyQuantizer']

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def __init__(self, module_to_wrap: torch.nn.Module, weight_bw: int, activation_bw: int, rounding_mode,
                 quant_scheme: QuantScheme, is_output_quantized=True, is_symmetric=False, num_inputs=1, num_outputs=1,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        super().__init__()
        if data_type == QuantizationDataType.float and weight_bw not in [8, 16, 32]:
            raise ValueError('weight_bw in [8, 16, 32] is the only supported configuration with floating point data type')

        if data_type == QuantizationDataType.float and activation_bw not in [8, 16, 32]:
            raise ValueError('activation_bw in [8, 16, 32] is the only supported configuration with floating point data type')

        # Save those parameters for v1 quant wrapper initialization
        self._weight_bw = weight_bw
        self._activation_bw = activation_bw
        self._rounding_mode = rounding_mode
        self._quant_scheme = quant_scheme
        self._is_output_quantized = is_output_quantized
        self._is_symmetric = is_symmetric
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._data_type = data_type
        self._module_to_wrap = module_to_wrap

        # Create quantizer for layer output
        self.output_quantizers = [self._lazy_qtzr_cls(activation_bw,
                                                     rounding_mode,
                                                     quant_scheme,
                                                     is_symmetric,
                                                     enabled_by_default=is_output_quantized,
                                                     data_type=data_type)
                                  for _ in range(num_outputs)]

        # Create quantizer for each parameter and compute encodings
        self.param_quantizers = {}

        # pylint: disable=import-outside-toplevel, cyclic-import
        from aimet_torch.v2.nn import BaseQuantizationMixin
        if isinstance(module_to_wrap, BaseQuantizationMixin):
            # NOTE: AIMET v2 qmodule always only quantizes the paramters that it directly owns
            recurse = False
        else:
            # NOTE: This is only for backwards-compatibility with v1 quant wrapper
            #       which sometimes tries to quantize not only the parameters it directly owns
            #       but also all the parameters of its submodules in some edge cases
            assert is_leaf_module(module_to_wrap)
            recurse = True

        for name, param in module_to_wrap.named_parameters(recurse=recurse):
            logger.debug("Adding quantizer for parameter: %s", name)
            qtzr = self._lazy_qtzr_cls(weight_bw,
                                       rounding_mode,
                                       quant_scheme,
                                       is_symmetric,
                                       enabled_by_default=True,
                                       data_type=data_type)
            from aimet_torch.v2.deepspeed_utils import _get_shape
            qtzr.input_tensor_shape = _get_shape(param)
            self.param_quantizers[name] = qtzr

        # Create quantizer for layer input
        self.input_quantizers = [self._lazy_qtzr_cls(activation_bw,
                                                     rounding_mode,
                                                     quant_scheme,
                                                     is_symmetric,
                                                     enabled_by_default=False,
                                                     data_type=data_type)
                                 for _ in range(num_inputs)]

        self.supported_kernels = {}

    def get_original_module(self):
        """
        Returns the floating point version of quantized module
        """
        return self._module_to_wrap

    def enable_per_channel_quantization(self):
        """
        Changes all parameter quantizers (if any) to per-channel mode.
        """
        for param_name, param_quantizer in self.param_quantizers.items():
            channel_axis = 0
            if isinstance(self._module_to_wrap, (torch.nn.ConvTranspose1d,
                                                 torch.nn.ConvTranspose2d,
                                                 torch.nn.ConvTranspose3d)):
                channel_axis = 1 if param_name == 'weight' else 0

            # pylint: disable = protected-access
            param_quantizer.channel_axis = channel_axis

    @staticmethod
    def forward(_):
        """
        Dummy forward-pass routine for implementing abstract function.
        """
        raise RuntimeError("forward function of LazyQuantizeWrapper should not be called before it is realized")

    @abstractmethod
    def realize(self):
        """ Returns v1 or v2 quantized module using collected information. """


class LazyQuantizer(ABC):
    """
    Quantizer builder class for supporting both v1 and v2 blocks
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, bitwidth: int, round_mode, quant_scheme: QuantScheme,
                 use_symmetric_encodings: bool, enabled_by_default: bool,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        self.round_mode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        self.quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = False
        self.is_unsigned_symmetric = False
        self.bitwidth = bitwidth
        self.enabled = enabled_by_default
        self.data_type = data_type
        self.is_const = False
        self.is_parm = False
        self.is_singleton = False
        self._encoding_min_max_fixed_vals = None
        self.input_tensor_shape = None # None indicates unknown
        self.channel_axis = None

    @property
    def encoding_min_max_fixed_vals(self) -> Optional[Tuple[float, float]]:
        """ Accessor to self._encoding_min_max_fixed_vals """
        return self._encoding_min_max_fixed_vals

    @encoding_min_max_fixed_vals.setter
    def encoding_min_max_fixed_vals(self, min_max_vals: Tuple[float, float]):
        """ self._encoding_min_max_fixed_vals setter """
        log_with_error_and_assert_if_false(isinstance(min_max_vals, tuple), logger, 'Min max vals must be a tuple')
        log_with_error_and_assert_if_false(len(min_max_vals) == 2, logger, 'Min max vals must be a tuple of two '
                                                                           'values')
        log_with_error_and_assert_if_false(min_max_vals[0] < min_max_vals[1], logger,
                                           'Min value ' + str(min_max_vals[0]) + ' is not less than max val ' +
                                           str(min_max_vals[1]))
        if self.quant_scheme != QuantScheme.post_training_tf:
            self.quant_scheme = QuantScheme.post_training_tf
        self._encoding_min_max_fixed_vals = min_max_vals

    @abstractmethod
    def realize(self):
        """ Returns v1 or v2 quantizer using collected information. """
