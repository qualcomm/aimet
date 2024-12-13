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
""" v1 lazy quant wrapper / quantizer """

from aimet_torch.utils import get_v1_quant_scheme_for_initialization
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper, LazyQuantizer
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper, StaticGridQuantWrapper, tensor_quantizer_factory
from aimet_torch.v1.tensor_quantizer import TensorQuantizer, StaticGridPerChannelQuantizer


class _V1LazyQuantizer(LazyQuantizer):
    def realize(self) -> TensorQuantizer:
        """ Returns v1 quantizer using collected information. """
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(self.quant_scheme)

        if self.channel_axis is not None:
            assert self.input_tensor_shape
            num_channels = self.input_tensor_shape[self.channel_axis]

            quantizer = StaticGridPerChannelQuantizer(self.bitwidth, self.round_mode, quant_scheme_for_initialization,
                                                      self.use_symmetric_encodings, num_channels,
                                                      self.enabled, self.channel_axis, self.data_type)
        else:
            quantizer = tensor_quantizer_factory(self.bitwidth, self.round_mode, quant_scheme_for_initialization,
                                                 self.use_symmetric_encodings, self.enabled, self.data_type)

        self._set_internal_quantizer_properties(quantizer)

        return quantizer

    def _set_internal_quantizer_properties(self, quantizer: TensorQuantizer):
        """
        Sets internal quantizer properties of v1 quantizer
        using collected information.

        :param quantizer: quantizer to update its internal properties
        """
        if self.encoding_min_max_fixed_vals is not None:
            quantizer.encoding_min_max_fixed_vals = self.encoding_min_max_fixed_vals
        quantizer.is_unsigned_symmetric = self.is_unsigned_symmetric
        quantizer.use_unsigned_symmetric = self.use_unsigned_symmetric
        quantizer.use_strict_symmetric = self.use_strict_symmetric
        quantizer.is_const = self.is_const


class _V1LazyQuantizeWrapper(LazyQuantizeWrapper):
    @property
    def _lazy_qtzr_cls(self):
        return _V1LazyQuantizer

    def realize(self) -> QcQuantizeWrapper:
        """
        Realizes v1 quant wrapper using collected information

        :return: v1 quant wrapper with specified properties
        """
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(self._quant_scheme)

        quantized_module = StaticGridQuantWrapper(self._module_to_wrap, self._weight_bw, self._activation_bw,
                                                  self._rounding_mode, quant_scheme_for_initialization,
                                                  self._is_output_quantized, self._is_symmetric, self._num_inputs,
                                                  self._num_outputs, self._data_type)

        quantized_module.input_quantizers = [quant_builder.realize() for quant_builder in self.input_quantizers]
        quantized_module.output_quantizers = [quant_builder.realize() for quant_builder in self.output_quantizers]
        quantized_module.param_quantizers = {param_name: quant_builder.realize() \
                      for (param_name, quant_builder) in self.param_quantizers.items()}
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module
