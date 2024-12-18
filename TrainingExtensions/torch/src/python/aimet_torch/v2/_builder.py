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
# pylint: disable=import-outside-toplevel
""" v2 lazy quant wrapper / quantizer """

import itertools
from typing import Sequence
import torch

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper, LazyQuantizer
import aimet_torch.fp_quantization as v1_fp_quantization
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer, PercentileEncodingAnalyzer, SqnrEncodingAnalyzer


class _V2LazyQuantizer(LazyQuantizer):
    def realize(self):
        """ Returns v2 quantizer using collected information. """
        if not self.enabled:
            return None

        if self.is_const and self.is_parm and self.is_singleton:
            return None

        self._validate_quantizer_properties()

        scale_shape = self._get_scale_shape()

        if self.data_type == QuantizationDataType.int:
            encoding_analyzer = self._get_v2_encoding_analyzer(scale_shape)
            quantizer = QuantizeDequantize(scale_shape, self.bitwidth,
                                           self.use_symmetric_encodings, encoding_analyzer)
        else:
            if self.bitwidth == 16:
                quantizer = FloatQuantizeDequantize(dtype=torch.float16)
            else:
                assert self.bitwidth == 8
                mantissa_bits = v1_fp_quantization.NUM_MANTISSA_BITS
                exponent_bits = 7 - mantissa_bits
                encoding_analyzer = self._get_v2_encoding_analyzer(scale_shape)
                quantizer = FloatQuantizeDequantize(exponent_bits, mantissa_bits,
                                                    encoding_analyzer=encoding_analyzer)
            # Float quantizers are not trainable in V1 quantsim
            for param in quantizer.parameters():
                param.requires_grad = False

        return quantizer

    def _validate_quantizer_properties(self):
        """
        Checks quantizer properties before creating quantizer.
        """
        if self.use_symmetric_encodings:
            assert not self.use_strict_symmetric, "Strict symmetric is not supported in quantsim v2"
            assert not self.use_unsigned_symmetric, "Unsigned symmetric is not supported in quantsim v2"
            assert not self.is_unsigned_symmetric, "Unsigned symmetric is not supported in quantsim v2"

        if self.channel_axis:
            assert self.input_tensor_shape
            assert 0 <= self.channel_axis < len(self.input_tensor_shape), \
                f"Channel axis {self.channel_axis} is out of bound of param shape {self.input_tensor_shape}"

    def _get_scale_shape(self) -> Sequence[int]:
        """ Returns shape of quantization scale/offset. """
        if self.channel_axis is not None:
            assert self.input_tensor_shape
            channel_axis = self.channel_axis if self.channel_axis else 0

            scale_shape = [1] * len(self.input_tensor_shape)
            scale_shape[channel_axis] = self.input_tensor_shape[channel_axis]

            return scale_shape

        return tuple()

    def _get_v2_encoding_analyzer(self, shape):
        """
        Converts v1 quant scheme into v2 quant scheme.

        :return: corresponding v2 quant scheme
        """
        if self.quant_scheme in (QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init):
            return MinMaxEncodingAnalyzer(shape)
        if self.quant_scheme == QuantScheme.post_training_percentile:
            return PercentileEncodingAnalyzer(shape)
        if self.quant_scheme in (QuantScheme.post_training_tf_enhanced,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init):
            return SqnrEncodingAnalyzer(shape)
        raise NotImplementedError(f"Quant scheme {self.quant_scheme} in old quantsim is not supported yet in quantsim v1.5")


class _V2LazyQuantizeWrapper(LazyQuantizeWrapper):
    _lazy_qtzr_cls = _V2LazyQuantizer

    def realize(self):
        """
        Realizes v2 quant wrapper using collected information

        :return: v2 quant wrapper with specified properties
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        from aimet_torch.v2.nn import QuantizationMixin
        from aimet_torch.v2.nn.fake_quant import _legacy_impl

        assert isinstance(self._module_to_wrap, (QuantizationMixin, _legacy_impl.FakeQuantizationMixin))
        quantized_module = self._module_to_wrap

        # For unused modules, quantsim assumes # inputs = # outputs = 1
        # If this is incorrect, propagate the configuration of the last input/output quantizers to the remaining
        # quantizer positions
        for i, _ in list(enumerate(quantized_module.input_quantizers)):
            q_idx = min(i, len(self.input_quantizers) - 1)
            quantizer = self.input_quantizers[q_idx].realize()
            quantized_module.input_quantizers[i] = quantizer

        for i, _ in list(enumerate(quantized_module.output_quantizers)):
            q_idx = min(i, len(self.output_quantizers) - 1)
            quantizer = self.output_quantizers[q_idx].realize()
            quantized_module.output_quantizers[i] = quantizer

        for param_name, quant_builder in self.param_quantizers.items():
            quantized_module.param_quantizers[param_name] = quant_builder.realize()

        self._apply_quant_param_value_constraints(quantized_module)
        self._update_quant_param_requires_grad(quantized_module)
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module

    def _update_quant_param_requires_grad(self, quantized_module):
        """
        Update requres_grad value of quantizers in quantized_module.

        :param quantized_module: module containing quantizers whose requires_grad need to be updated
        """
        if self._quant_scheme in (QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_tf, \
                                  QuantScheme.post_training_percentile):
            for quantizer in itertools.chain(quantized_module.input_quantizers,
                                             quantized_module.output_quantizers,
                                             quantized_module.param_quantizers.values()):
                if quantizer is not None:
                    for _, param in quantizer.named_parameters():
                        param.requires_grad = False

    def _apply_quant_param_value_constraints(self, quantized_module):
        """
        Update min and max of quantizers if their values are specified in config

        :param quantized_module: module containing quantizers whose params need to be updated
        """
        param_quantizer_dict = quantized_module.param_quantizers
        param_quantizers = []
        param_quantizer_info_list = []
        for key in param_quantizer_dict:
            param_quantizers.append(param_quantizer_dict[key])
            param_quantizer_info_list.append(self.param_quantizers[key])

        quantizer_list = quantized_module.input_quantizers + quantized_module.output_quantizers + param_quantizers
        quantizer_info_list = self.input_quantizers + self.output_quantizers + param_quantizer_info_list

        for quantizer, quantizer_info in zip(quantizer_list, quantizer_info_list):
            # pylint: disable=protected-access
            if quantizer is not None and quantizer_info.encoding_min_max_fixed_vals and \
                    'min' in quantizer._initial_parameters and 'max' in quantizer._initial_parameters:
                with torch.no_grad():
                    quantizer.min.copy_(quantizer_info.encoding_min_max_fixed_vals[0])
                    quantizer.max.copy_(quantizer_info.encoding_min_max_fixed_vals[1])
                quantizer.allow_overwrite(False)
                quantizer.requires_grad_(False)
