# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=import-outside-toplevel
"""v2 lazy quant wrapper / quantizer"""

import numpy as np
import torch

from aimet_torch.common.defs import QuantScheme, QuantizationDataType
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper, LazyQuantizer
from aimet_torch.utils import get_per_channel_scale_shape
import aimet_torch.fp_quantization as v1_fp_quantization
from aimet_torch.quantization.float import FloatQuantizeDequantize
from aimet_torch.quantization.affine import QuantizeDequantize, AffineQuantizerBase
from aimet_torch.quantization.encoding_analyzer import (
    MinMaxEncodingAnalyzer,
    PercentileEncodingAnalyzer,
    TfEnhancedEncodingAnalyzer,
)


class _V2LazyQuantizer(LazyQuantizer):
    def realize(self):
        """Returns v2 quantizer using collected information."""
        if not self.enabled:
            return None

        self._validate_quantizer_properties()

        scale_shape = self._get_scale_shape()

        if self.data_type == QuantizationDataType.int:
            encoding_analyzer = self._get_v2_encoding_analyzer(scale_shape)
            quantizer = QuantizeDequantize(
                scale_shape,
                self.bitwidth,
                self.use_symmetric_encodings,
                encoding_analyzer,
            )
        else:
            if self.bitwidth == 16:
                quantizer = FloatQuantizeDequantize(dtype=torch.float16)
            else:
                assert self.bitwidth == 8
                mantissa_bits = v1_fp_quantization.NUM_MANTISSA_BITS
                exponent_bits = 7 - mantissa_bits
                encoding_analyzer = self._get_v2_encoding_analyzer(scale_shape)
                quantizer = FloatQuantizeDequantize(
                    exponent_bits,
                    mantissa_bits,
                    shape=scale_shape,
                    encoding_analyzer=encoding_analyzer,
                )
            # Float quantizers are not trainable in V1 quantsim
            for param in quantizer.parameters():
                param.requires_grad = False

        return quantizer

    def _validate_quantizer_properties(self):
        """
        Checks quantizer properties before creating quantizer.
        """
        if self.use_symmetric_encodings:
            assert not self.use_strict_symmetric, (
                "Strict symmetric is not supported in quantsim v2"
            )
            assert not self.use_unsigned_symmetric, (
                "Unsigned symmetric is not supported in quantsim v2"
            )
            assert not self.is_unsigned_symmetric, (
                "Unsigned symmetric is not supported in quantsim v2"
            )

    def _get_scale_shape(self) -> tuple[int, ...]:
        """Returns shape of quantization scale/offset."""
        if self.channel_axis is not None:
            assert self.input_tensor_shape
            return get_per_channel_scale_shape(
                self.input_tensor_shape, self.channel_axis
            )

        return tuple()

    def _get_v2_encoding_analyzer(self, shape):
        """
        Converts v1 quant scheme into v2 quant scheme.

        :return: corresponding v2 quant scheme
        """
        if self.quant_scheme in (
            QuantScheme.post_training_tf,
            QuantScheme.training_range_learning_with_tf_init,
        ):
            return MinMaxEncodingAnalyzer(shape)
        if self.quant_scheme == QuantScheme.post_training_percentile:
            return PercentileEncodingAnalyzer(shape)
        if self.quant_scheme in (
            QuantScheme.post_training_tf_enhanced,
            QuantScheme.training_range_learning_with_tf_enhanced_init,
        ):
            return TfEnhancedEncodingAnalyzer(shape)
        raise NotImplementedError(
            f"Quant scheme {self.quant_scheme} in old quantsim is not supported yet in quantsim v1.5"
        )


class _V2LazyQuantizeWrapper(LazyQuantizeWrapper):
    _lazy_qtzr_cls = _V2LazyQuantizer

    def realize(self):
        """
        Realizes v2 quant wrapper using collected information

        :return: v2 quant wrapper with specified properties
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        from aimet_torch.v2.nn import QuantizationMixin
        from aimet_torch.nn.fake_quant import _legacy_impl

        assert isinstance(
            self._module_to_wrap,
            (QuantizationMixin, _legacy_impl.FakeQuantizationMixin),
        )
        quantized_module = self._module_to_wrap

        # For unused modules, quantsim assumes # inputs = # outputs = 1
        # If this is incorrect, propagate the configuration of the last input/output quantizers to the remaining
        # quantizer positions
        for i in range(
            max(len(self.input_quantizers), len(quantized_module.input_quantizers))
        ):
            src_idx = min(i, len(self.input_quantizers) - 1)
            quantizer = self.input_quantizers[src_idx].realize()
            if i < len(quantized_module.input_quantizers):
                quantized_module.input_quantizers[i] = quantizer
            elif quantized_module._supports_dynamic_input_count():  # pylint: disable=protected-access
                quantized_module.input_quantizers.append(quantizer)

        for i, _ in list(enumerate(quantized_module.output_quantizers)):
            q_idx = min(i, len(self.output_quantizers) - 1)
            quantizer = self.output_quantizers[q_idx].realize()
            quantized_module.output_quantizers[i] = quantizer

        for param_name, quant_builder in self.param_quantizers.items():
            quantized_module.param_quantizers[param_name] = quant_builder.realize()

        self._apply_quant_param_value_constraints(quantized_module)
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module

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

        quantizer_list = (
            quantized_module.input_quantizers
            + quantized_module.output_quantizers
            + param_quantizers
        )
        quantizer_info_list = (
            self.input_quantizers + self.output_quantizers + param_quantizer_info_list
        )

        for quantizer, quantizer_info in zip(quantizer_list, quantizer_info_list):
            # pylint: disable=protected-access
            if (
                isinstance(quantizer, AffineQuantizerBase)
                and quantizer_info.encoding_min_max_fixed_vals
            ):
                fixed_min, fixed_max = quantizer_info.encoding_min_max_fixed_vals

                if (
                    fixed_min is not None
                    and fixed_max is not None
                    and np.allclose(fixed_min, -fixed_max)
                ):
                    # Symmetric range. Set symmetric=True to ensure symmetry
                    quantizer.symmetric = True

                quantizer.set_range(fixed_min, fixed_max)
                quantizer.allow_overwrite(min=fixed_min is None, max=fixed_max is None)
                quantizer.min.requires_grad_(fixed_min is None)
                quantizer.max.requires_grad_(fixed_max is None)
