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
""" Concrete implementation for AIMET AutoQuant using v2 QuantSim """

import functools
import itertools
import torch

import aimet_torch.v2.quantization as Q
from aimet_torch._base.auto_quant import AutoQuantBase
from aimet_torch.v2.batch_norm_fold import fold_all_batch_norms
from aimet_torch.v2.adaround import Adaround, AdaroundParameters
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization import encoding_analyzer
from aimet_torch.v2.utils import flatten_nn_module_list
from aimet_common.defs import QuantScheme

# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None

_MAP_QSCHEME_TO_ENCODING_ANALYZER = {
    QuantScheme.post_training_tf: encoding_analyzer.MinMaxEncodingAnalyzer,
    QuantScheme.post_training_percentile: encoding_analyzer.PercentileEncodingAnalyzer,
    QuantScheme.post_training_tf_enhanced: encoding_analyzer.SqnrEncodingAnalyzer,
    QuantScheme.training_range_learning_with_tf_init: encoding_analyzer.MinMaxEncodingAnalyzer,
    QuantScheme.training_range_learning_with_tf_enhanced_init: encoding_analyzer.SqnrEncodingAnalyzer,
}


class AutoQuant(AutoQuantBase): # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    @staticmethod
    def _get_adaround():
        """ returns AdaRound """
        return Adaround

    @staticmethod
    def _fold_all_batch_norms(*args, **kwargs):
        return fold_all_batch_norms(*args, **kwargs)

    @functools.wraps(AutoQuantBase.__init__)
    def __init__(self, *args, rounding_mode: str = 'nearest', **kwargs):
        if rounding_mode == 'stochastic':
            raise ValueError("Stochastic rounding mode is not supported.")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_adaround_parameters(data_loader, num_batches):
        return AdaroundParameters(data_loader, num_batches)

    def _evaluate_model_performance(self, model) -> float:
        """
        Evaluate the model performance.
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    @staticmethod
    def _get_quantsim(model, dummy_input, **kwargs):
        return QuantizationSimModel(model, dummy_input, **kwargs)

    def _configure_quantsim(self, # pylint: disable=too-many-arguments
                            sim,
                            output_bw,
                            output_quant_scheme,
                            output_percentile,
                            param_bw,
                            param_quant_scheme,
                            param_percentile,
                            encoding_path):

        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                # Set input/output quantizers' quant schemes
                for quantizer in itertools.chain(flatten_nn_module_list(module.input_quantizers),
                                                 flatten_nn_module_list(module.output_quantizers)):
                    self._set_quantizer_qscheme(quantizer, output_quant_scheme, output_percentile)

                # Set param quantizers' quant schemes
                for quantizer in module.param_quantizers.values():
                    self._set_quantizer_qscheme(quantizer, param_quant_scheme, param_percentile)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        if output_bw == 32:
            self._disable_activation_quantizers(sim)

        if param_bw == 32:
            self._disable_param_quantizers(sim)


    @staticmethod
    def _set_quantizer_qscheme(quantizer, quant_scheme, percentile):
        if quantizer is None:
            return

        if quant_scheme in (QuantScheme.post_training_percentile, QuantScheme.post_training_tf,
                            QuantScheme.post_training_tf_enhanced):
            quantizer.requires_grad_(False)

        elif QuantScheme in (QuantScheme.training_range_learning, QuantScheme.training_range_learning_with_tf_init,
                             QuantScheme.training_range_learning_with_tf_enhanced_init):
            quantizer.requires_grad_(True)

        enc_analyzer = _MAP_QSCHEME_TO_ENCODING_ANALYZER[quant_scheme](quantizer.shape)
        if isinstance(enc_analyzer, encoding_analyzer.PercentileEncodingAnalyzer) and percentile is not None:
            enc_analyzer.set_percentile(percentile)

        quantizer.encoding_analyzer = enc_analyzer


    @staticmethod
    def _has_enabled_quantizers(sim):
        for module in sim.model.modules():
            if isinstance(module, Q.base.QuantizerBase):
                return True
        return False

    @staticmethod
    def _disable_activation_quantizers(sim):
        def recursive_disable_quantizers(quantizer_list):
            for idx, quantizer in enumerate(quantizer_list):
                if isinstance(quantizer, (list, tuple, torch.nn.ModuleList)):
                    recursive_disable_quantizers(quantizer)
                else:
                    quantizer_list[idx] = None

        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                recursive_disable_quantizers(module.input_quantizers)
                recursive_disable_quantizers(module.output_quantizers)

    @staticmethod
    def _disable_param_quantizers(sim):
        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                for name, _ in module.param_quantizers.items():
                    module.param_quantizers[name] = None
