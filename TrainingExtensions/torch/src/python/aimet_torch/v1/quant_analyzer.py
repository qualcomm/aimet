# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Quant Analyzer """

import os
import contextlib
from typing import Tuple, List, Type, Generator

from aimet_common.quant_analyzer import export_stats_histogram_plot
from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch._base.quant_analyzer import QuantAnalyzerBase
from aimet_torch.v1.tensor_quantizer import TensorQuantizer, StaticGridTensorQuantizer
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.v1.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.QuantAnalyzer)

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


class QuantAnalyzer(QuantAnalyzerBase):
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per PDF analysis and
     5) per layer MSE analysis
    """

    @staticmethod
    def _enable_disable_quantizers(quantizers: List[TensorQuantizer], enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """
        for quantizer in quantizers:
            quantizer.enabled = enabled

    # pylint: disable=no-self-use
    def _create_and_export_stats_histogram_plot(self,
                                                quantizer: StaticGridTensorQuantizer,
                                                results_dir: str,
                                                title: str,
                                                ):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        histograms = quantizer.get_stats_histogram()
        encodings = quantizer.encoding
        if not isinstance(encodings, List):
            encodings = [encodings]

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            export_stats_histogram_plot(histogram, encoding, results_dir, title=f"{title}_{index}")

    @staticmethod
    def patch_quantsim_to_store_histogram(_):
        """
        Placeholder function to prevent patching v1 quantsim
        """

    @staticmethod
    def _get_quantsim_cls() -> Type[QuantizationSimModel]:
        return QuantizationSimModel

    @staticmethod
    def _get_quant_wrapper_type() -> Tuple[Type]:
        return (QcQuantizeWrapper, QcQuantizeRecurrent)

    @staticmethod
    def _is_quantizer_enabled(quantizer: TensorQuantizer):
        return quantizer.enabled

    @staticmethod
    def _get_quantizer_encodings(quantizer: TensorQuantizer):
        if quantizer.encoding and not isinstance(quantizer.encoding, List):
            return [quantizer.encoding]
        return quantizer.encoding

    @classmethod
    @contextlib.contextmanager
    def _disable_param_quantizers(cls, sim: QuantizationSimModel):
        enabled_param_quantizers = cls._get_enabled_param_quantizers(sim)
        cls._enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        yield
        cls._enable_disable_quantizers(enabled_param_quantizers, enabled=True)

    @classmethod
    @contextlib.contextmanager
    def _disable_activation_quantizers(cls, sim: QuantizationSimModel):
        enabled_activation_quantizers = cls._get_enabled_activation_quantizers(sim)
        cls._enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        yield
        cls._enable_disable_quantizers(enabled_activation_quantizers, enabled=True)

    @staticmethod
    def _disable_quant_wrapper(module: QcQuantizeWrapper):
        return utils.disable_all_quantizers(module)

    @staticmethod
    def _get_quantized_modules(sim: QuantizationSimModel) -> Generator[QcQuantizeWrapper, None, None]:
        for module in sim.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                yield module

    @staticmethod
    def _fold_all_batch_norms(*args, **kwargs):
        return fold_all_batch_norms(*args, **kwargs)
