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

import contextlib
from dataclasses import dataclass
import itertools
import json
import math
from unittest.mock import patch, MagicMock
import os
import pytest
import shutil
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader
from bs4 import BeautifulSoup

from aimet_torch.v2.auto_quant import AutoQuantWithAutoMixedPrecision as AutoQuant
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
from aimet_common.defs import QuantizationDataType
from aimet_torch import utils
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.save_utils import SaveUtils


_W4A8 = (
    (8, QuantizationDataType.int), # A: int8
    (4, QuantizationDataType.int), # W: int4
)
_W7A9 = (
    (9, QuantizationDataType.int), # A: int8
    (7, QuantizationDataType.int), # W: int8
)
_W8A8 = (
    (8, QuantizationDataType.int), # A: int8
    (8, QuantizationDataType.int), # W: int8
)
_W8A16 = (
    (16, QuantizationDataType.int), # A: int16
    (8, QuantizationDataType.int),  # W: int8
)
_FP16 = (
    (16, QuantizationDataType.float), # A: fp16
    (16, QuantizationDataType.float), # W: fp16
)


class Model(torch.nn.Module):
    """
    Model
    """

    def __init__(self):
        super(Model, self).__init__()
        self._conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)
        self._relu = torch.nn.ReLU()

        # Test flags
        self.register_buffer("applied_bn_folding", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.register_buffer("applied_cle", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.register_buffer("applied_adaround", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.register_buffer("applied_amp", torch.tensor(False, dtype=torch.bool), persistent=True)

    def forward(self, x: torch.Tensor):
        # Return the test flags along with the forward pass results so that the test flags
        # don't get discarded when the model is converted to GraphModule by model preparer.
        return self._relu(self._conv_0(x)),\
               self.applied_bn_folding,\
               self.applied_cle,\
               self.applied_adaround,\
               self.applied_amp


@pytest.fixture(scope="session")
def cpu_model():
    return Model().cpu()


@pytest.fixture(scope="session")
def gpu_model():
    return Model().cuda()


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 3, 8, 8))


@pytest.fixture(scope="session")
def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([dummy_input[0, :] for _ in range(10)])
    return DataLoader(dataset)


def assert_html(html_parsed, properties):
    for id_, prop in properties.items():
        elem = html_parsed.find(id=id_)
        assert elem is not None
        for prop_name, prop_val in prop.items():
            if prop_val is None:
                assert prop_name not in elem.attrs
            else:
                assert elem[prop_name] == prop_val


_VISITED = { 'data-visited': 'true', }
_NOT_VISITED = { 'data-visited': None, }
_SUCCESS = {
    'data-visited': 'true',
    'data-stage-result': 'success'
}
_DISCARDED = {
    'data-visited': 'true',
    'data-stage-result': 'discarded'
}
_ERROR_IGNORED = {
    'data-visited': 'true',
    'data-stage-result': 'error-ignored'
}
_ERROR_FAILED = {
    'data-visited': 'true',
    'data-stage-result': 'error-failed'
}

def assert_applied_techniques(
        output_model, acc, encoding_path,
        target_acc, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
        results_dir,
):
    html_path = os.path.join(results_dir, 'diagnostics.html')
    with open(html_path) as f:
        html_parsed = BeautifulSoup(f.read(), features="html.parser")

    # Batchnorm folding is always applied.
    assert output_model.applied_bn_folding
    assert_html(html_parsed, {
        'node_batchnorm_folding': _SUCCESS,
        'node_test_batchnorm_folding': _VISITED,
    })

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not output_model.applied_cle
        assert not output_model.applied_adaround
        assert not output_model.applied_amp

        assert_html(html_parsed, {
            'node_cross_layer_equalization': _NOT_VISITED,
            'node_test_cross_layer_equalization': _NOT_VISITED,
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_automatic_mixed_precision': _NOT_VISITED,
            'node_test_automatic_mixed_precision': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    # CLE should be applied if and only if it brings accuracy gain
    assert output_model.applied_cle == (bn_folded_acc < cle_acc)

    assert_html(html_parsed, {
        'node_cross_layer_equalization': _SUCCESS if output_model.applied_cle else _DISCARDED,
        'node_test_cross_layer_equalization': _VISITED,
    })

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert output_model.applied_cle
        assert not output_model.applied_adaround
        assert not output_model.applied_amp

        assert_html(html_parsed, {
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_automatic_mixed_precision': _NOT_VISITED,
            'node_test_automatic_mixed_precision': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    assert output_model.applied_adaround == (adaround_acc >= max(bn_folded_acc, cle_acc))

    assert_html(html_parsed, {
        'node_adaround': _SUCCESS if output_model.applied_adaround else _DISCARDED,
        'node_test_adaround': _VISITED,
    })

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert output_model.applied_adaround

        assert_html(html_parsed, {
            'node_automatic_mixed_precision': _NOT_VISITED,
            'node_test_automatic_mixed_precision': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    if amp_final_acc is None:
        amp_final_acc = -math.inf

    assert output_model.applied_amp == (amp_final_acc >= max(bn_folded_acc, cle_acc, adaround_acc))

    assert_html(html_parsed, {
        'node_automatic_mixed_precision': _SUCCESS if output_model.applied_amp else _DISCARDED,
        'node_test_automatic_mixed_precision': _VISITED,
    })

    assert acc == max(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc)

    # If accuracy is good enough after amp
    if amp_final_acc >= target_acc:
        assert acc == amp_final_acc
        assert output_model.applied_amp

        assert_html(html_parsed, {
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    assert_html(html_parsed, {
        'node_result_fail': _VISITED,
        'node_result_success': _NOT_VISITED,
    })

    if max(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc) == bn_folded_acc:
        assert encoding_path.endswith("batchnorm_folding.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc) == cle_acc:
        assert encoding_path.endswith("cross_layer_equalization.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc) == adaround_acc:
        assert encoding_path.endswith("adaround.encodings")
    else:
        assert encoding_path.endswith("mixed_precision.encodings")


FP32_ACC = .8
W32_ACC = FP32_ACC # Assume W32 accuracy is equal to FP32 accuracy


class _QuantizationSimModel(QuantizationSimModel):
    def set_and_freeze_param_encodings(self, _):
        pass


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
                         fp32_acc=None, w32_acc=None):
    if fp32_acc is None:
        fp32_acc = FP32_ACC

    if w32_acc is None:
        w32_acc = W32_ACC

    const_true = torch.tensor(True, dtype=torch.bool)

    def bn_folding(model: Model, *_, **__):
        model.applied_bn_folding.copy_(const_true)
        return tuple()

    def cle(model: Model, *_, **__):
        model.applied_bn_folding.copy_(const_true)
        model.applied_cle.copy_(const_true)

    def adaround(sim, *_, path=None, filename_prefix=None):
        assert path is not None
        assert filename_prefix is not None
        sim.model.applied_adaround.copy_(const_true)
        SaveUtils.remove_quantization_wrappers(sim.model)
        with open(os.path.join(path, filename_prefix + '.encodings'), "w") as f:
            f.write("")
        return sim.model

    class _GreedyMixedPrecisionAlgo(GreedyMixedPrecisionAlgo):
        def run(self, *args, **kwargs):
            ret = super().run(*args, **kwargs)
            self._final_eval_score = amp_final_acc
            self._sim.model.applied_amp.copy_(const_true)
            return ret

    def mock_eval_callback(model, _):
        if not isinstance(model._conv_0, BaseQuantizationMixin):
            # Not quantized: return fp32 accuracy
            return fp32_acc
        if model._conv_0.param_quantizers["weight"] is None:
            # W32 evaluation for early exit. Return W32 accuracy
            return w32_acc

        acc = bn_folded_acc
        if model.applied_cle:
            acc = cle_acc
        if model.applied_adaround:
            acc = adaround_acc
        return acc

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock
        GreedyMixedPrecisionAlgo: MagicMock

    with patch("aimet_torch.v2.auto_quant.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim,\
            patch("aimet_torch._base.auto_quant.fold_all_batch_norms", side_effect=bn_folding) as mock_bn_folding,\
            patch("aimet_torch._base.auto_quant.equalize_model", side_effect=cle) as mock_cle,\
            patch("aimet_torch.v2.auto_quant.Adaround._apply_adaround", side_effect=adaround) as mock_adaround,\
            patch("aimet_torch.v2.auto_quant.GreedyMixedPrecisionAlgo", side_effect=_GreedyMixedPrecisionAlgo) as mock_amp:
        try:
            yield Mocks(
                eval_callback=mock_eval_callback,
                QuantizationSimModel=mock_qsim,
                fold_all_batch_norms=mock_bn_folding,
                equalize_model=mock_cle,
                apply_adaround=mock_adaround,
                GreedyMixedPrecisionAlgo=mock_amp,
            )
        finally:
            pass


@pytest.fixture(autouse=True)
def patch_dependencies():
    def bokeh_model_factory(*_, **__):
        import bokeh.model
        return MagicMock(bokeh.model.Model())

    with patch("aimet_torch.v2.auto_quant.create_pareto_curve", side_effect=bokeh_model_factory),\
            patch("aimet_torch.v2.auto_quant.create_sensitivity_plot", side_effect=bokeh_model_factory):
         yield


class TestAutoQuant:
    @pytest.mark.parametrize(
        "cle_acc, adaround_acc, amp_final_acc",
        itertools.permutations([.5, .6, .7])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [.05, .15])
    def test_auto_quant_cpu_with_amp(
            self, cpu_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, cle_acc, adaround_acc, amp_final_acc,
    ):
        bn_folded_acc = .4
        self._test_auto_quant_with_amp(
            cpu_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
        )

    @pytest.mark.cuda
    def test_auto_quant_gpu_with_amp(self, gpu_model, dummy_input, unlabeled_data_loader):
        bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7
        allowed_accuracy_drop = .15

        self._test_auto_quant_with_amp(
            gpu_model, dummy_input.cuda(), unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
        )

    def _test_auto_quant_with_amp(
            self, model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
    ):
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])
                self._do_test_optimize_auto_quant(
                    auto_quant, model, dummy_input,
                    allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
                )

    def _do_test_optimize_auto_quant(
            self, auto_quant, input_model, dummy_input,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
    ):
        target_acc = FP32_ACC - allowed_accuracy_drop

        output_model, acc, encoding_path, _ = auto_quant.optimize(allowed_accuracy_drop)

        assert utils.get_device(output_model) == utils.get_device(input_model)
        assert_applied_techniques(
            output_model, acc, encoding_path,
            target_acc, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
            auto_quant._auto_quant_base.results_dir,
        )

    def test_auto_quant_w32_early_exit(self, cpu_model, dummy_input, unlabeled_data_loader):
        allowed_accuracy_drop = 0.1
        w32_acc = FP32_ACC - (allowed_accuracy_drop * 2)

        with create_tmp_directory() as results_dir:
            with patch_ptq_techniques(
                bn_folded_acc=0, cle_acc=0, adaround_acc=0, amp_final_acc=0, w32_acc=w32_acc
            ) as mocks:
                eval_callback = mocks.eval_callback

                w32_eval_called = False
                def _eval_callback(model, _):
                    nonlocal w32_eval_called
                    bw = get_bitwidth(model)
                    if bw:
                        (output_bw, _), (param_bw, _) = bw
                        if param_bw == 32:
                            # Assert W32 evaluation was called with highest output bitwidth
                            assert output_bw == 16
                            w32_eval_called = True
                    return eval_callback(model, _)

                mocks.eval_callback = _eval_callback
                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=True)

                # AMP will early-exit with W8A16 as best candidate.
                # Hence, the final encodings exported by AutoQuant should be also W8A16.
                auto_quant.set_mixed_precision_params(candidates=[_W8A16, _W8A8, _FP16])
                output_model, acc, encoding_path, pareto_curve = auto_quant.optimize(allowed_accuracy_drop)

            assert w32_eval_called
            assert output_model is None
            assert acc is None
            assert encoding_path is None
            assert pareto_curve is None

            with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                html_parsed = BeautifulSoup(f.read(), features="html.parser")
                assert_html(html_parsed, {
                    'node_test_w32_eval_score': _VISITED,
                    'node_batchnorm_folding': _NOT_VISITED,
                    'node_cross_layer_equalization': _NOT_VISITED,
                    'node_adaround': _NOT_VISITED,
                    'node_automatic_mixed_precision': _NOT_VISITED,
                    'node_result_fail': _VISITED,
                })

    def test_auto_quant_with_amp_early_exit(
            self, cpu_model, dummy_input, unlabeled_data_loader
    ):
        """ Assert the encodings are properly set when AMP exits early """
        allowed_accuracy_drop = .0
        bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7

        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
        ) as mocks:
            eval_callback = mocks.eval_callback

            def _eval_callback(model, _):
                bw = get_bitwidth(model)
                if bw == _W8A8:
                    return eval_callback(model, _) * .996 # Discount eval score
                if bw == _W8A16:
                    return eval_callback(model, _) * .998 # Discount eval score
                if bw == _FP16:
                    return eval_callback(model, _) * .999 # Discount eval score
                return eval_callback(model, _)

            mocks.eval_callback = _eval_callback

            auto_quant = AutoQuant(cpu_model,
                                   dummy_input,
                                   unlabeled_data_loader,
                                   mocks.eval_callback)

            # AMP will early-exit with FP16 as best candidate.
            # Hence, the final encodings exported by AutoQuant should be also FP16.
            auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

            with create_tmp_directory() as results_dir:
                _, _, encoding_path, _ = auto_quant.optimize(allowed_accuracy_drop)
                with open(encoding_path) as f:
                    encodings = json.load(f)

                for param_encodings in encodings["param_encodings"].values():
                    for enc in param_encodings:
                        assert enc["bitwidth"] == 16
                        assert enc["dtype"] == "float"

                for activation_encodings in encodings["activation_encodings"].values():
                    for enc in activation_encodings:
                        assert enc["bitwidth"] == 16
                        assert enc["dtype"] == "float"

    def test_auto_quant_invalid_input(self, cpu_model, dummy_input, unlabeled_data_loader):
        auto_quant = AutoQuant(cpu_model,
                               dummy_input,
                               unlabeled_data_loader,
                               MagicMock(),
                               param_bw=8,
                               output_bw=8)

        # AMP candidate doesn't contain the default candidate (W8A8)
        with pytest.raises(ValueError):
            auto_quant.set_mixed_precision_params(candidates=[_W4A8, _W8A16])

        # Baseline candidate (W8A8) isn't the lowest candidate
        with pytest.raises(ValueError):
            auto_quant.set_mixed_precision_params(candidates=[_W4A8, _W8A8, _W8A16])

        # Baseline candidate (W8A8) isn't the lowest candidate
        with pytest.raises(ValueError):
            auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W7A9, _W8A16])

        # Empty candidates
        with pytest.raises(ValueError):
            auto_quant.set_mixed_precision_params(candidates=[])

        # Contains only one candidates
        with pytest.raises(ValueError):
            auto_quant.set_mixed_precision_params(candidates=[_W8A8])

    def test_auto_quant_fallback(
        self, cpu_model, dummy_input, unlabeled_data_loader,
    ):
        class _Exception(Exception):
            pass

        def error_fn(*_, **__):
            raise _Exception

        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc, amp_acc = .4, .5, .6, .7
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch._base.auto_quant.prepare_model", side_effect=error_fn):
                    # If prepare_model fails, should return AMP results
                    _, acc, _, _ = auto_quant.optimize(allowed_accuracy_drop)
                    assert acc == amp_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_prepare_model': _ERROR_IGNORED,
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _SUCCESS,
                            'node_automatic_mixed_precision': _SUCCESS,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch._base.auto_quant.fold_all_batch_norms", side_effect=error_fn):
                    # If batchnorm folding fails, should return AMP results
                    _, acc, _, _ = auto_quant.optimize(allowed_accuracy_drop)
                    assert acc == amp_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _SUCCESS,
                            'node_automatic_mixed_precision': _SUCCESS,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch._base.auto_quant.equalize_model", side_effect=error_fn):
                    # If CLE fails, should return AMP results
                    _, acc, _, _ = auto_quant.optimize(allowed_accuracy_drop)
                    assert acc == amp_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _SUCCESS,
                            'node_automatic_mixed_precision': _SUCCESS,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch.v2.auto_quant.Adaround._apply_adaround", side_effect=error_fn):
                    # If adaround fails, should return AMP results
                    _, acc, _, _ = auto_quant.optimize(allowed_accuracy_drop)
                    assert acc == amp_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _ERROR_IGNORED,
                            'node_automatic_mixed_precision': _SUCCESS,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch.v2.auto_quant.GreedyMixedPrecisionAlgo", side_effect=error_fn):
                    # If AMP fails, should return adaround results
                    _, acc, _, _ = auto_quant.optimize(allowed_accuracy_drop)
                    assert acc == adaround_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _SUCCESS,
                            'node_automatic_mixed_precision': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch._base.auto_quant.fold_all_batch_norms", side_effect=error_fn),\
                        patch("aimet_torch._base.auto_quant.equalize_model", side_effect=error_fn),\
                        patch("aimet_torch.v2.auto_quant.Adaround._apply_adaround", side_effect=error_fn),\
                        patch("aimet_torch.v2.auto_quant.GreedyMixedPrecisionAlgo", side_effect=error_fn):
                    # If everything fails, should raise an error
                    with pytest.raises(RuntimeError):
                        auto_quant.optimize(allowed_accuracy_drop)

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _ERROR_IGNORED,
                            'node_automatic_mixed_precision': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=True)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_torch.v2.auto_quant.Adaround._apply_adaround", side_effect=error_fn):
                    # Hard stop
                    with pytest.raises(_Exception):
                        auto_quant.optimize(allowed_accuracy_drop)

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _ERROR_FAILED,
                            'node_automatic_mixed_precision': _NOT_VISITED,
                        })

    @pytest.mark.skip(reason="GraphModule's custom pickle serialization mechanism doesn't work"
                             "when it contains v2 quantized modules")
    def test_auto_quant_caching(
        self, cpu_model, dummy_input, unlabeled_data_loader,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7
        cache_id = "unittest"
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       cache_id=cache_id,
                                       results_dir=results_dir)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                cache_files  = [
                    os.path.join(results_dir, ".auto_quant_cache", cache_id, f"{key}.pkl")
                    for key in ("batchnorm_folding", "cle", "adaround", "mixed_precision")
                ]

                # No previously cached results
                auto_quant.optimize(allowed_accuracy_drop)

                for cache_file in cache_files:
                    assert os.path.exists(cache_file)

                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       cache_id=cache_id,
                                       results_dir=results_dir)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                # Load cached result
                auto_quant.optimize(allowed_accuracy_drop)

                # PTQ functions should not be called twice.
                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

    def test_auto_quant_adaround_amp_stitching_logic(
        self, cpu_model, dummy_input, unlabeled_data_loader,
    ):
        def run_auto_quant_optimize(param_bw, output_bw, amp_candidates):
            allowed_accuracy_drop = .15
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7
            with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
            ) as mocks:
                eval_callback = mocks.eval_callback

                def _eval_callback(model, _):
                    bw = get_bitwidth(model)
                    if bw == _W4A8:
                        return eval_callback(model, _) * .997 # Discount eval score
                    if bw == _W8A8:
                        return eval_callback(model, _) * .998 # Discount eval score
                    if bw == _W8A16:
                        return eval_callback(model, _) * .999 # Discount eval score
                    return eval_callback(model, _)

                mocks.eval_callback = _eval_callback

                auto_quant = AutoQuant(cpu_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       param_bw=param_bw,
                                       output_bw=output_bw)
                if amp_candidates:
                    auto_quant.set_mixed_precision_params(candidates=amp_candidates)
                auto_quant.optimize(allowed_accuracy_drop)
                return mocks

        """
        Test 1: [ W4A8, W8A8, W8A16, FP16 ]
        """
        mocks = run_auto_quant_optimize(param_bw=4, output_bw=8,
                                        amp_candidates=[_W4A8, _W8A8, _W8A16, _FP16])

        # Adaround should have been called twice, with W4 and W8 respectively
        assert mocks.apply_adaround.call_count == 2

        # If adaround doesn't meet the target accuracy, AMP should be called only with
        # the adaround-compatible candidates
        (_, _, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
        assert set(_candidates) == {_W8A8, _W8A16, _FP16}


        """
        Test 2: [ W8A8, W8A16, FP16 ]
        """
        mocks = run_auto_quant_optimize(param_bw=8, output_bw=8,
                                        amp_candidates=[_W8A8, _W8A16, _FP16])

        # Adaround should have been called only once with W8
        assert mocks.apply_adaround.call_count == 1

        # If adaround doesn't meet the target accuracy, AMP should be called only with
        # the adaround-compatible candidates
        (_, _, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
        assert set(_candidates) == {_W8A8, _W8A16, _FP16}


        """
        Test 3: [ W4A8, W8A8, W8A16 ]
        """
        mocks = run_auto_quant_optimize(param_bw=4, output_bw=8,
                                        amp_candidates=[_W4A8, _W8A8, _W8A16])

        # Adaround should have been called twice, with W4 and W8 respectively
        assert mocks.apply_adaround.call_count == 2

        # If adaround doesn't meet the target accuracy, AMP should be called only with
        # the adaround-compatible candidates
        (_, _, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
        assert set(_candidates) == {_W8A8, _W8A16}


        """
        Test 4: [ W4A8, W8A8 ]
        """
        mocks = run_auto_quant_optimize(param_bw=4, output_bw=8,
                                        amp_candidates=[_W4A8, _W8A8])

        # Adaround should have been called twice, with W4 and W8 respectively
        assert mocks.apply_adaround.call_count == 2

        # After Adaround, we have only one Adaround-compatible candidate left
        # for AMP (W8A8). Therefore, AMP should not be called.
        assert mocks.GreedyMixedPrecisionAlgo.call_count == 0


        """
        Test 5: [ W4A8, FP16 ]
        """
        mocks = run_auto_quant_optimize(param_bw=4, output_bw=8,
                                        amp_candidates=[_W4A8, _FP16])

        # Adaround should have been called only once with W4
        assert mocks.apply_adaround.call_count == 1

        # If adaround doesn't meet the target accuracy, AMP should be called only with
        # the adaround-compatible candidates
        (_, _, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
        assert set(_candidates) == {_W4A8, _FP16}


        """
        Test 6: AMP not enabled
        """
        mocks = run_auto_quant_optimize(param_bw=4, output_bw=8,
                                        amp_candidates=[])

        # Adaround should have been called only once with W4
        assert mocks.apply_adaround.call_count == 1
        assert mocks.GreedyMixedPrecisionAlgo.call_count == 0


@contextlib.contextmanager
def create_tmp_directory(dirname: str = "/tmp/.aimet_unittest"):
    success = False
    try:
        os.makedirs(dirname, exist_ok=True)
        success = True
    except FileExistsError:
        raise

    try:
        yield dirname
    finally:
        if success:
            shutil.rmtree(dirname)


def get_bitwidth(model: torch.nn.Module):
    if not isinstance(model._conv_0, BaseQuantizationMixin):
        return None
    
    param_quantizer = model._conv_0.param_quantizers["weight"]
    if param_quantizer is None:
        param_bw = 32
        param_dtype = QuantizationDataType.int
    else:
        param_bw = param_quantizer.bitwidth
        param_dtype = get_quantizer_quantization_data_type(param_quantizer)

    output_quantizer = model._relu.output_quantizers[0]
    if output_quantizer is None:
        output_bw = 32
        output_dtype = QuantizationDataType.int
    else:
        output_bw = output_quantizer.bitwidth
        output_dtype = get_quantizer_quantization_data_type(output_quantizer)

    return ((output_bw, output_dtype), (param_bw, param_dtype))


def get_quantizer_quantization_data_type(quantizer: QuantizerBase):
    if issubclass(quantizer.__class__, AffineQuantizerBase):
        return QuantizationDataType.int
    elif issubclass(quantizer.__class__, FloatQuantizeDequantize):
        return QuantizationDataType.float
    raise RuntimeError(f"Cannot check quantization data type for quantizer class: {quantizer.__class__}")
