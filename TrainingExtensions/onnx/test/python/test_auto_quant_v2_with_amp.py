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
from typing import Callable, Dict, List
from bs4 import BeautifulSoup

import onnx
from onnxruntime.quantization.onnx_quantizer import ONNXModel
import numpy as np

from aimet_common.defs import QuantScheme, QuantizationDataType

from aimet_onnx.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
from aimet_onnx._deprecated.auto_quant_v2 import (
    AutoQuantWithAutoMixedPrecision as AutoQuant,
    PtqResult,
)
from aimet_onnx.quantsim import QuantizationSimModel

from .models.test_models import conv_relu_model


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

@pytest.fixture(scope="function")
def onnx_model():
    model = ONNXModel(conv_relu_model())
    setattr(model, 'applied_bn_folding', False)
    setattr(model, 'applied_cle', False)
    setattr(model, 'applied_adaround', False)
    setattr(model, 'applied_amp', False)
    return model


@pytest.fixture(scope="session")
def dummy_input():
    return {'input': np.random.randn(1, 3, 8, 8).astype(np.float32)}


class DataLoader:
    def __init__(self, batch_size, iterations):
        self._data = np.random.randn(batch_size*iterations, 3, 8, 8)
        self._batch_indx = 0
        self._iterations = iterations
        self.batch_size = batch_size

    def __iter__(self):
        self._batch_indx = 0
        return self

    def __next__(self):
        if self._batch_indx < self._iterations:
            str_idx = self._batch_indx
            end_idx = self._batch_indx + self.batch_size
            self._batch_indx += 1
            return self._data[str_idx:end_idx]
        else:
            raise StopIteration

    def __len__(self):
        return self._iterations


@pytest.fixture(scope="session")
def unlabeled_data_loader():
    data_loader = DataLoader(batch_size=1, iterations=10)
    return data_loader


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


class _QuantizationSimModel(QuantizationSimModel):
    def __init__(self,
                 model: onnx.ModelProto,
                 dummy_input: Dict[str, np.ndarray] = None,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest',
                 default_param_bw: int = 8,
                 default_activation_bw: int = 8,
                 use_symmetric_encodings: bool = False, use_cuda: bool = True,
                 device: int = 0, config_file: str = None, default_data_type: QuantizationDataType = QuantizationDataType.int,
                 user_onnx_libs: List[str] = None):
        super(_QuantizationSimModel, self).__init__(model, dummy_input, quant_scheme, rounding_mode, default_param_bw, default_activation_bw,
                                                    use_symmetric_encodings, use_cuda, device, config_file, default_data_type, user_onnx_libs)

        self.session = {'applied_bn_folding': getattr(model, 'applied_bn_folding'),
                        'applied_cle': getattr(model, 'applied_cle'),
                        'applied_adaround': getattr(model, 'applied_adaround'),
                        'applied_amp': getattr(model, 'applied_amp'),
                        'param_bw': default_param_bw,
                        'activation_bw': default_activation_bw,
                        'qc_quantize_op_dict': self.qc_quantize_op_dict}

    def compute_encodings(self, *_):
        pass

    def set_and_freeze_param_encodings(self, _):
        pass



FP32_ACC = .8
W32_ACC = FP32_ACC # Assume W32 accuracy is equal to FP32 accuracy


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
                         fp32_acc=None, w32_acc=None):
    if fp32_acc is None:
        fp32_acc = FP32_ACC

    if w32_acc is None:
        w32_acc = W32_ACC

    def bn_folding(model, *_, **__):
        setattr(model, 'applied_bn_folding', True)
        return [], []

    def cle(model, *_, **__):
        setattr(model, 'applied_bn_folding', True)
        setattr(model, 'applied_cle', True)

    def adaround(sim, model, *_, **__):
        setattr(model, 'applied_adaround', True)
        return model

    class _GreedyMixedPrecisionAlgo(GreedyMixedPrecisionAlgo):
        def run(self, *args, **kwargs):
            ret = super().run(*args, **kwargs)
            self._final_eval_score = amp_final_acc
            self._sim.model.applied_amp = True
            return ret

    class _PtqResult(PtqResult):
        def load_model(self) -> ONNXModel:
            model = super().load_model()
            bnf_val = True if "batchnorm_folding" in self.applied_techniques else False
            cle_val = True if "cross_layer_equalization" in self.applied_techniques else False
            if cle_val:
                bnf_val = True
            ada_val = True if "adaround" in self.applied_techniques else False
            amp_val = True if "automatic_mixed_precision" in self.applied_techniques else False
            model.__setattr__("applied_bn_folding", bnf_val)
            model.__setattr__("applied_cle", cle_val)
            model.__setattr__("applied_adaround", ada_val)
            model.__setattr__("applied_amp", amp_val)
            return model

    def mock_eval_callback(session, _):
        if isinstance(session, MagicMock):
            # Not quantized: return fp32 accuracy
            return fp32_acc
        if session['param_bw'] == 32:
            # W32 evaluation for early exit. Return W32 accuracy
            return w32_acc

        acc = bn_folded_acc
        if session['applied_cle']:
            acc = cle_acc
        if session['applied_adaround']:
            acc = adaround_acc
        return acc

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        PtqResult: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock
        GreedyMixedPrecisionAlgo: MagicMock

    with patch("aimet_onnx.auto_quant_v2.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim, \
            patch("aimet_onnx.auto_quant_v2._PtqResult", side_effect=_PtqResult) as mock_ptq,\
            patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=bn_folding) as mock_bn_folding,\
            patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=cle) as mock_cle,\
            patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=adaround) as mock_adaround,\
            patch("aimet_onnx.auto_quant_v2.GreedyMixedPrecisionAlgo", side_effect=_GreedyMixedPrecisionAlgo) as mock_amp:
        try:
            yield Mocks(
                eval_callback=mock_eval_callback,
                QuantizationSimModel=mock_qsim,
                PtqResult=mock_ptq,
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

    with patch("aimet_onnx.auto_quant_v2.create_pareto_curve", side_effect=bokeh_model_factory),\
            patch("aimet_onnx.auto_quant_v2.create_sensitivity_plot", side_effect=bokeh_model_factory):
        yield


class TestAutoQuant:
    @pytest.mark.parametrize(
        "cle_acc, adaround_acc, amp_final_acc",
        itertools.permutations([.5, .6, .7])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [.05, .15])
    def test_auto_quant_with_amp(
            self, onnx_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, cle_acc, adaround_acc, amp_final_acc,
    ):
        bn_folded_acc = .4
        self._test_auto_quant_with_amp(
            onnx_model, dummy_input, unlabeled_data_loader,
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

        assert_applied_techniques(
            output_model, acc, encoding_path,
            target_acc, bn_folded_acc, cle_acc, adaround_acc, amp_final_acc,
            auto_quant._auto_quant_base.results_dir,
        )

    def test_auto_quant_w32_early_exit(self, onnx_model, dummy_input, unlabeled_data_loader):
        allowed_accuracy_drop = 0.1
        w32_acc = FP32_ACC - (allowed_accuracy_drop * 2)

        with create_tmp_directory() as results_dir:
            with patch_ptq_techniques(
                bn_folded_acc=0, cle_acc=0, adaround_acc=0, amp_final_acc=0, w32_acc=w32_acc
            ) as mocks:
                eval_callback = mocks.eval_callback

                w32_eval_called = False
                def _eval_callback(session, _):
                    nonlocal w32_eval_called
                    bw = get_bitwidth(session)
                    if bw:
                        (output_bw, _), (param_bw, _) = bw
                        if param_bw == 32:
                            # Assert W32 evaluation was called with highest output bitwidth
                            assert output_bw == 16
                            w32_eval_called = True
                    return eval_callback(session, _)

                mocks.eval_callback = _eval_callback
                auto_quant = AutoQuant(onnx_model,
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
            self, onnx_model, dummy_input, unlabeled_data_loader
    ):
        """ Assert the encodings are properly set when AMP exits early """
        allowed_accuracy_drop = .0
        bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7

        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
        ) as mocks:
            eval_callback = mocks.eval_callback

            def _eval_callback(session, _):
                bw = get_bitwidth(session)
                if bw == _W8A8:
                    return eval_callback(session, _) * .996 # Discount eval score
                if bw == _W8A16:
                    return eval_callback(session, _) * .998 # Discount eval score
                if bw == _FP16:
                    return eval_callback(session, _) * .999 # Discount eval score
                return eval_callback(session, _)

            mocks.eval_callback = _eval_callback

            auto_quant = AutoQuant(onnx_model,
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

    def test_auto_quant_invalid_input(self, onnx_model, dummy_input, unlabeled_data_loader):
        auto_quant = AutoQuant(onnx_model,
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
        self, onnx_model, dummy_input, unlabeled_data_loader,
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
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=error_fn):
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

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=error_fn):
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

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=error_fn):
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

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.GreedyMixedPrecisionAlgo", side_effect=error_fn):
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

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=error_fn),\
                        patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=error_fn),\
                        patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=error_fn),\
                        patch("aimet_onnx.auto_quant_v2.GreedyMixedPrecisionAlgo", side_effect=error_fn):
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

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=True)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                with patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=error_fn):
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

    def test_auto_quant_caching(
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7
        cache_id = "unittest"
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       cache_id=cache_id,
                                       results_dir=results_dir)
                auto_quant.set_mixed_precision_params(candidates=[_W8A8, _W8A16, _FP16])

                cache_files  = [
                    os.path.join(results_dir, ".auto_quant_cache", cache_id, f"{key}.pkl")
                    for key in ("batchnorm_folding", "cle", "adaround")
                ]

                # No previously cached results
                auto_quant.optimize(allowed_accuracy_drop)

                for cache_file in cache_files:
                    assert os.path.exists(cache_file)

                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

                auto_quant = AutoQuant(onnx_model,
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
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        def run_auto_quant_optimize(param_bw, output_bw, amp_candidates):
            allowed_accuracy_drop = .15
            bn_folded_acc, cle_acc, adaround_acc, amp_final_acc = .4, .5, .6, .7
            with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc, amp_final_acc
            ) as mocks:
                eval_callback = mocks.eval_callback

                def _eval_callback(session, _):
                    bw = get_bitwidth(session)
                    if bw == _W4A8:
                        return eval_callback(session, _) * .997 # Discount eval score
                    if bw == _W8A8:
                        return eval_callback(session, _) * .998 # Discount eval score
                    if bw == _W8A16:
                        return eval_callback(session, _) * .999 # Discount eval score
                    return eval_callback(session, _)

                mocks.eval_callback = _eval_callback

                auto_quant = AutoQuant(onnx_model,
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
        (_, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
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
        (_, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
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
        (_, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
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
        (_, _candidates, *_), _ = mocks.GreedyMixedPrecisionAlgo.call_args
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


def get_bitwidth(session):
    if not isinstance(session, Dict):
        return None

    param_quantizer = session['qc_quantize_op_dict']['_conv_0.weight']
    param_bw = param_quantizer.bitwidth
    param_dtype = param_quantizer._data_type

    output_quantizer = session['qc_quantize_op_dict']['output']
    output_bw = output_quantizer.bitwidth
    output_dtype = output_quantizer._data_type

    return ((output_bw, output_dtype), (param_bw, param_dtype))
