# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import tempfile
from unittest.mock import patch, MagicMock
import os
from bs4 import BeautifulSoup
import pytest
from typing import Callable, Dict, List

import onnx
from onnxruntime.quantization.onnx_quantizer import ONNXModel
import numpy as np

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.adaround.adaround_weight import AdaroundParameters
from aimet_onnx._deprecated.auto_quant_v2 import AutoQuant, PtqResult

from aimet_common.defs import QuantScheme, QuantizationDataType

from .models.models_for_tests import conv_relu_model


@pytest.fixture(scope="function")
def onnx_model():
    model = ONNXModel(conv_relu_model())
    setattr(model, 'applied_bn_folding', False)
    setattr(model, 'applied_cle', False)
    setattr(model, 'applied_adaround', False)
    return model


@pytest.fixture(scope="session")
def dummy_input():
    return {'input': np.random.randn(1, 3, 8, 8).astype(np.float32)}


@pytest.fixture(scope="session")
def unlabeled_data_loader():
    data_loader = [np.random.randn(1, 3, 8, 8) for _ in range(10)]
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
        target_acc, bn_folded_acc, cle_acc, adaround_acc,
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

        assert_html(html_parsed, {
            'node_cross_layer_equalization': _NOT_VISITED,
            'node_test_cross_layer_equalization': _NOT_VISITED,
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
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

        assert_html(html_parsed, {
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
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
        assert encoding_path.endswith("adaround.encodings")
        assert output_model.applied_adaround

        assert_html(html_parsed, {
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    assert_html(html_parsed, {
        'node_result_fail': _VISITED,
        'node_result_success': _NOT_VISITED,
    })

    assert acc == max(bn_folded_acc, cle_acc, adaround_acc)

    if max(bn_folded_acc, cle_acc, adaround_acc) == bn_folded_acc:
        assert encoding_path.endswith("batchnorm_folding.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc) == cle_acc:
        assert encoding_path.endswith("cross_layer_equalization.encodings")
    else:
        assert encoding_path.endswith("adaround.encodings")


FP32_ACC = 0.8
W32_ACC = FP32_ACC # Assume W32 accuracy is equal to FP32 accuracy
RAW_QUANTSIM_ACC = 0.1


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc, fp32_acc=None, w32_acc=None, raw_quantsim_acc=None):
    if fp32_acc is None:
        fp32_acc = FP32_ACC

    if w32_acc is None:
        w32_acc = W32_ACC

    if raw_quantsim_acc is None:
        raw_quantsim_acc = RAW_QUANTSIM_ACC

    def bn_folding(model, *_, **__):
        setattr(model, 'applied_bn_folding', True)
        return [], []

    def cle(model, *_, **__):
        setattr(model, 'applied_bn_folding', True)
        setattr(model, 'applied_cle', True)

    def adaround(sim, model, *_, **__):
        setattr(model, 'applied_adaround', True)
        return model

    class _PtqResult(PtqResult):

        def load_model(self) -> ONNXModel:
            model = super().load_model()
            bnf_val = True if "batchnorm_folding" in self.applied_techniques else False
            cle_val = True if "cross_layer_equalization" in self.applied_techniques else False
            if cle_val:
                bnf_val = True
            ada_val = True if "adaround" in self.applied_techniques else False
            model.__setattr__("applied_bn_folding", bnf_val)
            model.__setattr__("applied_cle", cle_val)
            model.__setattr__("applied_adaround", ada_val)
            return model

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
                            'qsim_w32': True if default_param_bw == 32 else False}

        def compute_encodings(self, *_):
            pass

        def set_and_freeze_param_encodings(self, _):
            pass

    def mock_eval_callback(session, _):
        if isinstance(session, MagicMock):
            # Not quantized: return fp32 accuracy
            return fp32_acc
        if session['qsim_w32']:
            # W32 evaluation for early exit. Return W32 accuracy
            return w32_acc

        acc = raw_quantsim_acc
        if session['applied_bn_folding']:
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
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock
        PtqResult: MagicMock

    with patch("aimet_onnx.auto_quant_v2.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim,\
         patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=bn_folding) as mock_bn_folding,\
         patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=cle) as mock_cle,\
         patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=adaround) as mock_adaround, \
         patch("aimet_onnx.auto_quant_v2._PtqResult", side_effect=_PtqResult) as mock_ptq:
        try:
            yield Mocks(
                eval_callback=mock_eval_callback,
                QuantizationSimModel=mock_qsim,
                fold_all_batch_norms=mock_bn_folding,
                equalize_model=mock_cle,
                apply_adaround=mock_adaround,
                PtqResult=mock_ptq
            )
        finally:
            pass


class TestAutoQuant:
    def test_auto_quant_run_inference(self, onnx_model, dummy_input, unlabeled_data_loader):
        bn_folded_acc = .5

        with patch_ptq_techniques(
            bn_folded_acc, None, None
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                auto_quant.run_inference()

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        itertools.permutations([.5, .6, .7])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [.05, .15])
    def test_auto_quant(
            self, onnx_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        self._test_auto_quant(
            onnx_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    def test_consecutive_calls(self, onnx_model, dummy_input, unlabeled_data_loader):
        bn_folded_acc, cle_acc, adaround_acc = .5, .6, .7

        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)

                # Should return proper model & summary report
                # regardless of consecutive calls
                for allowed_accuracy_drop in (.5, .4, .3, .2, .1, .05):
                    self._do_test_optimize_auto_quant(
                        auto_quant, onnx_model,
                        allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
                    )

        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)

                # When run_inference() and optimize() are called in back-to-back,
                # reusable intermediate reseults should be always reused.
                auto_quant.run_inference()
                auto_quant.optimize()
                assert mocks.fold_all_batch_norms.call_count == 2
                assert mocks.equalize_model.call_count == 1

                auto_quant.optimize()
                assert mocks.fold_all_batch_norms.call_count == 2
                assert mocks.equalize_model.call_count == 1

                self._do_test_optimize_auto_quant(
                    auto_quant, onnx_model,
                    0.0, bn_folded_acc, cle_acc, adaround_acc
                )
                assert mocks.fold_all_batch_norms.call_count == 2
                assert mocks.equalize_model.call_count == 1

    def _test_auto_quant(
            self, model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                self._do_test_optimize_auto_quant(
                    auto_quant, model, allowed_accuracy_drop,
                    bn_folded_acc, cle_acc, adaround_acc
                )

    def _do_test_optimize_auto_quant(
            self, auto_quant, input_model,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
            target_acc = FP32_ACC - allowed_accuracy_drop

            output_model, acc, encoding_path = auto_quant.optimize(allowed_accuracy_drop)

            assert_applied_techniques(
                output_model, acc, encoding_path,
                target_acc, bn_folded_acc, cle_acc, adaround_acc,
                auto_quant.results_dir,
            )

    def test_auto_quant_invalid_input(self, onnx_model, dummy_input, unlabeled_data_loader):
        with pytest.raises(ValueError):
            AutoQuant(None, dummy_input, unlabeled_data_loader, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, None, unlabeled_data_loader, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, None, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, None)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, results_dir=None)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, strict_validation=None)

        # Bitwidth < 4 or bitwidth > 32
        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, param_bw=2)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, param_bw=64)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, output_bw=2)

        with pytest.raises(ValueError):
            AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None, output_bw=64)

        auto_quant = AutoQuant(onnx_model, dummy_input, unlabeled_data_loader, lambda: None)
        # Allowed accuracy drop < 0
        with pytest.raises(ValueError):
            _ = auto_quant.optimize(-1.0)

    def test_auto_quant_inference_fallback(
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        class _Exception(Exception):
            pass

        def error_fn(*_, **__):
            raise _Exception

        bn_folded_acc = .4
        raw_quantsim_acc = bn_folded_acc + 1e-5
        with patch_ptq_techniques(
            bn_folded_acc, None, None, raw_quantsim_acc=raw_quantsim_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=error_fn):
                    # If BN folding fail, should return raw quantsim model
                    _, acc = auto_quant.run_inference()
                    assert acc == raw_quantsim_acc

    def test_auto_quant_optimize_fallback(
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        class _Exception(Exception):
            pass

        def error_fn(*_, **__):
            raise _Exception

        bn_folded_acc, cle_acc, adaround_acc = .4, .5, .6
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=error_fn):
                    # If batchnorm folding fails, should return Adaround results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == adaround_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _SUCCESS,
                        })

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=error_fn):
                    # If CLE fails, should return Adaround results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == adaround_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _SUCCESS,
                        })

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=error_fn):
                    # If adaround fails, should return CLE results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == cle_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_onnx.auto_quant_v2.fold_all_batch_norms_to_weight", side_effect=error_fn),\
                        patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=error_fn),\
                        patch("aimet_onnx.auto_quant_v2.Adaround._apply_adaround", side_effect=error_fn):
                    # If everything fails, should raise an error
                    with pytest.raises(RuntimeError):
                        auto_quant.optimize()

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=True)
                with patch("aimet_onnx.auto_quant_v2.equalize_model", side_effect=error_fn):
                    # Hard stop if strict_validation=True
                    with pytest.raises(_Exception):
                        auto_quant.optimize()

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _ERROR_FAILED,
                            'node_adaround': _NOT_VISITED,
                        })

    def test_auto_quant_early_exit(self, onnx_model, dummy_input, unlabeled_data_loader):
        allowed_accuracy_drop = 0.1
        w32_acc = FP32_ACC - (allowed_accuracy_drop * 2)

        with tempfile.TemporaryDirectory() as results_dir:
            with patch_ptq_techniques(
                bn_folded_acc=0, cle_acc=0, adaround_acc=0, w32_acc=w32_acc
            ) as mocks:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                output_model, acc, encoding_path = auto_quant.optimize(allowed_accuracy_drop)

            assert output_model is None
            assert acc is None
            assert encoding_path is None

            with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                html_parsed = BeautifulSoup(f.read(), features="html.parser")
                assert_html(html_parsed, {
                    'node_test_w32_eval_score': _VISITED,
                    'node_batchnorm_folding': _NOT_VISITED,
                    'node_cross_layer_equalization': _NOT_VISITED,
                    'node_adaround': _NOT_VISITED,
                    'node_result_fail': _VISITED,
                })

    def test_auto_quant_caching(
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = .4, .5, .6
        cache_id = "unittest"

        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       cache_id=cache_id)

                cache_files = [
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
                                       results_dir=results_dir,
                                       cache_id=cache_id)
                # Load cached result
                auto_quant.optimize(allowed_accuracy_drop)

                # PTQ functions should not be called twice.
                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

    def test_auto_quant_scheme_selection(
        self, onnx_model, dummy_input, unlabeled_data_loader,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = .4, .5, .6
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            def eval_callback(session, _):
                # Assumes the model's eval score drops to zero
                # unless param_quant_scheme == tfe and output_quant_scheme == tf
                if isinstance(session, MagicMock):
                    return mocks.eval_callback(session, _)
                if 'param_quant_scheme' in session and session['param_quant_scheme'] != QuantScheme.post_training_tf_enhanced:
                    return 0.0
                if 'output_quant_scheme' in session and session['output_quant_scheme'] != QuantScheme.post_training_tf:
                    return 0.0
                return mocks.eval_callback(session, _)

            _optimize = AutoQuant.optimize
            def optimize(self, *args, **kwargs):
                # Since all the other candidates (tf-tf, tfe-tfe, and tfe-percentile) yields zero accuracy,
                # it is expected that tf-tfe is selected as the quant scheme for AutoQuant.
                ret = _optimize(self, *args, **kwargs)
                assert self._quantsim_params["quant_scheme"].param_quant_scheme == QuantScheme.post_training_tf_enhanced
                assert self._quantsim_params["quant_scheme"].output_quant_scheme == QuantScheme.post_training_tf
                return ret

            _mock_create_quantsim_and_encodings = AutoQuant._create_quantsim_and_encodings
            def mock_create_quantsim_and_encodings(self, *args, **kwargs):
                sim = _mock_create_quantsim_and_encodings(self, *args, **kwargs)
                if 'output_quant_scheme' in kwargs:
                    sim.session['output_quant_scheme'] = kwargs['output_quant_scheme']
                if 'param_quant_scheme' in kwargs:
                    sim.session['param_quant_scheme'] = kwargs['param_quant_scheme']
                return sim

            with patch("aimet_onnx.auto_quant_v2._AutoQuant.optimize", optimize), \
                    patch("aimet_onnx.auto_quant_v2._AutoQuant._create_quantsim_and_encodings", mock_create_quantsim_and_encodings):
                auto_quant = AutoQuant(onnx_model,
                                       dummy_input,
                                       unlabeled_data_loader,
                                       eval_callback)
                auto_quant.optimize(allowed_accuracy_drop)

    def test_set_additional_params(self, onnx_model, dummy_input, unlabeled_data_loader):
        allowed_accuracy_drop = 0
        bn_folded_acc = .1
        cle_acc = .2
        adaround_acc = .3
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            auto_quant = AutoQuant(onnx_model,
                                   dummy_input,
                                   unlabeled_data_loader,
                                   mocks.eval_callback)
            adaround_params = AdaroundParameters(unlabeled_data_loader, 1)
            auto_quant.set_adaround_params(adaround_params)
            self._do_test_optimize_auto_quant(
                auto_quant, onnx_model,
                allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
            )
            adaround_args, _ = mocks.apply_adaround.call_args
            _, _, actual_adaround_params = adaround_args
            assert adaround_params == actual_adaround_params
