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

import copy
import json
import os.path
import tempfile
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import onnxruntime as ort

from aimet_common.utils import CallbackFunc
from aimet_common.defs import QuantScheme
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.quant_analyzer import QuantAnalyzer
from .models import models_for_tests

def calibrate(session: ort.InferenceSession, dummy_input: Dict[str, np.ndarray]):
    """
    Helper function to calibrate model given dummy input
    :param model: ONNX model session.
    :param dummy_input: dummy input to model.
    """
    _ = session.run(None, dummy_input)

def evaluate(session: ort.InferenceSession, dummy_input: Dict[str, np.ndarray]):
    """
    Helper function to evaluate model performance given dummy input
    :param model: ONNX model session.
    :param dummy_input: dummy input to model.
    """
    for i in range(2):
        _ = session.run(None, dummy_input)
    return 0.8

class TestQuantAnalyzer:

    def test_check_model_sensitivity_to_quantization(self):
        """ test analyze_model_sensitivity API """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(calibrate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        fp32_acc, weight_quantized_acc, act_quantized_acc = quant_analyzer.check_model_sensitivity_to_quantization(sim)

        assert fp32_acc >= weight_quantized_acc
        assert fp32_acc >= act_quantized_acc
        assert quant_analyzer._onnx_model is model

    def test_get_enabled_activation_quantizers(self):
        """ test get_enabled_activation_quantizers()  """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        enabled_quantizers = quant_analyzer._get_enabled_activation_quantizers(sim)

        # total 8 activation quantizers (conv + relu is a supergroup) are enabled as per default config file.
        assert len(enabled_quantizers) == 8

    def test_get_enabled_param_quantizers(self):
        """ test get_enabled_param_quantizers() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        enabled_quantizers = quant_analyzer._get_enabled_param_quantizers(sim)

        # total 5 param quantizers are enabled as per default config file.
        assert len(enabled_quantizers) == 5

    def test_get_op_quantizers(self):
        """ test get op quantizers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        conn_graph = sim.connected_graph
        cg_ops = sim.connected_graph.ordered_ops

        # Verify the order of ops is according to their occurences in the model graph
        assert cg_ops.index(conn_graph.get_op_from_module_name("/conv2/Conv")) < cg_ops.index(conn_graph.get_op_from_module_name("/relu2/Relu"))
        assert cg_ops.index(conn_graph.get_op_from_module_name("/conv4/Conv")) < cg_ops.index(conn_graph.get_op_from_module_name("/fc/Gemm"))

        # Verify op-specific quantizers are captured correctly
        input_quantizers, output_quantizers, param_quantizers = quant_analyzer._get_op_quantizers(conn_graph.get_op_from_module_name("/conv1/Conv"), sim)
        assert len(input_quantizers) == 1
        assert not output_quantizers
        assert len(param_quantizers) == 1

        input_quantizers, output_quantizers, param_quantizers = quant_analyzer._get_op_quantizers(conn_graph.get_op_from_module_name("/relu1/Relu"), sim)
        assert not input_quantizers
        assert len(output_quantizers) == 1
        assert not param_quantizers

        # Disable all the quantizers and verify empty items are returned.
        for _, qc_quantize_op in sim.qc_quantize_op_dict.items():
            qc_quantize_op.enabled = False
        for op in cg_ops:
            input_quantizers, output_quantizers, param_quantizers = quant_analyzer._get_op_quantizers(op, sim)
            assert not input_quantizers
            assert not output_quantizers
            assert not param_quantizers

    def test_perform_per_layer_analysis_by_enabling_quantizers(self):
        """ test perform per layer analysis by enabling quantizers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        layer_names = []
        for node in model.nodes():
            layer_names.append(node.name)

        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

        with tempfile.TemporaryDirectory() as tmp_dir:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_enabling_quantizers(sim, results_dir=tmp_dir)
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 10

            # Test whether layer_wise_eval_score_dict consists of correct keys (op names).
            for op_name in layer_wise_eval_score_dict.keys():
                assert op_name in layer_names

            # Check if it is exported to correct html file.
            assert os.path.isfile(Path(tmp_dir, "per_layer_quant_enabled.html"))

    def test_perform_per_layer_analysis_by_disabling_quantizers(self):
        """ test perform per layer analysis by disabling quantizers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        layer_names = []
        for node in model.nodes():
            layer_names.append(node.name)

        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

        with tempfile.TemporaryDirectory() as tmp_dir:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_disabling_quantizers(sim, results_dir=tmp_dir)
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 10

            # Test whether layer_wise_eval_score_dict consists of correct keys (op names).
            for op_name in layer_wise_eval_score_dict.keys():
                assert op_name in layer_names

            # Check if it is exported to correct html file.
            assert os.path.isfile(Path(tmp_dir, "per_layer_quant_disabled.html"))

    def test_export_per_layer_encoding_min_max_range(self):
        """ test export_per_layer_encoding_min_max_range() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir=tmp_dir)
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "weights.html"))
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "activations.html"))

    def test_export_per_layer_encoding_min_max_range_per_channel(self):
        """ test export_per_layer_encoding_min_max_range() for per channel quantization """
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {"Gemm": {"per_channel_quantization": "False"}},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(Path(tmp_dir, "quantsim_config.json"), 'w') as f:
                json.dump(quantsim_config, f)

            input_shape = (1, 3, 32, 32)
            dummy_input = torch.randn(*input_shape)
            model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
            dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
            fold_all_batch_norms_to_weight(model)
            sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict, config_file=Path(tmp_dir, "quantsim_config.json"))
            sim.compute_encodings(evaluate, dummy_input_dict)
            forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
            eval_callback = CallbackFunc(evaluate, dummy_input_dict)
            quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir=tmp_dir)
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "activations.html"))
            # Dense (Gemm) is disabled to per-channel quantization, it should be in weights.html
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "weights.html"))

    def test_export_per_layer_stats_histogram(self):
        """ test export_per_layer_stats_histogram() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_analyzer.export_per_layer_stats_histogram(sim, results_dir=tmp_dir)
            assert os.path.exists(Path(tmp_dir, "activations_pdf"))
            assert os.path.exists(Path(tmp_dir, "weights_pdf"))
            assert len([file for file in os.listdir(os.path.join(tmp_dir, "weights_pdf", "_conv1_Conv")) if
                        file.endswith(".html")]) == 32

    def test_export_per_layer_stats_histogram_per_tensor(self):
        """ test export_per_layer_stats_histogram() for per channel quantization """
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "per_channel_quantization": "False",
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {"Gemm": {"per_channel_quantization": "False"}},
            "supergroups": [],
            "model_input": { "is_input_quantized": "True" },
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(Path(tmp_dir, "quantsim_config.json"), 'w') as f:
                json.dump(quantsim_config, f)

            input_shape = (1, 3, 32, 32)
            dummy_input = torch.randn(*input_shape)
            model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
            dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
            fold_all_batch_norms_to_weight(model)
            sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict, config_file=Path(tmp_dir, "quantsim_config.json"))
            sim.compute_encodings(evaluate, dummy_input_dict)
            forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
            eval_callback = CallbackFunc(evaluate, dummy_input_dict)
            quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)

            quant_analyzer.export_per_layer_stats_histogram(sim, results_dir=tmp_dir)
            assert os.path.exists(Path(tmp_dir, "activations_pdf"))
            assert os.path.exists(Path(tmp_dir, "weights_pdf"))
            assert len([file for file in os.listdir(os.path.join(tmp_dir, "activations_pdf"))]) <= len(sim.activation_names)
            assert len([file for file in os.listdir(os.path.join(tmp_dir, "weights_pdf", "_conv1_Conv")) if file.endswith(".html")]) == 1


    def test_export_per_layer_mse_loss(self):
        """ test export_per_layer_mse_loss() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        unlabeled_dataset_iterable = [dummy_input.numpy() for i in range(10)]
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        layer_names = []
        for node in model.nodes():
            layer_names.append(node.name)

        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        quant_analyzer.enable_per_layer_mse_loss(unlabeled_dataset_iterable, num_batches=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            layerwise_mse_loss_dict = quant_analyzer.export_per_layer_mse_loss(sim, results_dir=tmp_dir)
            assert type(layerwise_mse_loss_dict) == dict
            node_names = [op.name for op in quant_analyzer._onnx_model.nodes()]
            assert set(layerwise_mse_loss_dict.keys()).issubset(node_names)

            # Test whether layerwise_mse_loss_dict consists of correct keys (op names).
            for op_name in layerwise_mse_loss_dict.keys():
                assert op_name in layer_names

            # Check if it is exported to correct html file.
            assert os.path.isfile(Path(tmp_dir, "per_layer_mse_loss.html"))

    def test_analyze(self):
        """ test end to end for analyze() method """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        unlabeled_dataset_iterable = [dummy_input.numpy() for i in range(10)]
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        quant_analyzer.enable_per_layer_mse_loss(unlabeled_dataset_iterable, num_batches=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_analyzer.analyze(quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=8,
                                   default_activation_bw=8,
                                   config_file=None,
                                   results_dir=tmp_dir)

            assert os.path.isfile(Path(tmp_dir, "per_layer_quant_disabled.html"))
            assert os.path.isfile(Path(tmp_dir, "per_layer_quant_enabled.html"))
            assert os.path.exists(Path(tmp_dir, "activations_pdf"))
            assert os.path.exists(Path(tmp_dir, "weights_pdf"))
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "weights.html"))
            assert os.path.isfile(Path(tmp_dir, "min_max_ranges", "activations.html"))
            assert os.path.isfile(Path(tmp_dir, "per_layer_mse_loss.html"))
