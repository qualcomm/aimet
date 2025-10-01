# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import copy
import itertools
import json
import os
import tempfile
import tracemalloc
from unittest.mock import patch
from functools import partial
import pathlib
import time
import random

import onnx.numpy_helper
import torch
import numpy as np
from onnx import load_model
import onnx
import onnxruntime as ort
import pytest
from onnxsim import simplify

from aimet_common import quantsim
from aimet_common import libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType, EncodingType, qtype
from aimet_common.quantsim_config.utils import (
    get_path_for_per_channel_config,
    get_path_for_per_tensor_config,
)
from aimet_common.quantsim import _is_bias_out_of_int32_range
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    load_encodings_to_sim,
    set_blockwise_quantization_for_weights,
    _apply_constraints,
    clamp_activation_encodings,
    set_grouped_blockwise_quantization_for_weights,
    _INT32_MINIMUM_SCALE,
)
import aimet_onnx
from aimet_onnx.qc_quantize_op import OpMode, GroupedBlockQuantizeDequantize
from aimet_onnx.utils import make_dummy_input, disable_quantizers
from aimet_onnx import int8, int16
from .models import models_for_tests, test_models
from .models.models_for_tests import (
    batchnorm_model,
    batchnorm_model_constants,
    BNAfterConv,
    build_dummy_model,
    build_lstm_gru_dummy_model,
    conv_relu,
    custom_add_model,
    depthwise_transposed_conv_model,
    instance_norm_model,
    layernorm_model,
    linear_split_into_matmul_add,
    model_with_split_matmul,
    multi_input_with_constant_model,
    multi_output_model,
    reshape_with_multiple_consumers,
    single_residual_model,
    SingleResidual,
    standalone_batchnorm,
    standalone_batchnorm_constants,
    standalone_gemm,
    standalone_instancenorm,
    standalone_layernorm,
    transposed_conv_model,
    _convert_to_onnx,
    make_model,
)

CPU_PROVIDERS = ["CPUExecutionProvider"]
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _compare_encodings(dst, src):
    return (
        dst.min == src.min
        and dst.max == src.max
        and dst.delta == src.delta
        and dst.offset == src.offset
    )


def _default_callback(session):
    session.run(
        None,
        {
            t.name: np.random.randn(*t.shape).astype(np.float32)
            for t in session.get_inputs()
        },
    )


def _default_callback_with_args(session, args):
    session.run(
        None,
        {
            t.name: np.random.randn(*t.shape).astype(np.float32)
            for t in session.get_inputs()
        },
    )


class DummyModel(SingleResidual):
    """
    Model
    """

    def __init__(self):
        super().__init__()
        # change padding size to 0, onnxruntime only support input size is the factor of output size for pooling
        self.conv4 = torch.nn.Conv2d(
            32, 8, kernel_size=2, stride=2, padding=0, bias=True
        )
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        del self.bn1
        del self.bn2

    def forward(self, inputs):
        x = self.conv1(inputs)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@contextlib.contextmanager
def set_encoding_version(version):
    old_version = quantsim.encoding_version
    quantsim.encoding_version = version

    yield

    quantsim.encoding_version = old_version


class TestQuantSim:
    """Tests for QuantizationSimModel"""

    def test_insert_quantize_op_nodes(self):
        """Test to insert qc quantize op to the graph"""
        model = build_dummy_model()
        dummy_input = make_dummy_input(model)
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            assert len(sim.model.nodes()) == 14

            node_ls = [node.op_type for node in sim.model.nodes()]
            assert (
                node_ls
                == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm"] + ["QcQuantizeOp"] * 9
            )

            # Check if qc quantize op node is correctly connect to the corresponding onnx node
            assert (
                sim.model.find_node_by_name(
                    "QcQuantizeOp_input", [], sim.model.graph()
                ).output[0]
                == sim.model.find_node_by_name("conv", [], sim.model.graph()).input[0]
            )
            # Check if op_mode is set correctly for each qc quantize op node
            qc_quantize_op_dict = sim.get_qc_quantize_op()
            for name in sim.param_names:
                assert (
                    qc_quantize_op_dict[name].op_mode
                    == OpMode.oneShotQuantizeDequantize
                )
            for name in sim.activation_names:
                assert qc_quantize_op_dict[name].op_mode == OpMode.updateStats

    def test_create_quantsim_dynamic_batch_size(self):
        """Test to insert qc quantize op to the graph"""
        model = BNAfterConv()
        inputs = torch.randn((2, 10, 24, 24))
        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                inputs,
                os.path.join(tempdir, "dummy_model.onnx"),
                training=torch.onnx.TrainingMode.PRESERVE,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            onnx_model = load_model(os.path.join(tempdir, "dummy_model.onnx"))
            dummy_input = make_dummy_input(onnx_model)
            sim = QuantizationSimModel(onnx_model, path=tempdir)
            sim.session.run(None, dummy_input)

    @pytest.mark.parametrize("with_context_manager", (True, False))
    def test_compute_encodings(self, with_context_manager):
        """Test to perform compute encodings"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert not qc_op.is_initialized()

            inputs = [make_dummy_input(model) for _ in range(5)]

            if with_context_manager:
                with aimet_onnx.compute_encodings(sim):
                    for item in inputs:
                        sim.session.run(None, item)
            else:
                sim.compute_encodings(inputs)

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.get_encodings()[0].bw == 8

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.is_initialized()
                assert qc_op.op_mode == OpMode.quantizeDequantize

    def test_compute_encodings_with_non_lennable_iterator(self):
        model = build_dummy_model()

        class DataIterator:
            def __init__(self):
                self.iter = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.iter < 5:
                    self.iter += 1
                    return make_dummy_input(model)
                raise StopIteration()

            def __len__(self):
                raise NotImplementedError()

        sim = QuantizationSimModel(model)
        sim.compute_encodings(DataIterator())
        for quantizer in sim.qc_quantize_op_dict.values():
            if quantizer.enabled:
                assert quantizer.is_initialized()

    @pytest.mark.parametrize(
        "args, kwargs",
        (
            ((_default_callback_with_args, None), {}),
            ((_default_callback_with_args,), {"forward_pass_callback_args": None}),
            (
                (),
                {
                    "forward_pass_callback": _default_callback_with_args,
                    "forward_pass_callback_args": None,
                },
            ),
        ),
    )
    def test_compute_encodings_deprecation_warnings(self, args, kwargs):
        model = build_dummy_model()

        sim = QuantizationSimModel(
            copy.deepcopy(model), providers=["CPUExecutionProvider"]
        )
        # Enable all quantizers
        for quantizer in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[quantizer].enabled = True

        # Compute encodings should raise deprecation warning
        with pytest.warns(DeprecationWarning):
            sim.compute_encodings(*args, **kwargs)

        # Assert that all quantizers are initialized
        for name, qc_op in sim.get_qc_quantize_op().items():
            assert qc_op.is_initialized()

    @pytest.mark.parametrize(
        "args, kwargs",
        (
            (
                ([make_dummy_input(build_dummy_model())],),
                {"forward_pass_callback": _default_callback},
            ),  # Inputs and callback provided
            (
                (_default_callback,),
                {"inputs": [make_dummy_input(build_dummy_model())]},
            ),  # Inputs and callback provided
            (
                ([make_dummy_input(build_dummy_model())], None),
                {},
            ),  # Inputs and callback args provided
            (
                ([make_dummy_input(build_dummy_model())],),
                {"forward_pass_callback_args": None},
            ),  # Inputs and callback args provided
            (
                ([make_dummy_input(build_dummy_model())],),
                {"argname": None},
            ),  # Inputs and unknown kwarg passed
            (
                ([make_dummy_input(build_dummy_model())],),
                {"inputs": [make_dummy_input(build_dummy_model())]},
            ),  # inputs provided twice
            (
                (_default_callback,),
                {"forward_pass_callback": _default_callback},
            ),  # Callback passed twice
            (
                (_default_callback_with_args, None),
                {"forward_pass_callback_args": None},
            ),  # Too many arguments
            (
                (),
                {"forward_pass_callback_args": None},
            ),  # Neither inputs nor callback provided
            ((0,), {}),  # Non-iterable or callback first arg
        ),
    )
    def test_compute_encodings_unsupported_signatures(self, args, kwargs):
        model = build_dummy_model()
        sim = QuantizationSimModel(copy.deepcopy(model))

        # Compute encodings should raise TypeError for unsupported signatures
        with pytest.raises(TypeError):
            sim.compute_encodings(*args, **kwargs)

    def test_export_model_with_quant_args(self):
        """Test to export encodings and model"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model,
                activation_type="int16",
                param_type="int16",
                quant_scheme=QuantScheme.post_training_tf,
                path=tempdir,
            )

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def dummy_callback(session):
                session.run(None, make_dummy_input(model))

            sim.compute_encodings(dummy_callback)
            sim.export(tempdir, "quant_sim_model_with_quant_args")
            with open(
                os.path.join(tempdir, "quant_sim_model_with_quant_args.encodings")
            ) as json_file:
                encoding_data = json.load(json_file)

            assert "quantizer_args" in encoding_data
            quantizer_args = encoding_data["quantizer_args"]
            assert quantizer_args["activation_bitwidth"] == 16
            assert quantizer_args["param_bitwidth"] == 16
            assert quantizer_args["per_channel_quantization"]
            assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf.name
            assert quantizer_args["dtype"] == "int"
            assert "is_symmetric" in quantizer_args

    @pytest.mark.parametrize("export_model", (True, False))
    def test_export_model(self, export_model):
        """Test to export encodings and model"""
        model = build_dummy_model()
        dummy_input = make_dummy_input(model)
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def dummy_callback(session):
                session.run(None, make_dummy_input(model))

            sim.compute_encodings(dummy_callback)

            nodes_before_export = copy.deepcopy(sim.model.model.graph.node)
            output_before_export = sim.session.run(None, dummy_input)[0]

            sim.export(tempdir, "quant_sim_model", export_model=export_model)

            # model.graph should not be changed
            for node in nodes_before_export:
                assert node in sim.model.model.graph.node

            # Output should not change after export
            sim._rebuild_session()
            output_after_export = sim.session.run(None, dummy_input)[0]
            assert np.allclose(output_before_export, output_after_export)
            assert (
                os.path.exists(os.path.join(tempdir, "quant_sim_model.onnx"))
                == export_model
            )

            with open(
                os.path.join(tempdir, "quant_sim_model.encodings"), "rb"
            ) as json_file:
                encoding_data = json.load(json_file)

            activation_names = {
                encoding["name"] for encoding in encoding_data["activation_encodings"]
            }
            param_names = {
                encoding["name"] for encoding in encoding_data["param_encodings"]
            }
            assert activation_names == {"3", "4", "5", "input", "output"}
            assert param_names == {"conv_b", "conv_w", "fc_b", "fc_w"}

            if export_model:
                model_path = os.path.join(tempdir, "quant_sim_model.onnx")
                onnx.checker.check_model(model_path)
                model = onnx.load(model_path)
                # Exported graph should not have any QcQuantizeOps
                assert not any(
                    node.op_type == "QcQuantizeOp" for node in model.graph.node
                )
                # Exported graph should not have updated output names
                assert not any(
                    output.name.endswith("updated") for output in model.graph.output
                )

    def test_export_model_1_0_0(self):
        """Test to export encodings and model in 1.0.0 format"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model, path=tempdir, config_file=get_path_for_per_channel_config()
            )

            sim.compute_encodings([make_dummy_input(model)])
            with set_encoding_version("1.0.0"):
                sim.export(tempdir, "quant_sim_model")

            with open(
                os.path.join(tempdir, "quant_sim_model.encodings"), "rb"
            ) as json_file:
                encoding_data = json.load(json_file)

            assert encoding_data["version"] == "1.0.0"
            assert isinstance(encoding_data["activation_encodings"], list)
            assert isinstance(encoding_data["param_encodings"], list)

            activation_keys = {
                enc["name"] for enc in encoding_data["activation_encodings"]
            }
            param_keys = {enc["name"] for enc in encoding_data["param_encodings"]}
            assert activation_keys == {"4", "input", "output"}
            assert param_keys == {"conv_w", "fc_w"}

            for enc in itertools.chain(
                encoding_data["param_encodings"], encoding_data["activation_encodings"]
            ):
                assert isinstance(enc, dict)
                assert enc.keys() == {
                    "name",
                    "enc_type",
                    "dtype",
                    "bw",
                    "is_sym",
                    "scale",
                    "offset",
                }
                assert isinstance(enc["scale"], list)
                assert enc["dtype"] == "INT"
                # Gemm layers do not use per-channel in the default_per_channel_config
                if enc["name"] == "conv_w":
                    assert enc["enc_type"] == EncodingType.PER_CHANNEL.name
                else:
                    assert enc["enc_type"] == EncodingType.PER_TENSOR.name

    @pytest.mark.skip(
        reason="FIXME: LSTM with per-channel quantzation fails at QuantizationSimModel.__init__"
    )
    def test_lstm_gru(self):
        """Test for LSTM and GRU dummy model"""
        model = build_lstm_gru_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def callback(session):
                in_tensor = {"input": np.random.rand(1, 8, 64).astype(np.float32)}
                session.run(None, in_tensor)

            sim.compute_encodings(callback)

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.get_encodings()[0].bw == 8

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.is_initialized()
                assert qc_op.op_mode == OpMode.quantizeDequantize

            sim.export(tempdir, "quant_sim_model")

            with open(
                os.path.join(tempdir, "quant_sim_model.encodings"), "rb"
            ) as json_file:
                encoding_data = json.load(json_file)

            activation_names = {
                encoding["name"] for encoding in encoding_data["activation_encodings"]
            }
            param_names = {
                encoding["name"] for encoding in encoding_data["param_encodings"]
            }
            assert activation_names == {"2", "input", "output"}
            assert param_names == {"gru_r_w", "gru_w", "lstm_r_w", "lstm_w"}

    def test_single_residual(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model, providers=["CPUExecutionProvider"], path=tempdir
            )
            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            sim.compute_encodings(inputs=[make_dummy_input(model)])
            sim.export(tempdir, "quant_sim_model")

            with open(
                os.path.join(tempdir, "quant_sim_model.encodings"), "rb"
            ) as json_file:
                encoding_data = json.load(json_file)

            assert len(encoding_data["activation_encodings"]) + len(
                encoding_data["param_encodings"]
            ) == len(sim.qc_quantize_op_dict.keys())

            # Check that exported model is the same as original model
            model = single_residual_model().model
            exported_model = onnx.load(os.path.join(tempdir, "quant_sim_model.onnx"))

            for idx, t in enumerate(model.graph.input):
                assert t.name == exported_model.graph.input[idx].name

            for idx, t in enumerate(model.graph.output):
                assert t.name == exported_model.graph.output[idx].name

            model_cg = ConnectedGraph(model)
            exported_cg = ConnectedGraph(exported_model)
            for name, op in model_cg.get_all_ops().items():
                for idx, tensor in enumerate(op.inputs):
                    assert (
                        tensor.name == exported_cg.get_all_ops()[name].inputs[idx].name
                    )

                for idx, tensor in enumerate(op.outputs):
                    assert (
                        tensor.name == exported_cg.get_all_ops()[name].outputs[idx].name
                    )

    def test_insert_quantizer(self):
        model = single_residual_model().model
        reshape_output = next(
            iter(op.output[0] for op in model.graph.node if op.op_type == "Reshape")
        )
        sim = QuantizationSimModel(model, providers=["CPUExecutionProvider"])
        sim._insert_quantizer(reshape_output, is_param=False)
        sim.activation_names.append(reshape_output)
        sim._rebuild_session()
        sim.compute_encodings([make_dummy_input(model)])
        assert sim.qc_quantize_op_dict[reshape_output].get_encodings() is not None

    @pytest.mark.parametrize(
        "act_enc, param_enc",
        [
            (
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "offset": [0.0],
                    "scale": [0.023529411764705882],
                },
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "offset": [0.0],
                    "scale": [0.023529411764705882],
                },
            ),
            (
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "offset": [0.0],
                    "scale": [0.023529411764705882],
                },
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_CHANNEL",
                    "is_sym": True,
                    "offset": [-128.0] * 10,
                    "scale": [0.023529411764705882] * 10,
                },
            ),
        ],
    )
    @pytest.mark.parametrize("allow_overwrite", [True, False])
    def test_load_encodings_to_sim_with_allow_overwrite(
        self, allow_overwrite, act_enc, param_enc
    ):
        model = test_models.model_with_split().model

        sim = QuantizationSimModel(
            model,
            providers=CPU_PROVIDERS,
        )

        encodings = {
            "param_encodings": [
                {"name": "conv1.weight", **param_enc},
            ],
            "activation_encodings": [
                {"name": "input", **act_enc},
                {"name": "/conv1/Conv_output_0", **act_enc},
            ],
            "version": "1.0.0",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            enc_path = os.path.join(temp_dir, "model.encodings")
            with open(enc_path, "w") as f:
                json.dump(encodings, f, indent=4)

            load_encodings_to_sim(
                sim, enc_path, strict=False, allow_overwrite=allow_overwrite
            )

            sim.compute_encodings([make_dummy_input(model)])

            # If allow_overwrite is True, compute_encodings call will re-write the encodings.
            # If allow_overwrite is False, loaded encodings will be frozen
            for encoding in itertools.chain(
                encodings["activation_encodings"], encodings["param_encodings"]
            ):
                sim_enc = sim.qc_quantize_op_dict[encoding["name"]].export_encodings(
                    "1.0.0"
                )
                enc = (
                    act_enc
                    if encoding in encodings["activation_encodings"]
                    else param_enc
                )
                assert sim_enc != enc if allow_overwrite else sim_enc == enc

    def test_load_partial_encodings_to_sim(self, tmp_path):
        model = single_residual_model().model
        sim = QuantizationSimModel(copy.deepcopy(model))
        for name in sim.activation_names:
            sim.qc_quantize_op_dict[name].enabled = False

        sim.compute_encodings([make_dummy_input(model)])
        sim.export(tmp_path, "model")

        sim = QuantizationSimModel(copy.deepcopy(model))
        enabled_quantizers = {
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }
        load_encodings_to_sim(
            sim,
            os.path.join(tmp_path, "model.encodings"),
            strict=False,
            disable_missing_quantizers=False,
        )
        # No quantizers should be disabled by this
        enabled_quantizers_after_load = {
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }
        assert enabled_quantizers_after_load == enabled_quantizers
        # None of the activation quantizers should be initialized
        assert not any(
            sim.qc_quantize_op_dict[name].is_initialized()
            for name in sim.activation_names
        )
        # All of the enabled param quantizers should be initialized
        assert all(
            sim.qc_quantize_op_dict[name].is_initialized()
            for name in sim.param_names
            if sim.qc_quantize_op_dict[name].enabled
        )

    @pytest.mark.parametrize(
        "param_type, act_type",
        [(aimet_onnx.int8, aimet_onnx.int16), (aimet_onnx.int8, aimet_onnx.int8)],
    )
    def test_fixed_range_for_model_inputs(self, param_type, act_type):
        def onnx_callback(session, inputs):
            in_tensor = {"input": inputs}
            session.run(None, in_tensor)

        np.random.seed(0)
        torch.manual_seed(0)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()

        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                torch.as_tensor(inputs),
                os.path.join(tempdir, "dummy_model.onnx"),
                training=torch.onnx.TrainingMode.PRESERVE,
                input_names=["input"],
                output_names=["output"],
            )

            onnx_model_cpu = load_model(os.path.join(tempdir, "dummy_model.onnx"))

            onnx_sim_cpu = QuantizationSimModel(
                onnx_model_cpu,
                param_type=param_type,
                activation_type=act_type,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                path=tempdir,
            )

            onnx_sim_cpu.qc_quantize_op_dict["input"].set_fixed_encoding_range(
                (0, 255.0)
            )
            onnx_sim_cpu.qc_quantize_op_dict["output"].set_fixed_encoding_range(
                (0.0, 1.0)
            )

            onnx_sim_cpu.compute_encodings(onnx_callback, inputs)

            assert onnx_sim_cpu.qc_quantize_op_dict["input"].encodings[0].min == 0.0
            assert onnx_sim_cpu.qc_quantize_op_dict["input"].encodings[0].max == 255.0

            assert onnx_sim_cpu.qc_quantize_op_dict["output"].encodings[0].min == 0.0
            assert onnx_sim_cpu.qc_quantize_op_dict["output"].encodings[0].max == 1.0

            out_cpu = onnx_sim_cpu.session.run(None, {"input": inputs})[0]

    @pytest.mark.cuda
    def test_compare_encodings_cpu_gpu(self):
        """Test to compare encodings with PT"""

        def onnx_callback(session, inputs):
            in_tensor = {"input": inputs}
            session.run(None, in_tensor)

        np.random.seed(0)
        torch.manual_seed(0)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()

        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                torch.as_tensor(inputs),
                os.path.join(tempdir, "dummy_model.onnx"),
                training=torch.onnx.TrainingMode.PRESERVE,
                input_names=["input"],
                output_names=["output"],
            )

            onnx_model_cpu = load_model(os.path.join(tempdir, "dummy_model.onnx"))
            onnx_model_gpu = load_model(os.path.join(tempdir, "dummy_model.onnx"))

            onnx_sim_cpu = QuantizationSimModel(
                onnx_model_cpu,
                providers=CPU_PROVIDERS,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                path=tempdir,
            )
            onnx_sim_gpu = QuantizationSimModel(
                onnx_model_gpu,
                providers=CUDA_PROVIDERS,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                path=tempdir,
            )

            for node in onnx_sim_gpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        assert node.domain == "aimet.customop.cuda"
            for node in onnx_sim_cpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    assert node.domain == "aimet.customop.cpu"

            onnx_sim_cpu.compute_encodings(onnx_callback, inputs)
            onnx_sim_gpu.compute_encodings(onnx_callback, inputs)
            out_cpu = onnx_sim_cpu.session.run(None, {"input": inputs})[0]
            out_gpu = onnx_sim_gpu.session.run(None, {"input": inputs})[0]
            onnx_sim_cpu.export(tempdir, "onnx_sim_cpu")
            onnx_sim_gpu.export(tempdir, "onnx_sim_gpu")

            assert np.max(np.abs(out_cpu - out_gpu)) < 0.05
            print(np.max(np.abs(out_cpu - out_gpu)))

            with open(os.path.join(tempdir, "onnx_sim_cpu.encodings")) as f:
                cpu_encodings = json.load(f)
            with open(os.path.join(tempdir, "onnx_sim_gpu.encodings")) as f:
                gpu_encodings = json.load(f)

            for i, name in enumerate(cpu_encodings["activation_encodings"]):
                assert (
                    np.max(
                        np.abs(
                            cpu_encodings["activation_encodings"][i]["scale"][0]
                            - gpu_encodings["activation_encodings"][i]["scale"][0]
                        )
                    )
                    < 0.05
                )
                assert (
                    cpu_encodings["activation_encodings"][i]["offset"]
                    == gpu_encodings["activation_encodings"][i]["offset"]
                )

            for i, name in enumerate(cpu_encodings["param_encodings"]):
                # Comparing the scale for first channel only
                assert (
                    np.max(
                        np.abs(
                            cpu_encodings["param_encodings"][i]["scale"][0]
                            - gpu_encodings["param_encodings"][i]["scale"][0]
                        )
                    )
                    < 0.05
                )
                assert (
                    cpu_encodings["param_encodings"][i]["offset"]
                    == gpu_encodings["param_encodings"][i]["offset"]
                )

    @pytest.mark.cuda
    def test_compare_encodings_cpu_gpu_fp16(self):
        """Test to compare encodings with PT"""
        np.random.seed(0)
        torch.manual_seed(0)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                torch.as_tensor(inputs),
                os.path.join(tempdir, "dummy_model.onnx"),
                training=torch.onnx.TrainingMode.PRESERVE,
                input_names=["input"],
                output_names=["output"],
            )

            onnx_model_cpu = load_model(os.path.join(tempdir, "dummy_model.onnx"))
            onnx_model_gpu = load_model(os.path.join(tempdir, "dummy_model.onnx"))

            onnx_sim_cpu = QuantizationSimModel(
                onnx_model_cpu,
                providers=CPU_PROVIDERS,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                param_type="float16",
                activation_type="float16",
                path=tempdir,
            )
            onnx_sim_gpu = QuantizationSimModel(
                onnx_model_gpu,
                providers=CUDA_PROVIDERS,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                param_type="float16",
                activation_type="float16",
                path=tempdir,
            )

            for node in onnx_sim_gpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        assert node.domain == "aimet.customop.cuda"
            for node in onnx_sim_cpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    assert node.domain == "aimet.customop.cpu"

            out_cpu = onnx_sim_cpu.session.run(None, {"input": inputs})[0]
            out_gpu = onnx_sim_gpu.session.run(None, {"input": inputs})[0]

            assert np.max(np.abs(out_cpu - out_gpu)) < 0.05

    def test_per_channel_quantization(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model,
                providers=CPU_PROVIDERS,
                config_file=get_path_for_per_channel_config(),
                path=tempdir,
            )

            def dummy_callback(session, args):
                in_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            sim.qc_quantize_op_dict["fc.weight"].enable_per_channel_quantization()
            sim.compute_encodings(inputs=[make_dummy_input(model)])

            sim.export(tempdir, "encodings")
            with open(os.path.join(tempdir, "encodings.encodings")) as json_file:
                encoding_data = json.load(json_file)
                param_encodings = {
                    encoding["name"]: encoding
                    for encoding in encoding_data["param_encodings"]
                }

            for param_name in sim.param_names:
                qc_op = sim.qc_quantize_op_dict[param_name]
                if qc_op.quant_info.usePerChannelMode and qc_op.enabled:
                    num_channels = qc_op.tensor_quantizer_params.tensor_shape[
                        qc_op.tensor_quantizer_params.channel_axis
                    ]
                    assert num_channels == len(qc_op.get_encodings())
                    assert num_channels == len(param_encodings[param_name]["scale"])
                    for encoding in qc_op.get_encodings():
                        assert encoding.bw == 8
                        assert encoding.min != encoding.max

    @pytest.mark.parametrize(
        "model_factory", (transposed_conv_model, depthwise_transposed_conv_model)
    )
    def test_per_channel_quant_conv_transpose(self, model_factory):
        model = model_factory()
        conv_transpose_weight_names = []
        for node in model.graph().node:
            if node.op_type == "ConvTranspose":
                conv_transpose_weight_names.append(node.input[1])

        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model,
                providers=CPU_PROVIDERS,
                config_file=get_path_for_per_channel_config(),
                path=tempdir,
            )

            def dummy_callback(session, args):
                in_tensor = {"input": np.random.rand(10, 10, 4, 4).astype(np.float32)}
                session.run(None, in_tensor)

            with aimet_onnx.compute_encodings(sim):
                dummy_callback(sim.session, None)

            for param_name in sim.param_names:
                if param_name in conv_transpose_weight_names:
                    for weight in sim.model.graph().initializer:
                        if weight.name == param_name:
                            break
                    else:
                        raise RuntimeError(f"Param {param_name} not found in model")
                    qc_op = sim.qc_quantize_op_dict[param_name]
                    assert qc_op.quant_info.usePerChannelMode
                    assert qc_op.quant_info.enabled
                    assert qc_op.quant_info.channelAxis == 1
                    assert len(qc_op.get_encodings()) == weight.dims[1]

    def test_load_encodings_ptq(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(copy.deepcopy(model), path=tempdir)

            dummy_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

            sim.compute_encodings([dummy_tensor])
            sim.export(tempdir, "onnx_sim")

            out2 = sim.session.run(None, dummy_tensor)

            del sim

            sim = QuantizationSimModel(copy.deepcopy(model), path=tempdir)
            load_encodings_to_sim(sim, os.path.join(tempdir, "onnx_sim.encodings"))
            out3 = sim.session.run(None, dummy_tensor)

            assert np.allclose(out2, out3)

    def test_load_encodings_pcq(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                copy.deepcopy(model),
                config_file=get_path_for_per_channel_config(),
                path=tempdir,
            )

            dummy_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

            sim.compute_encodings((dummy_tensor,))
            sim.export(tempdir, "onnx_sim")

            out2 = sim.session.run(None, dummy_tensor)

            del sim

            sim = QuantizationSimModel(
                copy.deepcopy(model),
                config_file=get_path_for_per_channel_config(),
                path=tempdir,
            )
            load_encodings_to_sim(sim, os.path.join(tempdir, "onnx_sim.encodings"))
            out3 = sim.session.run(None, dummy_tensor)
            assert np.allclose(out2, out3)

    def test_load_encodings_assertion(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model, config_file=get_path_for_per_channel_config(), path=tempdir
            )

            def callback(session, args):
                in_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            with aimet_onnx.compute_encodings(sim):
                callback(sim.session, None)

            sim.export(tempdir, "onnx_sim")
            model = multi_output_model().model
            sim = QuantizationSimModel(model, path=tempdir)
            with pytest.raises(AssertionError):
                load_encodings_to_sim(
                    sim, os.path.join(tempdir, "onnx_sim.encodings"), strict=False
                )

    def test_load_encodings_with_missing_quantizer(self, tmp_path):
        model = models_for_tests.conv_relu_model()
        sim = QuantizationSimModel(
            copy.deepcopy(model), providers=["CPUExecutionProvider"], path=tmp_path
        )
        dummy_input = make_dummy_input(sim.model.model)

        sim.compute_encodings([make_dummy_input(model)])
        quantized_tensors = {
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }
        output = sim.session.run(None, dummy_input)
        sim.export(tmp_path, "onnx_sim")

        # Create a new quantsim model
        sim_2 = QuantizationSimModel(
            copy.deepcopy(model), providers=["CPUExecutionProvider"], path=tmp_path
        )

        # Clear all quantizers from the sim
        for node in list(sim_2.model.graph().node):
            if node.op_type == "QcQuantizeOp":
                sim_2.model.graph().node.remove(node)
                sim_2.model.replace_input_of_all_nodes(node.output[0], node.input[0])
        sim_2.qc_quantize_op_dict = {}

        # Loading encodings with strict=False should re-load all the quantizers
        load_encodings_to_sim(
            sim_2, os.path.join(tmp_path, "onnx_sim.encodings"), strict=False
        )
        loaded_quantized_tensors = {
            name for name, q in sim_2.qc_quantize_op_dict.items() if q.enabled
        }
        assert loaded_quantized_tensors == quantized_tensors

        # Outputs should exactly match after loading
        output_after_load = sim_2.session.run(None, dummy_input)
        for tensor1, tensor2 in zip(output, output_after_load):
            assert np.all(tensor1 == tensor2)

    @pytest.mark.parametrize("strict", [False, True])
    def test_load_encodings_strict_and_non_strict(self, strict):
        torch.random.manual_seed(0)
        np.random.seed(0)
        model = single_residual_model().model
        output_name = model.graph.output[0].name

        # Update weights for testing is_unsigned_symmetric override later
        weight_initializers = [
            i.name for i in model.graph.initializer if len(i.dims) > 1
        ]
        weight_initializer_3 = [
            i for i in model.graph.initializer if i.name == weight_initializers[3]
        ][0]
        weight_initializer_3_data = onnx.numpy_helper.to_array(weight_initializer_3)
        weight_initializer_3.raw_data = np.asarray(
            np.abs(weight_initializer_3_data), dtype=np.float32
        ).tobytes()

        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(copy.deepcopy(model), path=tempdir)

            conv_ops = [
                node for node in sim.model.model.graph.node if node.op_type == "Conv"
            ]
            relu_ops = [
                node for node in sim.model.model.graph.node if node.op_type == "Relu"
            ]
            avgpool_ops = [
                node
                for node in sim.model.model.graph.node
                if node.op_type == "AveragePool"
            ]

            act_1 = conv_ops[0].output[0]
            act_2 = relu_ops[0].output[0]
            act_3 = avgpool_ops[0].output[0]
            act_4 = conv_ops[2].output[0]
            sim.get_qc_quantize_op()[act_1].enabled = True
            sim.get_qc_quantize_op()[act_2].enabled = False
            sim.get_qc_quantize_op()[act_3].data_type = QuantizationDataType.float
            sim.get_qc_quantize_op()[weight_initializers[0]].bitwidth = 16
            sim.get_qc_quantize_op()[act_4].bitwidth = 4
            sim.get_qc_quantize_op()[
                weight_initializers[1]
            ].use_symmetric_encodings = False
            sim.get_qc_quantize_op()[weight_initializers[2]].use_strict_symmetric = True
            sim.get_qc_quantize_op()[
                weight_initializers[3]
            ].use_unsigned_symmetric = True

            def callback(session, args):
                in_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            dummy_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

            with aimet_onnx.compute_encodings(sim):
                callback(sim.session, None)
            sim.export(tempdir, "onnx_sim")
            out2 = sim.session.run(None, dummy_tensor)
            del sim

            sim = QuantizationSimModel(copy.deepcopy(model), path=tempdir)
            if strict:
                with pytest.raises(AssertionError):
                    load_encodings_to_sim(
                        sim, os.path.join(tempdir, "onnx_sim.encodings"), strict=strict
                    )
            else:
                mismatched_encodings = load_encodings_to_sim(
                    sim, os.path.join(tempdir, "onnx_sim.encodings"), strict=strict
                )
                out3 = sim.session.run(None, dummy_tensor)
                sim.export(tempdir, "loaded_onnx_sim")

                assert sim.get_qc_quantize_op()[act_1].enabled
                assert not sim.get_qc_quantize_op()[act_2].enabled
                assert (
                    sim.get_qc_quantize_op()[act_3].data_type
                    == QuantizationDataType.float
                )
                assert sim.get_qc_quantize_op()[weight_initializers[0]].bitwidth == 16
                assert sim.get_qc_quantize_op()[act_4].bitwidth == 4
                assert not sim.get_qc_quantize_op()[
                    weight_initializers[1]
                ].use_symmetric_encodings
                assert sim.get_qc_quantize_op()[
                    weight_initializers[2]
                ].use_strict_symmetric
                assert sim.get_qc_quantize_op()[
                    weight_initializers[3]
                ].use_unsigned_symmetric
                assert len(mismatched_encodings) == 8
                assert np.allclose(
                    out2,
                    out3,
                    atol=sim.qc_quantize_op_dict[output_name].get_encodings()[0].delta,
                )  # Bit flip is possible from recomputing min/max during load

    @pytest.mark.parametrize(
        "swap_quantizer_func, is_lpbq",
        [
            (
                partial(
                    set_grouped_blockwise_quantization_for_weights,
                    op_types=("MatMul", "Conv", "Gemm"),
                    decompressed_bw=8,
                    strict=False,
                ),
                True,
            ),
            (
                partial(
                    set_blockwise_quantization_for_weights,
                    op_types=("MatMul", "Conv", "Gemm"),
                    strict=False,
                    symmetric=True,
                ),
                False,
            ),
        ],
    )
    def test_load_per_block_and_lpbq_encodings(self, swap_quantizer_func, is_lpbq):
        torch.manual_seed(0)
        np.random.seed(0)
        model = single_residual_model()
        model_2 = copy.deepcopy(model)
        model_3 = copy.deepcopy(model)
        dummy_input = make_dummy_input(model.model)
        bq_layers = ("MatMul", "Conv", "Gemm")
        bq_weights = set()

        for node in model.graph().node:
            if node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(model, param_type="int16", activation_type="int16")
        swap_quantizer_func(sim=sim, bitwidth=4, block_size=4)

        sim.compute_encodings([dummy_input])
        out1 = sim.session.run(None, dummy_input)
        with tempfile.TemporaryDirectory() as tempdir, set_encoding_version("1.0.0"):
            sim.export(tempdir, "export")

            sim_2 = QuantizationSimModel(
                model_2, param_type="int16", activation_type="int16"
            )
            swap_quantizer_func(sim=sim_2, bitwidth=4, block_size=4)

            load_encodings_to_sim(
                sim_2, os.path.join(tempdir, "export.encodings"), strict=True
            )
            out2 = sim_2.session.run(None, dummy_input)
            sim_2.export(tempdir, "export_2")
            with open(os.path.join(tempdir, "export.encodings"), "rb") as f1:
                encodings_1 = json.load(f1)
            with open(os.path.join(tempdir, "export_2.encodings"), "rb") as f2:
                encodings_2 = json.load(f2)
            assert encodings_1 == encodings_2
            seen_lpbq_or_per_block = False
            for encoding in encodings_1["param_encodings"]:
                if is_lpbq:
                    if encoding["enc_type"] == "LPBQ":
                        seen_lpbq_or_per_block = True
                else:
                    if encoding["enc_type"] == "PER_BLOCK":
                        seen_lpbq_or_per_block = True
            assert seen_lpbq_or_per_block
            assert np.allclose(out1, out2)

            sim_3 = QuantizationSimModel(
                model_3, param_type="int16", activation_type="int16"
            )

            # TODO: switch to strict=True when we support swapping to LPBQ quantizer from non-LPBQ quantizer
            if is_lpbq:
                with pytest.raises(AssertionError):
                    load_encodings_to_sim(
                        sim_3, os.path.join(tempdir, "export.encodings"), strict=False
                    )

    @pytest.mark.parametrize(
        "swap_quantizer_func, is_lpbq",
        [
            (
                partial(
                    set_grouped_blockwise_quantization_for_weights,
                    op_types=("ConvTranspose",),
                    decompressed_bw=8,
                    strict=True,
                ),
                True,
            ),
            (
                partial(
                    set_blockwise_quantization_for_weights,
                    op_types=("ConvTranspose",),
                    strict=True,
                    symmetric=True,
                ),
                False,
            ),
        ],
    )
    def test_load_per_block_and_lpbq_conv_transpose(self, swap_quantizer_func, is_lpbq):
        torch.manual_seed(0)
        np.random.seed(0)
        model = models_for_tests.pointwise_convtranspose1d((1, 64, 32))
        model_2 = copy.deepcopy(model)
        sim = QuantizationSimModel(model)
        swap_quantizer_func(sim=sim, bitwidth=4, block_size=4)
        dummy_input = make_dummy_input(model)

        sim.compute_encodings([dummy_input])
        out1 = sim.session.run(None, dummy_input)
        with tempfile.TemporaryDirectory() as tempdir, set_encoding_version("1.0.0"):
            sim.export(tempdir, "export")

            sim2 = QuantizationSimModel(model_2)
            swap_quantizer_func(sim=sim2, bitwidth=4, block_size=4)

            load_encodings_to_sim(
                sim2, os.path.join(tempdir, "export.encodings"), strict=True
            )
            out2 = sim2.session.run(None, dummy_input)

            sim2.export(tempdir, "export_2")
            with open(os.path.join(tempdir, "export.encodings"), "rb") as f1:
                encodings_1 = json.load(f1)
            with open(os.path.join(tempdir, "export_2.encodings"), "rb") as f2:
                encodings_2 = json.load(f2)
            assert encodings_1 == encodings_2
            seen_lpbq_or_per_block = False
            for encoding in encodings_1["param_encodings"]:
                if is_lpbq:
                    if encoding["enc_type"] == "LPBQ":
                        seen_lpbq_or_per_block = True
                else:
                    if encoding["enc_type"] == "PER_BLOCK":
                        seen_lpbq_or_per_block = True
            assert seen_lpbq_or_per_block
            assert encodings_1["param_encodings"]
            assert np.allclose(out1, out2)

    @pytest.mark.parametrize("strict", [False])
    def test_mismatching_lpbq_settings(self, strict):
        torch.manual_seed(0)
        np.random.seed(0)
        model = single_residual_model()
        model_2 = copy.deepcopy(model)
        dummy_input = make_dummy_input(model.model)

        sim = QuantizationSimModel(model, param_type="int16", activation_type="int16")
        set_grouped_blockwise_quantization_for_weights(
            sim,
            op_types=("MatMul", "Conv", "Gemm"),
            bitwidth=4,
            decompressed_bw=8,
            block_size=4,
            strict=False,
        )

        sim.compute_encodings([dummy_input])
        out1 = sim.session.run(None, dummy_input)
        with tempfile.TemporaryDirectory() as tempdir, set_encoding_version("1.0.0"):
            sim.export(tempdir, "export")

            sim_2 = QuantizationSimModel(
                model_2, param_type="int16", activation_type="int16"
            )
            set_grouped_blockwise_quantization_for_weights(
                sim_2,
                op_types=("MatMul", "Conv", "Gemm"),
                bitwidth=2,
                decompressed_bw=4,
                block_size=2,
                strict=False,
            )

            sim_2.compute_encodings(
                lambda session, _: session.run(None, dummy_input), None
            )
            out2 = sim_2.session.run(None, dummy_input)
            assert not np.allclose(out1, out2)

            if strict:
                with pytest.raises(AssertionError):
                    load_encodings_to_sim(
                        sim_2, os.path.join(tempdir, "export.encodings"), strict=strict
                    )
            else:
                load_encodings_to_sim(
                    sim_2, os.path.join(tempdir, "export.encodings"), strict=strict
                )
                out2 = sim_2.session.run(None, dummy_input)
                sim_2.export(tempdir, "export_2")
                with open(os.path.join(tempdir, "export.encodings"), "rb") as f1:
                    encodings_1 = json.load(f1)
                with open(os.path.join(tempdir, "export_2.encodings"), "rb") as f2:
                    encodings_2 = json.load(f2)
                assert encodings_1 == encodings_2
                assert np.allclose(out1, out2)

    def test_model_with_constants(self):
        model = multi_input_with_constant_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            assert sim.qc_quantize_op_dict["/add0/Constant_output_0"].enabled == True
            assert sim.qc_quantize_op_dict["/add2/Constant_output_0"].enabled == True

    def test_multiple_output_quantsim(self):
        model = multi_output_model()
        sample_input = np.random.rand(128, 3, 32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model=model,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                param_type="int8",
                activation_type="int8",
                path=tempdir,
            )
            sim.session.run(None, {"input": sample_input})

    def test_quantsim_init_memory_usage(self):
        """
        When: Instantiate a quantsim model with high activation memory usage
        Then: Memory usage should not spike
        """
        num_layers = 2**9
        activation_dim = 2**13
        batch_size = 2**8
        total_act_memory = num_layers * activation_dim * batch_size

        # Create a model with very high total activation memory usage
        layers = [
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["shape"],
                name="shape",
                value=onnx.numpy_helper.from_array(
                    np.array([batch_size, activation_dim], dtype=np.dtype("int64"))
                ),
            ),
            onnx.helper.make_node(
                "Expand", inputs=["input", "shape"], outputs=["act0"], name="reshape"
            ),
        ]
        for idx in range(num_layers):
            layers.append(
                onnx.helper.make_node(
                    "Sigmoid",
                    inputs=[f"act{idx}"],
                    outputs=[f"act{idx + 1}"],
                    name=f"layer_{idx}",
                )
            )

        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 1]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            f"act{num_layers}", onnx.TensorProto.FLOAT, [batch_size, activation_dim]
        )
        graph = onnx.helper.make_graph(
            layers,
            "graph",
            initializer=[],
            inputs=[input_tensor],
            outputs=[output_tensor],
        )
        model = make_model(graph)

        with tempfile.TemporaryDirectory() as tempdir:
            tracemalloc.start()
            sim = QuantizationSimModel(model, path=tempdir)
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        assert peak_mem < current_mem + 0.25 * total_act_memory
        assert peak_mem < current_mem * 5

    def test_model_with_custom_ops(self):
        from onnxruntime_extensions import get_library_path

        def dummy_callback(session, args):
            calib_data = {"input": np.random.rand(1, 3, 64, 64).astype(np.float32)}
            _ = session.run(None, calib_data)

        model = custom_add_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model=model,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                param_type="int8",
                activation_type="int8",
                user_onnx_libs=[get_library_path()],
                path=tempdir,
            )
            sim.save_model_graph("./quantized_custom_model")
            with aimet_onnx.compute_encodings(sim):
                dummy_callback(sim.session, None)

            sim.export(tempdir, "custom_op_model")

    @pytest.mark.parametrize(
        "model",
        [
            models_for_tests.weight_matmul_model(10, 20),
            models_for_tests.weight_gemm_model(10, 20, False),
            models_for_tests.weight_gemm_model(10, 20, True),
        ],
    )
    def test_matmul_quantization_axis(self, model):
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
                "strict_symmetric": "False",
                "per_channel_quantization": "True",
            },
            "params": {"weight": {"is_quantized": "True"}},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }
        output_features = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        dummy_input = make_dummy_input(model)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "config.json")
            with open(config_file, "w") as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(
                model=model, config_file=config_file, path=temp_dir
            )

            sim.compute_encodings([make_dummy_input(model)])
            assert len(sim.qc_quantize_op_dict["weight"].encodings) == output_features

    def test_linear_split_into_matmul_add(self):
        model = linear_split_into_matmul_add()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, activation_type="int16", path=tempdir)

            sim.compute_encodings(make_dummy_input(model.model) for _ in range(3))
            sim.export(tempdir, "linear_matmul_add_pattern")
            with open(
                os.path.join(tempdir, "linear_matmul_add_pattern.encodings")
            ) as json_file:
                encoding_data = json.load(json_file)
                # Ensure that the encodings for the second input of Add op (bias) and output of MatMul aren't in JSON file.
                assert len(encoding_data["activation_encodings"]) == 2
                assert len(encoding_data["param_encodings"]) == 1
                activation_names = {
                    encoding["name"]
                    for encoding in encoding_data["activation_encodings"]
                }
                assert activation_names == {"input", "output"}

    @pytest.mark.skip(
        "OOM issues from high CPU memory usage, optimize quantsim memory usage before enabling"
    )
    def test_large_model(self):
        """
        When: Model is > 2GB
        Then: 1) We can still run the model
              2) We can still export the model
              3) Exported model contains all weights
        """
        # First create a model with is >= 2GB
        # Model size: (2 ** 5 layers) * (2 ** 15 * 2 ** 15 weights/layer) * (4 bytes/weight) = 2 ** 31 bytes
        num_layers = 2**5
        weight_shape = [2**12, 2**12]
        weights = []
        layers = []
        for idx in range(num_layers):
            layers.append(
                onnx.helper.make_node(
                    "MatMul",
                    inputs=[f"act{idx}", f"weight_{idx}"],
                    outputs=[f"act{idx + 1}_relu"],
                    name=f"matmul_{idx}",
                )
            )
            layers.append(
                onnx.helper.make_node(
                    "Relu",
                    inputs=[f"act{idx + 1}_relu"],
                    outputs=[f"act{idx + 1}"],
                    name=f"relu_{idx}",
                )
            )
            data = np.empty(weight_shape, dtype=np.float32)
            data[0][0] = idx  # Prevents simplifier from combining weights
            weights.append(onnx.numpy_helper.from_array(data, name=f"weight_{idx}"))

        input_tensor = onnx.helper.make_tensor_value_info(
            "act0", onnx.TensorProto.FLOAT, [1, weight_shape[0]]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            f"act{num_layers}", onnx.TensorProto.FLOAT, [1, weight_shape[1]]
        )
        graph = onnx.helper.make_graph(
            layers,
            "large_graph",
            initializer=weights,
            inputs=[input_tensor],
            outputs=[output_tensor],
        )
        model = make_model(graph)

        assert model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            sim.export(tempdir, "large_model")
            loaded_model = onnx.load(os.path.join(tempdir, "large_model.onnx"))
            # Check that all weights are contained in the loaded model
            assert len(loaded_model.graph.initializer) == len(model.graph.initializer)
            assert loaded_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF
            assert sim.model.model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF

        # Check that the model data is unchanged
        for idx in range(num_layers):
            assert (
                onnx.numpy_helper.to_array(sim.model.graph().initializer[idx])[0][0]
                == idx
            )

    def test_op_params_to_ignore(self):
        model = models_for_tests.resize_op_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            # params of specific ops shouldn't be quantized (here resize op param is testified)
            assert not sim.qc_quantize_op_dict.get("const_scale", None)

    def test_groupnorm_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {
                "GroupNormalization": {
                    "per_channel_quantization": "False",
                    "params": {"bias": {"is_quantized": "True"}},
                },
            },
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                param_type="int8",
                activation_type="int16",
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            def model_inputs():
                for _ in range(5):
                    yield make_dummy_input(model)

            sim.compute_encodings(model_inputs())
            sim.export(tempdir, "conv_matmul_groupnorm_model")

            with open(
                os.path.join(tempdir, "conv_matmul_groupnorm_model.encodings")
            ) as json_file:
                encoding_data = json.load(json_file)
                param_encodings = {
                    encoding["name"]: encoding
                    for encoding in encoding_data["param_encodings"]
                }
                groupnorm_weight_enc = param_encodings["groupnorm_0.scale"]
                groupnorm_bias_enc = param_encodings["groupnorm_0.bias"]

                # groupnorm param-encodings should follow output-activation-encoding config
                assert groupnorm_weight_enc["bw"] == 16
                assert groupnorm_weight_enc["is_sym"] is False

                assert groupnorm_bias_enc["bw"] == 16
                assert groupnorm_bias_enc["is_sym"] is False

    def test_matmul_v73_lower_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults": {
                "hw_version": "V66",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                param_type="int16",
                activation_type="int8",
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = make_dummy_input(model)
            sim.compute_encodings([dummy_tensor])
            sim.export(tempdir, "conv_matmul_groupnorm_model")

            with open(
                os.path.join(tempdir, "conv_matmul_groupnorm_model.encodings")
            ) as json_file:
                encoding_data = json.load(json_file)
                activation_encodings = {
                    encoding["name"]: encoding
                    for encoding in encoding_data["activation_encodings"]
                }
                matmul_second_input = activation_encodings["matmul_0.weight"]

                # matmul's second input encoding should be of 8 bitwidth and symmetric
                assert matmul_second_input["bw"] == 8
                assert matmul_second_input["is_sym"] is True

    def test_matmul_v73_lower_exception_rule_fp16(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults": {
                "hw_version": "V66",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                param_type="int4",
                activation_type="float16",
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            for name in sim.activation_names:
                quantizer = sim.qc_quantize_op_dict[name]
                assert quantizer.data_type == QuantizationDataType.float
                assert quantizer.bitwidth == 16

    def test_raise_error_with_bfloat16(self):
        model = models_for_tests.single_residual_model(dtype=torch.bfloat16)
        with pytest.raises(RuntimeError):
            sim = QuantizationSimModel(model)

        model = models_for_tests.model_with_cast(tensor_type=onnx.TensorProto.BFLOAT16)
        with pytest.raises(RuntimeError):
            sim = QuantizationSimModel(model)

        model = models_for_tests.model_with_constant(
            tensor_type=onnx.TensorProto.BFLOAT16
        )
        with pytest.raises(RuntimeError):
            sim = QuantizationSimModel(model)

    def test_matmul_v73_higher_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                param_type="int8",
                activation_type="int16",
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            dummy_tensor = make_dummy_input(model)
            sim.compute_encodings([dummy_tensor])
            sim.export(tempdir, "conv_matmul_groupnorm_model")

            with open(
                os.path.join(tempdir, "conv_matmul_groupnorm_model.encodings")
            ) as json_file:
                encoding_data = json.load(json_file)
                activation_encodings = {
                    encoding["name"]: encoding
                    for encoding in encoding_data["activation_encodings"]
                }
                matmul_second_input = activation_encodings["matmul_0.weight"]

                # if matmul's second input is 16bw then first input should also be 16bw
                assert matmul_second_input["is_sym"] is True

    def test_matmul_v73_exception_rule_matmul_branch(self, tmpdir):
        model = models_for_tests.add_matmul_model()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {"Gather": {"is_output_quantized": "False"}},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with open(os.path.join(tmpdir, "quantsim_config.json"), "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            param_type="int16",
            activation_type="int16",
            path=tmpdir,
            config_file=os.path.join(tmpdir, "quantsim_config.json"),
        )

        dummy_tensor = {
            "input": np.random.rand(3, 3).astype(np.float32),
            "input_2": np.random.rand(3, 3).astype(np.float32),
        }
        sim.compute_encodings([dummy_tensor])

        quantizer_1 = sim.qc_quantize_op_dict.get("added_output")
        assert quantizer_1.bitwidth == 16
        assert quantizer_1.use_symmetric_encodings
        assert len(quantizer_1.encodings) == 1

    @pytest.mark.parametrize(
        "model",
        (
            models_for_tests.pointwise_conv1d((1, 64, 32)),
            models_for_tests.conv_model(
                (64, 64, 3, 3), (1, 64, 32, 32), (1, 64, 32, 32), transpose=False
            ),
            models_for_tests.pointwise_conv3d((1, 64, 32, 32, 4)),
        ),
    )
    def test_blockwise_quantization_conv(self, model):
        block_size = 16
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(
            sim, "Conv", 4, True, block_size=block_size, strict=True
        )
        dummy_input = make_dummy_input(model)

        sim.compute_encodings([dummy_input])

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == block_size
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.blockAxis == 1
        assert len(weight_quantizer.encodings) == 64 * 64 / block_size

    @pytest.mark.parametrize(
        "model",
        (
            models_for_tests.pointwise_convtranspose1d((1, 64, 32)),
            models_for_tests.conv_model(
                (64, 64, 3, 3), (1, 64, 32, 32), (1, 64, 32, 32), transpose=True
            ),
            models_for_tests.pointwise_convtranspose3d((1, 64, 32, 32, 4)),
        ),
    )
    def test_blockwise_quantization_convtranspose(self, model):
        block_size = 16
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(
            sim, "ConvTranspose", 4, True, block_size=block_size, strict=True
        )
        dummy_input = make_dummy_input(model)

        sim.compute_encodings([dummy_input])

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == block_size
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.blockAxis == 0
        assert len(weight_quantizer.encodings) == 64 * 64 / block_size

    @pytest.mark.parametrize(
        "model",
        (
            models_for_tests.weight_gemm_model(
                in_features=16, out_features=32, transposed_weight=False
            ),
            models_for_tests.weight_gemm_model(
                in_features=16, out_features=32, transposed_weight=True
            ),
            models_for_tests.weight_matmul_model(in_features=16, out_features=32),
        ),
    )
    def test_blockwise_quantization_matmul(self, model):
        block_size = 4
        input_features = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        output_features = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        transposed_weight = model.graph.initializer[0].dims[0] == output_features
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(
            sim, ("MatMul", "Gemm"), 4, True, block_size=block_size, strict=True
        )
        dummy_input = make_dummy_input(model)

        sim.compute_encodings([dummy_input])

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert (
            len(weight_quantizer.encodings)
            == output_features * input_features / block_size
        )
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.channelAxis == (
            0 if transposed_weight else 1
        )
        assert weight_quantizer.quant_info.blockAxis == (1 if transposed_weight else 0)
        assert weight_quantizer.quant_info.blockSize == block_size
        sim.session.run(None, dummy_input)

    def test_blockwise_quantization_with_dynamic_matmul(self):
        block_size = 2
        model = models_for_tests.dynamic_matmul_model(batch_size=1)
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(
            sim, ("MatMul", "Gemm"), 4, True, block_size=block_size
        )

        assert sim.qc_quantize_op_dict["linear.weight"].quant_info.blockSize == 2

        for name, quantizer in sim.qc_quantize_op_dict.items():
            if name != "linear.weight":
                # Blockwise quantization should only be enabled for the linear layer
                assert quantizer.quant_info.blockSize == 0

    def test_blockwise_quantization_nonstrict(self):
        model = models_for_tests.weight_matmul_model(in_features=16, out_features=32)
        sim = QuantizationSimModel(model)
        with pytest.raises(ValueError):
            set_blockwise_quantization_for_weights(
                sim, ("MatMul", "Gemm"), 4, True, block_size=7, strict=True
            )

        set_blockwise_quantization_for_weights(
            sim, ("MatMul", "Gemm"), 4, True, block_size=7, strict=False
        )

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == 0
        sim.session.run(None, make_dummy_input(model))

    def test_blockwise_quantization_excluded_ops(self):
        model = models_for_tests.weight_matmul_model(in_features=16, out_features=32)
        ops = [node for node in model.graph.node]
        excluded_ops = [ops[0].name]  # Exclude matmul, set_blockwise should be a no op
        sim = QuantizationSimModel(model)

        set_blockwise_quantization_for_weights(
            sim,
            ("MatMul", "Gemm"),
            4,
            True,
            block_size=7,
            strict=False,
            excluded_nodes=excluded_ops,
        )

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == 0
        sim.session.run(None, make_dummy_input(model))

    @pytest.mark.parametrize(
        "model, block_size",
        (
            (models_for_tests.single_residual_model(), 4),
            (test_models.linear_layer_model(), 64),
        ),
    )
    def test_blockwise_quantization(self, model, block_size, tmpdir):
        dummy_input = make_dummy_input(model.model)
        bq_layers = ("MatMul", "Conv", "Gemm")
        bq_weights = set()

        for node in model.graph().node:
            if node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(
            sim, ("MatMul", "Conv", "Gemm"), 8, True, block_size, strict=False
        )
        sim.compute_encodings([dummy_input])

        initializers = {param.name: param for param in sim.model.graph().initializer}

        for name, quantizer in sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue

            param = initializers.get(name, None)

            if name in bq_weights:
                assert quantizer.quant_info.usePerChannelMode
                assert quantizer.quant_info.blockSize == block_size
                assert (
                    len(quantizer.encodings)
                    == param.dims[0] * param.dims[1] / block_size
                )
            elif quantizer.quant_info.usePerChannelMode:
                assert quantizer.quant_info.blockSize == 0
                assert len(quantizer.encodings) in tuple(param.dims)
            else:
                assert quantizer.quant_info.blockSize == 0
                assert len(quantizer.encodings) == 1

        sim.export(tmpdir, "tmp_model")
        with open(os.path.join(tmpdir, "tmp_model.encodings")) as f:
            encodings = json.load(f)

        for enc in encodings["param_encodings"]:
            quantizer = sim.qc_quantize_op_dict[enc["name"]]
            param = initializers[enc["name"]]
            if enc["name"] in bq_weights:
                assert len(enc["scale"]) == param.dims[0] * param.dims[1] / block_size
                assert enc["enc_type"] == "PER_BLOCK"
            elif quantizer.quant_info.usePerChannelMode:
                assert len(enc["scale"]) in tuple(param.dims)
                assert enc["enc_type"] == "PER_CHANNEL"
            else:
                assert len(enc["scale"]) == 1
                assert enc["enc_type"] == "PER_TENSOR"

        for enc in encodings["activation_encodings"]:
            assert len(enc["scale"]) == 1
            assert enc["enc_type"] == "PER_TENSOR"

    def test_model_with_initializers_as_activations(self):
        model = models_for_tests.model_with_initializers_as_activations()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = {
                "model_input": np.random.rand(1, 3, 8, 8).astype(np.float32)
            }
            with aimet_onnx.compute_encodings(sim):
                callback(sim.session, dummy_tensor)

            sim.export(tempdir, "model_with_initializers_as_activations")

            with open(
                os.path.join(
                    tempdir, "model_with_initializers_as_activations.encodings"
                )
            ) as json_file:
                encoding_data = json.load(json_file)

            assert all(
                x in [i.name for i in model.graph.initializer]
                for x in ["add_input2", "mul_input2"]
            )
            activation_encodings = {
                encoding["name"]: encoding
                for encoding in encoding_data["activation_encodings"]
            }
            assert activation_encodings["add_input2"]
            assert activation_encodings["mul_input2"]

    def test_load_float16_encodings(self, tmpdir):
        model = models_for_tests.weight_matmul_model(10, 10)
        sim = QuantizationSimModel(
            model, param_type="float16", activation_type="float16"
        )
        sim.export(tmpdir, "model")

        model = models_for_tests.weight_matmul_model(10, 10)
        sim = QuantizationSimModel(
            model, param_type="float16", activation_type="float16"
        )
        load_encodings_to_sim(sim, os.path.join(tmpdir, "model.encodings"), strict=True)

    def test_gather_exception_rule_for_float_data(self):
        model = models_for_tests.gather_op_model()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {"Gather": {"is_output_quantized": "False"}},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                param_type="int8",
                activation_type="int16",
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            dummy_input = {"model_input": np.asarray([[0, 1, 2, 3]], dtype=np.int64)}
            sim.compute_encodings([dummy_input])
            sim.export(tempdir, "gather_model")

            with open(os.path.join(tempdir, "gather_model.encodings")) as json_file:
                encoding_data = json.load(json_file)
                activation_encodings = {
                    encoding["name"]: encoding
                    for encoding in encoding_data["activation_encodings"]
                }
                gather_weight_enc = activation_encodings["gather_weight"]

                # gather param-encodings should follow output-activation-encoding config
                assert gather_weight_enc["bw"] == 16
                assert gather_weight_enc["is_sym"] is False

    def test_gather_with_int_data(self):
        model = models_for_tests.gather_op_with_int_data_model()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {"Gather": {"is_output_quantized": "False"}},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            dummy_input = {"model_input": np.asarray([[0, 1, 2, 3]], dtype=np.int64)}

            sim = QuantizationSimModel(
                model,
                param_type="int8",
                activation_type="int16",
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            sim.compute_encodings([dummy_input])
            sim.export(tempdir, "gather_model")

            with open(os.path.join(tempdir, "gather_model.encodings")) as json_file:
                encoding_data = json.load(json_file)
                activation_encoding_names = {
                    encoding["name"]
                    for encoding in encoding_data["activation_encodings"]
                }
                assert "gather_weight" not in activation_encoding_names

    @pytest.mark.parametrize(
        "model, block_size",
        (
            (models_for_tests.single_residual_model(), 4),
            (test_models.linear_layer_model(), 64),
        ),
    )
    def test_low_power_blockwise_quantization(self, model, block_size, tmpdir):
        dummy_input = make_dummy_input(model.model)
        bq_layers = ("MatMul", "Conv", "Gemm")
        bq_weights = set()
        bitwidth = 4
        decompressed_bw = 8

        for node in model.graph().node:
            if node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(model, param_type="int16", activation_type="int16")
        set_grouped_blockwise_quantization_for_weights(
            sim,
            ("MatMul", "Conv", "Gemm"),
            bitwidth,
            decompressed_bw,
            block_size,
            strict=False,
        )

        sim.compute_encodings([dummy_input])
        for name, quantizer in sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            if name in bq_weights:
                assert isinstance(quantizer, GroupedBlockQuantizeDequantize)
                assert quantizer.quant_info.usePerChannelMode
                assert quantizer.quant_info.blockSize == block_size
                assert len(quantizer.encodings) > 1
            else:
                assert quantizer.quant_info.blockSize == 0

        with set_encoding_version("1.0.0"):
            sim.export(tmpdir, "tmp_model")

        with open(os.path.join(tmpdir, "tmp_model.encodings")) as f:
            encodings = json.load(f)

        for enc in encodings["param_encodings"]:
            if enc["name"] not in bq_weights:
                assert enc["enc_type"] in (
                    EncodingType.PER_TENSOR.name,
                    EncodingType.PER_CHANNEL.name,
                )
            else:
                assert enc["enc_type"] == EncodingType.LPBQ.name
                assert enc["compressed_bw"] == bitwidth
                assert enc["bw"] == decompressed_bw

    def test_low_power_blockwise_quantization_with_excluded_ops(self, tmpdir):
        model = models_for_tests.single_residual_model()
        block_size = 4
        dummy_input = make_dummy_input(model.model)
        bq_layers = ["MatMul", "Conv", "Gemm"]
        bq_weights = set()
        bitwidth = 4
        decompressed_bw = 8
        ops = [node for node in model.graph().node]
        excluded_ops = [ops[3].name]  # Exclude conv3
        for node in model.graph().node:
            if node.name in excluded_ops:
                continue
            elif node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(
            model, dummy_input, default_param_bw=16, default_activation_bw=16
        )
        set_grouped_blockwise_quantization_for_weights(
            sim,
            bq_layers,
            bitwidth,
            decompressed_bw,
            block_size,
            strict=False,
            excluded_nodes=excluded_ops,
        )

        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)
        assert len(bq_weights) == 3
        for name, quantizer in sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            if name in bq_weights:
                assert isinstance(quantizer, GroupedBlockQuantizeDequantize)
                assert quantizer.quant_info.usePerChannelMode
                assert quantizer.quant_info.blockSize == block_size
                assert len(quantizer.encodings) > 1
            else:
                assert quantizer.quant_info.blockSize == 0

        with set_encoding_version("1.0.0"):
            sim.export(tmpdir, "tmp_model")

        with open(os.path.join(tmpdir, "tmp_model.encodings")) as f:
            encodings = json.load(f)

        for enc in encodings["param_encodings"]:
            if enc["name"] not in bq_weights:
                assert enc["enc_type"] in (
                    EncodingType.PER_TENSOR.name,
                    EncodingType.PER_CHANNEL.name,
                )
            else:
                assert enc["enc_type"] == EncodingType.LPBQ.name
                assert enc["compressed_bw"] == bitwidth
                assert enc["bw"] == decompressed_bw

    def test_lpbq_strict(self):
        model = models_for_tests.weight_matmul_model(in_features=16, out_features=32)
        sim = QuantizationSimModel(
            model, param_type="float16", activation_type="float16"
        )
        quantizers = set(sim.qc_quantize_op_dict.values())

        with pytest.raises(ValueError):
            set_grouped_blockwise_quantization_for_weights(
                sim, ("MatMul", "Gemm"), 4, 8, block_size=7, strict=True
            )
        """
        When: Call block size is incompatible with weight shape
        Then: Original quantizer/quant_info should be unchanged from the call
        """
        set_grouped_blockwise_quantization_for_weights(
            sim, ("MatMul", "Gemm"), 4, 8, block_size=7, strict=False
        )
        assert quantizers == set(sim.qc_quantize_op_dict.values())

        for quantizer in sim.qc_quantize_op_dict.values():
            assert quantizer.bitwidth == 16
            assert not quantizer.quant_info.usePerChannelMode
            assert quantizer.quant_info.blockSize == 0
            assert not quantizer.quant_info.isIntDataType

    @pytest.mark.parametrize("activation_type", [aimet_onnx.int8, aimet_onnx.int16])
    def test_encoding_constraints(self, activation_type):
        """
        When: Create quantsim with HTP quantsim config
        Then:
          - Softmax and Sigmoid output quantizers should be fixed to [0, 1]
          - Tanh output quantizers should be fixed to [-1, 1]
        """
        model = models_for_tests.softmax_model()
        sim = QuantizationSimModel(
            model, activation_type=activation_type, config_file="htp_v81"
        )
        sim.compute_encodings([make_dummy_input(model)])

        assert sim.qc_quantize_op_dict["model_output"].encodings[0].max == 1.0
        assert sim.qc_quantize_op_dict["model_output"].encodings[0].min == 0.0
        assert sim.qc_quantize_op_dict["softmax.output"].encodings[0].max == 1.0
        assert sim.qc_quantize_op_dict["softmax.output"].encodings[0].min == 0.0

        assert np.allclose(
            sim.qc_quantize_op_dict["tanh.output"].encodings[0].max,
            1.0,
            atol=sim.qc_quantize_op_dict["tanh.output"].encodings[0].delta,
        )
        assert np.allclose(
            sim.qc_quantize_op_dict["tanh.output"].encodings[0].min,
            -1.0,
            atol=sim.qc_quantize_op_dict["tanh.output"].encodings[0].delta,
        )
        assert sim.qc_quantize_op_dict["tanh.output"].use_symmetric_encodings
        assert sim.qc_quantize_op_dict["matmul.output"].encodings[0].max not in (
            1.0,
            2.0,
        )
        assert sim.qc_quantize_op_dict["matmul.output"].encodings[0].min != 0.0

        """
        When: Switch tanh output bitwidth from 8 to 16 or vice versa
        Then: Tanh output encoding constraints should hold
        """
        if sim.qc_quantize_op_dict["tanh.output"].bitwidth == 8:
            sim.qc_quantize_op_dict["tanh.output"].set_bitwidth(16)
        else:
            sim.qc_quantize_op_dict["tanh.output"].set_bitwidth(8)

        sim.qc_quantize_op_dict["tanh.output"].compute_encodings()

        assert np.allclose(
            sim.qc_quantize_op_dict["tanh.output"].encodings[0].max,
            1.0,
            atol=sim.qc_quantize_op_dict["tanh.output"].encodings[0].delta,
        )
        assert np.allclose(
            sim.qc_quantize_op_dict["tanh.output"].encodings[0].min,
            -1.0,
            atol=sim.qc_quantize_op_dict["tanh.output"].encodings[0].delta,
        )

    def test_matmul_3d_weight(self, tmp_path):
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }
        config_name = os.path.join(tmp_path, "quantsim_config.json")
        with open(config_name, "w") as f:
            json.dump(quantsim_config, f)
        model = models_for_tests.model_with_4d_matmul_weight()
        sim = QuantizationSimModel(model, config_file=config_name)
        sim.compute_encodings([make_dummy_input(model)])

        quantizer = sim.qc_quantize_op_dict["matmul_weight"]
        assert len(quantizer.get_encodings()) == model.graph.initializer[0].dims[-1]

        block_size = 8
        quantizer._enable_blockwise_quantization(block_size)
        sim.compute_encodings([make_dummy_input(model)])
        assert (
            len(quantizer.get_encodings())
            == model.graph.initializer[0].dims[-1]
            * model.graph.initializer[0].dims[-2]
            // block_size
        )

    @pytest.mark.cuda
    def test_quantsim_init_args(self):
        with pytest.raises(TypeError):
            QuantizationSimModel(single_residual_model(), rounding_mode="stochastic")

        sim = QuantizationSimModel(single_residual_model(), providers=CPU_PROVIDERS)
        assert sim.session.get_providers() == CPU_PROVIDERS

        available_providers = ort.get_available_providers()

        # Remove this check once onnxruntime-gpu becomes valid dependency for PyPI wheel.
        # This test will only fail in pypi-release pipeline (aimet-onnx-gpu whl has onnxruntime dependency)
        if "CUDAExecutionProvider" in available_providers:
            sim = QuantizationSimModel(
                single_residual_model(), providers=CUDA_PROVIDERS
            )
            assert sim.session.get_providers() == CUDA_PROVIDERS

            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider",
            ]
            sim = QuantizationSimModel(single_residual_model(), providers=providers)
            assert sim.session.get_providers() == CUDA_PROVIDERS
            assert (
                sim.session.get_provider_options()["CUDAExecutionProvider"][
                    "cudnn_conv_algo_search"
                ]
                == "DEFAULT"
            )

        dummy_input = make_dummy_input(single_residual_model().model)
        with pytest.warns(DeprecationWarning):
            QuantizationSimModel(single_residual_model(), dummy_input)

        with pytest.warns(DeprecationWarning):
            QuantizationSimModel(
                single_residual_model(), dummy_input, QuantScheme.min_max
            )

        with pytest.warns(DeprecationWarning):
            QuantizationSimModel(
                single_residual_model(), default_param_bw=8, default_activation_bw=16
            )

        # Remove this check once onnxruntime-gpu becomes valid dependency for PyPI wheel.
        # This test will only fail in pypi-release pipeline (aimet-onnx-gpu whl has onnxruntime dependency)
        if "CUDAExecutionProvider" in available_providers:
            with pytest.warns(DeprecationWarning):
                sim = QuantizationSimModel(single_residual_model(), use_cuda=True)
                assert "CUDAExecutionProvider" in sim.session.get_providers()

        with pytest.warns(DeprecationWarning):
            QuantizationSimModel(
                single_residual_model(),
                default_param_bw=16,
                default_activation_bw=16,
                default_data_type=QuantizationDataType.float,
            )

        with pytest.warns(DeprecationWarning):
            QuantizationSimModel(
                single_residual_model(), default_data_type=QuantizationDataType.int
            )

        with pytest.raises(RuntimeError):
            QuantizationSimModel(
                single_residual_model(),
                param_type="float16",
                activation_type="float16",
                default_data_type=QuantizationDataType.float,
            )

        with pytest.raises(RuntimeError):
            QuantizationSimModel(
                single_residual_model(), param_type="int4", default_param_bw=4
            )

        with pytest.raises(TypeError):
            QuantizationSimModel(single_residual_model(), unknown_arg=None)

        with pytest.raises(RuntimeError):
            QuantizationSimModel(single_residual_model(), param_type=qtype.float(6, 1))

    def test_quantsim_init_dtypes(self, tmp_path):
        sim = QuantizationSimModel(
            single_residual_model(), param_type="int4", activation_type="float16"
        )
        for name in sim.activation_names:
            assert sim.qc_quantize_op_dict[name].data_type == QuantizationDataType.float
            assert sim.qc_quantize_op_dict[name].bitwidth == 16

        for name in sim.param_names:
            assert sim.qc_quantize_op_dict[name].data_type == QuantizationDataType.int
            assert sim.qc_quantize_op_dict[name].bitwidth == 4

        sim.compute_encodings([make_dummy_input(sim.model.model)])
        sim.export(tmp_path, "model")

        sim = QuantizationSimModel(
            single_residual_model(),
            param_type=qtype.float(5, 10),
            activation_type=qtype.int(6),
        )
        for name in sim.activation_names:
            assert sim.qc_quantize_op_dict[name].data_type == QuantizationDataType.int
            assert sim.qc_quantize_op_dict[name].bitwidth == 6

        for name in sim.param_names:
            assert sim.qc_quantize_op_dict[name].data_type == QuantizationDataType.float
            assert sim.qc_quantize_op_dict[name].bitwidth == 16

    def test_compute_param_encodings(self):
        model = single_residual_model().model
        calibration_input = make_dummy_input(model)
        sim = QuantizationSimModel(copy.deepcopy(model))

        for quantizer in sim.qc_quantize_op_dict.values():
            assert not quantizer.is_initialized()

        """
        When: Call sim._compute_param_encodings()
        Then: 1) All param quantizers should be calibrated
              2) No activation quantizers should be calibrated
              3) Set of enabled quantizers should not change
        """
        enabled_quantizers = {
            name: quantizer
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.enabled
        }
        sim._compute_param_encodings()

        assert enabled_quantizers == {
            name: quantizer
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.enabled
        }

        for name, quantizer in enabled_quantizers.items():
            if name in sim.param_names:
                assert quantizer.is_initialized()

            else:
                assert not quantizer.is_initialized()

        """
        When: Call compute_encodings after sim._compute_param_encodings()
        Then: Encodings should be identical to just calling sim.compute_encodings
        """
        # Create an identical QuantizationSimModel
        sim_2 = QuantizationSimModel(copy.deepcopy(model))

        # Compute encodings for both sims using the same input
        sim.compute_encodings([calibration_input])
        sim_2.compute_encodings([calibration_input])

        # All encodings should be identical
        for name, quantizer in enabled_quantizers.items():
            quantizer_2 = sim_2.qc_quantize_op_dict[name]
            assert quantizer.export_encodings("2.0.0") == quantizer_2.export_encodings(
                "2.0.0"
            )

    def test_compute_param_encodings_overwrite(self):
        model = models_for_tests.weight_matmul_model()
        sim = QuantizationSimModel(model)
        sim.compute_encodings([make_dummy_input(sim.model.model)])

        weight_quantizer = sim.qc_quantize_op_dict["weight"]
        weight_encoding_unclipped = weight_quantizer.export_encodings("2.0.0")

        max_val = max(
            weight_quantizer.get_encodings()[0].max,
            -weight_quantizer.get_encodings()[0].min,
        )

        # Clip weight_quantizer encodings to a new value
        weight_quantizer.clip_and_recompute_encodings(0.5 * max_val)
        weight_encoding_clipped = weight_quantizer.export_encodings("2.0.0")

        """
        When: Compute param encodings with overwrite=False
        Then: Encoding for calibrated quantizer should not change
        """
        sim._compute_param_encodings(
            dummy_input=make_dummy_input(sim.model.model), overwrite=False
        )

        assert weight_quantizer.export_encodings("2.0.0") == weight_encoding_clipped

        """
        When: Compute param encodings with overwrite=True
        Then: Encoding for calibrated quantizer should be changed
        """
        sim._compute_param_encodings(
            dummy_input=make_dummy_input(sim.model.model), overwrite=True
        )
        assert weight_quantizer.export_encodings("2.0.0") == weight_encoding_unclipped

    def test_get_enabled_quantizer(self):
        model = models_for_tests.diverse_ops()
        sim = QuantizationSimModel(model)

        quantizer = sim._get_enabled_quantizer("output")
        assert quantizer == sim.qc_quantize_op_dict["relu_output"]
        path = sim._get_path_to_effective_quantizer("output_updated")
        assert path[0].op_type == "MaxPool"
        assert path[1].op_type == "Reshape"
        assert path[2].op_type == "QcQuantizeOp"

        sim.qc_quantize_op_dict["relu_output"].enabled = False
        assert sim._get_enabled_quantizer("output") is None
        assert sim._get_path_to_effective_quantizer("output_updated") is None

    @pytest.mark.parametrize("providers", [CPU_PROVIDERS, CUDA_PROVIDERS])
    def test_fp16_model_encodings(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        fp32_model = models_for_tests.diverse_ops(onnx.TensorProto.FLOAT)
        fp16_model = models_for_tests.diverse_ops(onnx.TensorProto.FLOAT16)

        fp32_dummy_input = make_dummy_input(fp32_model)
        fp16_dummy_input = {
            key: value.astype(np.float16) for key, value in fp32_dummy_input.items()
        }

        fp32_sim = QuantizationSimModel(fp32_model, providers=[providers])
        fp32_sim.compute_encodings([fp32_dummy_input])

        fp16_sim = QuantizationSimModel(fp16_model, providers=[providers])
        fp16_sim.compute_encodings([fp16_dummy_input])

        assert (
            fp32_sim.qc_quantize_op_dict.keys() == fp16_sim.qc_quantize_op_dict.keys()
        )

        fp32_quantizers = list(fp32_sim.qc_quantize_op_dict.values())
        fp16_quantizers = list(fp16_sim.qc_quantize_op_dict.values())

        for i in range(len(fp32_quantizers)):
            assert fp32_quantizers[i].enabled == fp16_quantizers[i].enabled
            if fp32_quantizers[i].enabled and fp16_quantizers[i].enabled:
                fp32_encodings = fp32_quantizers[i].encodings[0]
                fp16_encodings = fp16_quantizers[i].encodings[0]

                fp32_values = [
                    fp32_encodings.min,
                    fp32_encodings.max,
                    fp32_encodings.delta,
                ]
                fp16_values = [
                    fp16_encodings.min,
                    fp16_encodings.max,
                    fp16_encodings.delta,
                ]

                assert np.allclose(fp32_values, fp16_values, atol=0.01)
                assert abs(fp32_encodings.offset - fp16_encodings.offset) <= 1

    @pytest.mark.parametrize("providers", [CPU_PROVIDERS, CUDA_PROVIDERS])
    def test_fp16_model_with_weights_encodings(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        fp32_model = models_for_tests.single_residual_model(dtype=torch.float32).model
        fp16_model = models_for_tests.single_residual_model(dtype=torch.float16).model

        fp32_weights_dict = {}
        for weight in fp16_model.graph.initializer:
            np_weight = onnx.numpy_helper.to_array(weight)
            if np_weight.dtype != np.float16:
                continue

            np_weight_fp32 = np_weight.astype(np.float32)
            weight_fp32 = onnx.numpy_helper.from_array(np_weight_fp32, weight.name)
            fp32_weights_dict[weight.name] = weight_fp32

        for i, w in enumerate(fp32_model.graph.initializer):
            if w.name in fp32_weights_dict:
                fp32_model.graph.initializer[i].CopyFrom(fp32_weights_dict[w.name])

        fp32_dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        fp16_dummy_input = {
            key: value.astype(np.float16) for key, value in fp32_dummy_input.items()
        }

        fp32_sim = QuantizationSimModel(fp32_model, providers=providers)
        fp32_sim.compute_encodings([fp32_dummy_input])

        fp16_sim = QuantizationSimModel(fp16_model, providers=providers)
        fp16_sim.compute_encodings([fp16_dummy_input])

        assert (
            fp32_sim.qc_quantize_op_dict.keys() == fp16_sim.qc_quantize_op_dict.keys()
        )

        fp32_quantizers = list(fp32_sim.qc_quantize_op_dict.values())
        fp16_quantizers = list(fp16_sim.qc_quantize_op_dict.values())

        for i in range(len(fp32_quantizers)):
            assert fp32_quantizers[i].enabled == fp16_quantizers[i].enabled
            if fp32_quantizers[i].enabled and fp16_quantizers[i].enabled:
                fp32_encodings = fp32_quantizers[i].encodings[0]
                fp16_encodings = fp16_quantizers[i].encodings[0]

                fp32_values = [
                    fp32_encodings.min,
                    fp32_encodings.max,
                    fp32_encodings.delta,
                ]
                fp16_values = [
                    fp16_encodings.min,
                    fp16_encodings.max,
                    fp16_encodings.delta,
                ]

                assert np.allclose(fp32_values, fp16_values, atol=0.01)
                assert abs(fp32_encodings.offset - fp16_encodings.offset) <= 1


class TestEncodingPropagation:
    def test_output(self):
        """
        Given: model as below

                   +-> q_in1 -> conv1 -> relu1 ---> q_out1 -------v
          [input] -+                                           concat -> q_out3 -> [output]
                   +-> q_in2 -> conv2 -> relu2 ---> q_out2 -------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = x2 = x
                x1 = self.conv1(x1)
                x1 = self.relu1(x1)
                x2 = self.conv2(x2)
                x2 = self.relu2(x2)
                return torch.cat([x1, x2])

        """
       When: _apply_constraints(True)

       Then: q_out1 and q_out2 are replaced with q_out3 as below

                  +-> q_in1 -> conv1 -> relu1 -> **q_out3** -----v
         [input] -+                                           concat -> q_out3- > [output]
                  +-> q_in2 -> conv2 -> relu2 -> **q_out3** -----^
        """
        pt_model = Model().eval()
        x = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, x)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)

            sim.compute_encodings([dummy_input])
            assert _compare_encodings(
                sim.qc_quantize_op_dict["/relu1/Relu_output_0"].encodings[0],
                sim.qc_quantize_op_dict["output"].encodings[0],
            )
            assert _compare_encodings(
                sim.qc_quantize_op_dict["/relu2/Relu_output_0"].encodings[0],
                sim.qc_quantize_op_dict["output"].encodings[0],
            )

    def test_math_invariant(self):
        """
        Given: model as below

                   +-> q_in1 -> conv1 ---> relu1 -> q_out1 ------v
          [input] -+                                          concat -> q_out2 -> [output]
                   +-> q_in2 -> reshape -> permute --------------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x1 = x2 = x
                x1 = self.conv1(x1)
                x1 = self.relu1(x1)
                x2 = torch.reshape(x2, (-1, 24, 24, 3))
                x2 = torch.permute(x2, (0, 3, 1, 2))
                return torch.cat([x1, x2])

        """
        When: _apply_constraints(True)

        Then: q_out1 and q_in2 are replaced with q_out3 as below

                   +-> q_in1 -> conv1 ---> relu1 -----> **q_out2**- --------v
          [input] -+                                                     concat -> q_out2 -> [output]
                   +-> **q_out2** -> reshape -> transpose -> permute -------^
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)
            sim.compute_encodings([dummy_input])

            assert _compare_encodings(
                sim.qc_quantize_op_dict["/relu1/Relu_output_0"].encodings[0],
                sim.qc_quantize_op_dict["output"].encodings[0],
            )
            assert _compare_encodings(
                sim.qc_quantize_op_dict["input"].encodings[0],
                sim.qc_quantize_op_dict["output"].encodings[0],
            )

    def test_concat_tree(self):
        """
        Given: model as below

                    +-> q_in1a -> conv1a -> q_out1a -> concat1 -> q_out1c -> reshape --+
                    +-> q_in1b -> conv1b -> q_out1b ------^                            v
          [input] --+                                                               concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> q_out2a -> concat2 -> q_out2c -------------^
                    +-> q_in2b -> conv2b -> q_out2b ------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1a = torch.nn.Conv2d(3, 3, 3)
                self.conv1b = torch.nn.Conv2d(3, 3, 3)
                self.conv2a = torch.nn.Conv2d(3, 3, 3)
                self.conv2b = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x1a = x1b = x2a = x2b = x
                x1a = self.conv1a(x1a)
                x1b = self.conv1b(x1b)
                x1 = torch.cat([x1a, x1b])
                x1 = torch.reshape(x1, (-1, 22, 22, 3))
                x1 = torch.permute(x1, (0, 3, 1, 2))
                x2a = self.conv2a(x2a)
                x2b = self.conv2b(x2b)
                x2 = torch.cat([x2a, x2b])
                return torch.cat([x1, x2])

        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        """
        When: _apply_constraints(True)

        Then: All q_out{*} are replaced with q_out3 as below

                    +-> q_in1a -> conv1a -> *q_out3* -> concat1 -> *q_out3* -> reshape --+
                    +-> q_in1b -> conv1b -> *q_out3* ------^                             v
          [input] --+                                                                 concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> *q_out3* -> concat2 -> *q_out3* -------------^
                    +-> q_in2b -> conv2b -> *q_out3* ------^
        """
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)
            sim.compute_encodings([dummy_input])

            for cg_op in sim.connected_graph.ordered_ops:
                if cg_op.type in ["Conv", "Concat"]:
                    _, out_qtzr, __ = sim.get_op_quantizers(cg_op)
                    assert _compare_encodings(
                        out_qtzr[0].encodings[0],
                        sim.qc_quantize_op_dict["output"].encodings[0],
                    )

    @pytest.mark.parametrize("bitwidth", [8, 16])
    def test_encoding_constraints(self, bitwidth: int):
        """
        Given: model as below

        [input] -> Sigmoid -> MaxPool -> Softmax -> Resize -> Reshape -+
                                                                       V
                                                              ... -> MatMul -> [output]
        """

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                x = torch.nn.functional.sigmoid(x)
                x = torch.nn.functional.max_pool2d(x, (3, 3))
                x *= 100
                x = torch.nn.functional.softmax(x)
                x = torch.nn.functional.interpolate(x, size=(50, 50), mode="bilinear")
                return torch.ones(50, 50) @ x.reshape(50, 150)

        """
        When: _apply_constraints(True)
        Then:
          1. Sigmoid output encoding should be fixed to [0, 1]
          2. Sigmoid output quantizer and MaxPool output quantizer should be identical
          3. Softmax output encoding should be symmetric and fixed to [-1, 1]
          4. Softmax output quantizer and Resize output quantizer should be identical
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(
                model,
                activation_type=aimet_onnx.int8 if bitwidth == 8 else aimet_onnx.int16,
                config_file="htp_v81",
            )
            sim.compute_encodings([dummy_input])

        assert (
            sim.qc_quantize_op_dict["/Sigmoid_output_0"]
            is sim.qc_quantize_op_dict["/MaxPool_output_0"]
        )
        (output_encoding,) = sim.qc_quantize_op_dict[
            "/Sigmoid_output_0"
        ].get_encodings()
        expected_scale = 1 / (2**bitwidth - 1)
        assert output_encoding.min == 0
        assert output_encoding.max == 1
        assert output_encoding.offset == 0
        assert np.allclose(
            output_encoding.delta, expected_scale, atol=np.finfo(np.float32).eps
        )

        assert (
            sim.qc_quantize_op_dict["/Softmax_output_0"]
            is sim.qc_quantize_op_dict["/Resize_output_0"]
        )
        (output_encoding,) = sim.qc_quantize_op_dict[
            "/Softmax_output_0"
        ].get_encodings()
        expected_scale = 1 / (2 ** (bitwidth - 1) - 1)
        assert np.allclose(output_encoding.min, -1, atol=expected_scale)
        assert output_encoding.max == 1
        assert output_encoding.offset == -(2 ** (bitwidth - 1))
        assert np.allclose(
            output_encoding.delta, expected_scale, atol=np.finfo(np.float32).eps
        )

    def test_encoding_constraints2(self):
        """
        Given: model as below

        [input] -> Conv -> Reshape -> Resize -> [output]
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x: torch.Tensor):
                x = self.conv(x)
                x = x.reshape(9, 1, 222, 222)
                return torch.nn.functional.interpolate(
                    x, size=(50, 50), mode="bilinear"
                )

        """
        When: _apply_constraints(True)
        Then: Resize output encoding should inherit Conv output encoding (not Reshape)
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(3, 3, 224, 224)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, config_file="htp_v81")
            sim.compute_encodings([dummy_input])

        assert (
            sim.qc_quantize_op_dict["/conv/Conv_output_0"]
            is sim.qc_quantize_op_dict["output"]
        )

    def test_encoding_constraints3(self):
        """
        Given: model as below

        [input] -+-> Sigmoid -> Concat -> [output]
                 +-> Softmax ----^
        """

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.cat(
                    (
                        torch.sigmoid(x),
                        torch.nn.functional.softmax(x),
                    )
                )

        """
        When: _apply_constraints(True)
        Then: Concat output quantizers should be fixed to range [0, 1]
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, config_file="htp_v81")
            sim.compute_encodings([dummy_input])

        assert (
            sim.qc_quantize_op_dict["/Sigmoid_output_0"]
            is sim.qc_quantize_op_dict["/Softmax_output_0"]
        )
        assert (
            sim.qc_quantize_op_dict["/Softmax_output_0"]
            is sim.qc_quantize_op_dict["output"]
        )
        (output_encoding,) = sim.qc_quantize_op_dict["output"].get_encodings()
        expected_scale = 1 / 255
        assert output_encoding.min == 0
        assert output_encoding.max == 1
        assert output_encoding.offset == 0
        assert np.allclose(
            output_encoding.delta, expected_scale, atol=np.finfo(np.float32).eps
        )

        """
        Given: model as below

        [input] -+-> Sigmoid -> Concat -> [output]
                 +-> MatMul -----^
        """

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.cat(
                    (
                        torch.sigmoid(x),
                        x @ x,
                    )
                )

        """
        When: _apply_constraints(True)
        Then: Only MatMul output quantizer must be tied with Concat output quantizer
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(100, 100)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, config_file="htp_v81")
            sim.compute_encodings([dummy_input])

        assert (
            sim.qc_quantize_op_dict["/Sigmoid_output_0"]
            is not sim.qc_quantize_op_dict["output"]
        )
        assert (
            sim.qc_quantize_op_dict["/MatMul_output_0"]
            is sim.qc_quantize_op_dict["output"]
        )

        (sigmoid_output_encoding,) = sim.qc_quantize_op_dict[
            "/Sigmoid_output_0"
        ].get_encodings()
        expected_scale = 1 / 255
        assert sigmoid_output_encoding.min == 0
        assert sigmoid_output_encoding.max == 1
        assert sigmoid_output_encoding.offset == 0
        assert np.allclose(
            sigmoid_output_encoding.delta, expected_scale, atol=np.finfo(np.float32).eps
        )

    def test_partial_encoding_constraints(self):
        """
        Given: model as below

        [input] --> Conv -> Relu -> [output]
        """
        pt_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.ReLU(),
        ).eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        model = _convert_to_onnx(pt_model, dummy_input)

        """
        When: Create quantsim with HTP config file
        Then: Relu output quantizers should be fixed to range [0, ?]
        """
        dummy_input = make_dummy_input(model.model)
        sim = QuantizationSimModel(model, config_file="htp_v81")

        output_qtzr = sim.qc_quantize_op_dict["output"]
        assert output_qtzr._encoding_min_max_fixed_vals == (0.0, None)

        sim.compute_encodings([dummy_input])

        (output_encoding,) = output_qtzr.get_encodings()
        assert output_encoding.min == 0

        (output,) = sim.session.run(None, dummy_input)
        assert np.all(output >= 0)

    @pytest.mark.parametrize(
        "op_type_under_test",
        [torch.nn.AvgPool2d, torch.nn.Upsample],
    )
    def test_output_parametrized(self, op_type_under_test):
        """
        Given: model as below
           [input] -+-> q_in1 -> conv1 -> q_out1 -> op_type_under_test -> q_out2 -> [output]
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.op_type_under_test = op_type_under_test(3)

            def forward(self, x):
                x1 = self.conv1(x)
                return self.op_type_under_test(x1)

        """
       When: _apply_constraints(True)

       Then: q_out1 will be replaced with q_out2 as below

             [input] -+-> q_in1 -> conv1 -> *q_out2* -> op_type_under_test -> q_out2 -> [output]

        """
        pt_model = Model().eval()
        x = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, x)
        # simplifier required to transform torch.nn.Upsample into a single onnx Resize op
        model.model, _ = simplify(model.model)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)
            sim.compute_encodings([dummy_input])

            for cg_op in sim.connected_graph.ordered_ops:
                if cg_op.type in ["Conv"]:
                    _, out_qtzr, __ = sim.get_op_quantizers(cg_op)
                    if out_qtzr:
                        assert _compare_encodings(
                            out_qtzr[0].encodings[0],
                            sim.qc_quantize_op_dict["output"].encodings[0],
                        )

    def test_integer_concat(self):
        """
        When: Model contains unquantizable layers with op_type in quantsim.op_types_to_tie_qtzrs
        Then: Error should not be thrown during quantsim init
        """
        model = models_for_tests.integer_concat_model()
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)

        with pytest.raises(ValueError):
            sim.set_quantizers({"out_shape": sim.qc_quantize_op_dict["model_input"]})

    def test_gather_concat(self):
        model = models_for_tests.gather_concat_model()
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)

        sim.compute_encodings([make_dummy_input(model)])
        concat_out_scale = sim.qc_quantize_op_dict["out"].get_encodings()[0].delta

        # Encoding should propagate through the 'x' input of Gather
        assert (
            sim.qc_quantize_op_dict["x_2"].get_encodings()[0].delta == concat_out_scale
        )
        # Encoding should not propagate through the 'indices' input of Gather
        assert (
            not sim.qc_quantize_op_dict["z"].get_encodings()[0].delta
            == concat_out_scale
        )
        # Encoding should not propagate through Mul
        assert (
            not sim.qc_quantize_op_dict["x"].get_encodings()[0].delta
            == concat_out_scale
        )

    def test_set_quantizers(self):
        model = models_for_tests.gather_concat_model()
        sim = QuantizationSimModel(model)

        assert sim.qc_quantize_op_dict["x"] is not sim.qc_quantize_op_dict["out"]
        assert sim.qc_quantize_op_dict["y"] is not sim.qc_quantize_op_dict["out"]

        """
        When: Tie quantizers for two tensors together
        Then: sim.qc_quantize_op_dict points to the same object for both tensors
        """
        quantizer = sim.qc_quantize_op_dict["out"]
        sim.set_quantizers({"x": quantizer, "y": quantizer})

        assert sim.qc_quantize_op_dict["x"] is sim.qc_quantize_op_dict["out"]
        assert sim.qc_quantize_op_dict["y"] is sim.qc_quantize_op_dict["out"]

        """
        When: An tensor name passed to sim.set_quantizers does not exist in sim.qc_quantize_op_dict
        Then: raise ValueError
        """
        with pytest.raises(ValueError):
            sim.set_quantizers({"z_int": quantizer})
        with pytest.raises(ValueError):
            sim.set_quantizers({"x_updated": quantizer})

        """
        When: quantizer is not of type QcQuantizeOp
        Then: raise TypeError
        """
        with pytest.raises(TypeError):
            sim.set_quantizers({"out": "x"})

        quantizer.set_bitwidth(4)
        sim.compute_encodings([make_dummy_input(model)])

        out_delta = sim.qc_quantize_op_dict["out"].get_encodings()[0].delta
        assert sim.qc_quantize_op_dict["x"].get_encodings()[0].delta == out_delta
        assert sim.qc_quantize_op_dict["y"].get_encodings()[0].delta == out_delta

    def test_clamp_activation_encodings(self):
        model = models_for_tests.matmul_add_model()
        dummy_input = {
            "model_input": np.expand_dims(np.identity(8, np.float32), axis=(0, 1))
        }
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
            )

            sim.compute_encodings([dummy_input])
            clamp_activation_encodings(sim, 100.0)

            sim.export(tempdir, "matmul_add_quantsim")

            with open(
                os.path.join(tempdir, "matmul_add_quantsim.encodings")
            ) as json_file:
                encodings = json.load(json_file)

            activation_encodings = {
                encoding["name"]: encoding
                for encoding in encodings["activation_encodings"]
            }
            add_act_encoding = activation_encodings["add_1.output"]
            matmul_act_encoding = activation_encodings["matmul_2.output"]

            assert (
                round(
                    add_act_encoding["scale"][0] * (255 + add_act_encoding["offset"][0])
                )
                == 100.0
            )
            assert (
                round(
                    matmul_act_encoding["scale"][0]
                    * (255 + matmul_act_encoding["offset"][0])
                )
                == 100.0
            )

    def test_matmul_with_constant_first_input(self):
        model = models_for_tests.matmul_with_constant_first_input()
        quantsim_config = {
            "defaults": {
                "hw_version": "V73",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {"Unsqueeze": {"is_output_quantized": "False"}},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                activation_type="int16",
            )
            assert sim.qc_quantize_op_dict["model_input"].enabled
            assert sim.qc_quantize_op_dict["model_input"].use_symmetric_encodings
            assert sim.qc_quantize_op_dict["matmul.weight"].enabled

    def test_matmul_with_constant_second_input(self):
        model = models_for_tests.weight_matmul_model()
        quantsim_config = {
            "defaults": {
                "hw_version": "V69",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "False"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                param_type="int4",
                activation_type="int16",
            )
            """
            Exception rule should not be applied to non-dynamic matmuls
            """
            assert sim.qc_quantize_op_dict["weight"].bitwidth == 4

    @pytest.mark.parametrize("per_channel", [True, False])
    def test_matmul_add_bias_quantizer(self, per_channel: bool):
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": str(per_channel),
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        """
        Given: Model that contains matmul-add sequence that can be interpreted as
               weight_matmul - bias_add
        """
        model = models_for_tests.matmul_bias_add_model()

        """
        When: Create QuantizationSimModel
        Then:
          1) Bias quantizer should be disabled
          2) Bias quantizer should follow the same granularity as weight quantizer
          3) get_op_quantizer should return bias quantizer of Add
        """
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, "quantsim_config.json")
            with open(config_file, "w") as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, config_file=config_file)

        input_qtzr = sim.qc_quantize_op_dict[f"input"]
        weight_qtzr = sim.qc_quantize_op_dict[f"matmul.weight"]
        bias_qtzr = sim.qc_quantize_op_dict[f"add.bias"]
        assert not bias_qtzr.enabled
        assert (
            bias_qtzr.quant_info.usePerChannelMode
            == weight_qtzr.quant_info.usePerChannelMode
        )

        _, _, param_quantizers = sim.get_op_quantizers(sim.connected_graph._ops["add"])
        assert list(param_quantizers.values()) == [bias_qtzr]

        """
        When: Concretize int32 bias quantizers
        Then: Bias scale should be derived as input_scale * weight_scale of matmul
        """
        with aimet_onnx.compute_encodings(sim):
            _ = sim.session.run(
                None, {"input": np.random.randn(10, 10).astype(np.float32)}
            )

        sim._concretize_int32_bias_quantizers()
        assert bias_qtzr.enabled
        bias_scale = (np.array(bias_qtzr.export_encodings("2.0.0")["y_scale"]),)
        expected = np.array(
            weight_qtzr.export_encodings("2.0.0")["y_scale"]
        ) * np.array(input_qtzr.export_encodings("2.0.0")["y_scale"])
        assert np.allclose(bias_scale, expected)

    @pytest.mark.parametrize(
        "per_channel, weight_encoding",
        [
            (
                True,
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": True,
                    "offset": [0.0],
                    "scale": [0.023529411764705882],
                },
            ),
            (
                False,
                {
                    "bw": 8,
                    "dtype": "INT",
                    "enc_type": "PER_CHANNEL",
                    "is_sym": True,
                    "offset": [0.0] * 8,
                    "scale": [0.023529411764705882] * 8,
                },
            ),
        ],
    )
    def test_concretize_bias_qtzr_with_mismatched_setting(
        self, per_channel, weight_encoding
    ):
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": str(per_channel),
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        """
        Given: Model that contains matmul-add sequence that can be interpreted as
               weight_matmul - bias_add
        """
        model = models_for_tests.conv_relu()
        encodings = {
            "param_encodings": [{**weight_encoding, "name": "conv_weight"}],
            "activation_encodings": [],
            "version": "1.0.0",
        }

        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, "quantsim_config.json")
            with open(config_file, "w") as f:
                json.dump(quantsim_config, f)
                f.close()

            enc_overrides = os.path.join(tempdir, "enc_overrides.encodings")
            with open(enc_overrides, "w") as f:
                json.dump(encodings, f)
                f.close()

            sim = QuantizationSimModel(model, config_file=config_file)

            """
            When Quantsim is instantiated with PTQ, and the PCQ weight encodings are loaded into Qsim,
            the weight quantizer setting changes to PCQ and vice-versa
            """

            if per_channel:
                assert sim.qc_quantize_op_dict[
                    "conv_weight"
                ].quant_info.usePerChannelMode
            else:
                assert not sim.qc_quantize_op_dict[
                    "conv_weight"
                ].quant_info.usePerChannelMode

            load_encodings_to_sim(
                sim, enc_overrides, strict=False, allow_overwrite=False
            )

            if per_channel:
                assert not sim.qc_quantize_op_dict[
                    "conv_weight"
                ].quant_info.usePerChannelMode
            else:
                assert sim.qc_quantize_op_dict[
                    "conv_weight"
                ].quant_info.usePerChannelMode

            sim.compute_encodings([make_dummy_input(model)])

            """
            When Quantsim is instantiated with PTQ, and the PCQ weight encodings are loaded into Qsim,
            the weight quantizer setting changes to PCQ. Now, when concretizing bias quantizers,
            the bias quantizer setting should also change to PCQ (based on weight quantizer) and vice-versa.
            """
            if per_channel:
                assert sim.qc_quantize_op_dict["conv_bias"].quant_info.usePerChannelMode
            else:
                assert not sim.qc_quantize_op_dict[
                    "conv_bias"
                ].quant_info.usePerChannelMode

            sim._concretize_int32_bias_quantizers()

            if per_channel:
                assert not sim.qc_quantize_op_dict[
                    "conv_bias"
                ].quant_info.usePerChannelMode
            else:
                assert sim.qc_quantize_op_dict["conv_bias"].quant_info.usePerChannelMode

    def test_identity_conv_perchannel(self):
        model = models_for_tests.conv_with_weight_identity_input()

        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(
                model, path=tempdir, config_file=get_path_for_per_channel_config()
            )
            assert sim.qc_quantize_op_dict[
                "identity.input"
            ].quant_info.usePerChannelMode
            assert sim.qc_quantize_op_dict["identity.input"].quant_info.channelAxis == 0

    def test_customop_model(self):
        from onnxruntime_extensions import get_library_path

        model = models_for_tests.custom_op_model()
        sim = QuantizationSimModel(model, user_onnx_libs=[get_library_path()])
        assert {
            "model_input",
            "output",
            "model_output",
            "y",
            "z",
        } == sim.qc_quantize_op_dict.keys()

    def test_set_and_freeze_param_encodings(self):
        torch.manual_seed(0)
        np.random.seed(0)
        model = single_residual_model().model
        model_2 = copy.deepcopy(model)
        dummy_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            sim.compute_encodings([dummy_tensor])
            pre_load_out = sim.session.run(None, dummy_tensor)
            new_encoding = libpymo.TfEncoding()
            new_encoding.min = -16.0
            new_encoding.max = 15.875
            new_encoding.bw = 8
            new_encoding.delta = 0.125
            new_encoding.offset = -128
            sim.qc_quantize_op_dict["conv3.weight"].load_encodings([new_encoding] * 8)
            post_load_out = sim.session.run(None, dummy_tensor)

            sim.export(tempdir, "onnx_sim")

            del sim

            sim = QuantizationSimModel(model_2, path=tempdir)
            sim.compute_encodings([dummy_tensor])
            pre_load_out_2 = sim.session.run(None, dummy_tensor)

            with open(os.path.join(tempdir, "onnx_sim.encodings"), "r") as f:
                encodings = json.load(f)

            with open(os.path.join(tempdir, "param_encodings.json"), "w") as f:
                json.dump(encodings["param_encodings"], f, sort_keys=True, indent=4)

            sim.set_and_freeze_param_encodings(
                os.path.join(tempdir, "param_encodings.json")
            )
            post_load_out_2 = sim.session.run(None, dummy_tensor)

            assert np.allclose(pre_load_out, pre_load_out_2)
            assert np.allclose(post_load_out, post_load_out_2)

    @pytest.mark.parametrize(
        "model_factory,             input_shape, block_size, lpbq",
        [
            (single_residual_model, (1, 3, 32, 32), None, False),
            (single_residual_model, (1, 3, 32, 32), 4, False),
            (single_residual_model, (1, 3, 32, 32), 4, True),
            (transposed_conv_model, (10, 10, 4, 4), None, False),
            (transposed_conv_model, (10, 10, 4, 4), 5, False),
            (transposed_conv_model, (10, 10, 4, 4), 5, True),
            (instance_norm_model, (2, 10, 24, 24), None, False),
            (layernorm_model, (1, 4, 64, 64), None, False),
        ],
    )
    def test_detect_bias_overflow(
        self, model_factory, input_shape, block_size, lpbq, tmp_path
    ):
        torch.manual_seed(0)
        np.random.seed(0)
        model = model_factory()
        input = np.random.randn(*input_shape).astype(np.float32)

        def _update_bias(initializer):
            bias_tensor = onnx.numpy_helper.to_array(
                initializer
            ).copy()  # Make it writable
            bias_tensor[:] = (
                100  # Ensures that we exceed int32 range for all bias values.
            )
            updated_tensor = onnx.numpy_helper.from_array(
                bias_tensor, name=initializer.name
            )
            initializer.CopyFrom(updated_tensor)

        dummy_tensor = {"input": input}

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            param_type=aimet_onnx.int16,
            activation_type=aimet_onnx.int16,
        )
        if block_size:
            op_types = ("Conv", "ConvTranspose", "Gemm")
            if lpbq:
                set_grouped_blockwise_quantization_for_weights(
                    sim,
                    op_types,
                    bitwidth=4,
                    decompressed_bw=16,
                    block_size=block_size,
                    strict=False,
                )
            else:
                set_blockwise_quantization_for_weights(
                    sim,
                    op_types,
                    bitwidth=16,
                    symmetric=True,
                    block_size=block_size,
                    strict=False,
                )
        """
        When: Update bias values to very large value
        """
        linear_ops_with_bias = {
            op: sim._get_weight_and_bias(op)
            for op in sim.connected_graph.get_all_ops().values()
            if op.type in ("Conv", "ConvTranspose", "Gemm")
        }
        for _, (__, bias) in linear_ops_with_bias.items():
            if bias is None:
                continue
            for ini in sim.model.model.graph.initializer:
                if ini.name == bias.name:
                    print(f"Updated bias: {bias.name}")
                    _update_bias(ini)

        sim.compute_encodings([dummy_tensor])
        sim.export(tmp_path, "before_weight_adj")
        with open(tmp_path / "before_weight_adj.encodings") as f:
            encodings = json.load(f)
            before_weight_adj = {
                enc["name"]: enc
                for enc in itertools.chain(
                    encodings["activation_encodings"], encodings["param_encodings"]
                )
            }

        """
        When: Call _adjust_weight_scales_for_int32_bias() and _concretize_int32_bias_quantizers() before export
        """
        sim._adjust_weight_scales_for_int32_bias()
        sim._concretize_int32_bias_quantizers()
        sim.export(tmp_path, "after_weight_adj")

        with open(tmp_path / "after_weight_adj.encodings") as f:
            encodings = json.load(f)
            after_weight_adj = {
                enc["name"]: enc
                for enc in itertools.chain(
                    encodings["activation_encodings"], encodings["param_encodings"]
                )
            }
        for op, (weight, bias) in linear_ops_with_bias.items():
            if bias is None:
                continue

            input, *_ = op.inputs
            bias_scale = np.array(after_weight_adj[bias.name]["scale"])
            weight_scale = np.array(after_weight_adj[weight.name]["scale"])
            weight_qtzr = sim.qc_quantize_op_dict[weight.name]
            input_qtzr = sim._get_enabled_quantizer(input.name)
            input_scale = input_qtzr._get_scale()

            bias_proto = sim.model.get_initializer(bias.name) or next(
                iter(
                    node.attribute[0].t
                    for node in sim.model.graph().node
                    if node.output == [bias.name]
                )
            )
            bias_value = onnx.numpy_helper.to_array(bias_proto)

            """
            Then: If the Bias is in exported encodings, then the bias_quantized value should be clipped to 2.14748365e+09
                  For BQ/LPBQ quantizers, bias_quantized values can be greater than 2.14748365e+09, since weight adjustment is not applied.
            """
            bias_quantized = bias_value / bias_scale
            overflow_mask = np.any(bias_quantized >= 2**31)
            assert np.all(overflow_mask)

            if np.any(overflow_mask):
                """
                Then: If the Bias is in exported encodings, then the adjusted weight scale must match (bias_float/(2**31 * input_scale))
                      For BQ/LPBQ scales, not weight scales adjustment applied, so it should match before the weight adjustment scale.
                """
                if weight_qtzr.quant_info.blockSize:
                    expected_weight_scale = np.array(
                        before_weight_adj[weight.name]["scale"]
                    )  # Before and after scales should be same for BQ/LPBQ
                else:
                    expected_weight_scale = bias_value / (2**31 * input_scale)
                assert np.allclose(weight_scale, expected_weight_scale)


@pytest.mark.parametrize(
    "model_factory,             input_shape,     block_size, lpbq, enable_mp",
    [
        (single_residual_model, (1, 3, 32, 32), None, False, True),
        (single_residual_model, (1, 3, 32, 32), None, False, False),
        (single_residual_model, (1, 3, 32, 32), 4, False, False),
        (single_residual_model, (1, 3, 32, 32), 4, True, False),
        (transposed_conv_model, (10, 10, 4, 4), None, False, False),
        (transposed_conv_model, (10, 10, 4, 4), 5, False, False),
        (transposed_conv_model, (10, 10, 4, 4), 5, True, False),
        (batchnorm_model, (10, 10, 8, 8), None, False, False),
        (batchnorm_model_constants, (10, 10, 8, 8), None, False, False),
        (instance_norm_model, (2, 10, 24, 24), None, False, False),
        (layernorm_model, (1, 4, 64, 64), None, False, False),
        # TODO: Add tests with GroupNormalization
    ],
)
def test_bias_export(model_factory, input_shape, block_size, lpbq, enable_mp, tmp_path):
    model = model_factory()
    input = np.random.randn(*input_shape).astype(np.float32)

    """
    When: Call _concretize_int32_bias_quantizers() before export
    """
    sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf)

    if block_size:
        op_types = ("Conv", "ConvTranspose", "Gemm")
        if lpbq:
            set_grouped_blockwise_quantization_for_weights(
                sim,
                op_types,
                bitwidth=4,
                decompressed_bw=8,
                block_size=block_size,
                strict=False,
            )
        else:
            set_blockwise_quantization_for_weights(
                sim,
                op_types,
                bitwidth=4,
                symmetric=True,
                block_size=block_size,
                strict=False,
            )

    if enable_mp:
        ops_with_disabled_weight_quant = [
            (wb[0].name, wb[1].name)
            for op in sim.connected_graph.get_all_ops().values()
            if (wb := sim._get_weight_and_bias(op))[1]
        ]
        # Disable weight quantizers except the last layer in the list.
        if len(ops_with_disabled_weight_quant) > 1:
            ops_with_disabled_weight_quant = ops_with_disabled_weight_quant[:-1]

        for weight, _ in ops_with_disabled_weight_quant:
            weight_qtzr = sim.qc_quantize_op_dict[weight]
            weight_qtzr.enabled = False

    sim.compute_encodings(lambda sess: sess.run(None, {"input": input}))
    sim._concretize_int32_bias_quantizers()
    sim.export(tmp_path, "model")

    with open(tmp_path / "model.encodings") as f:
        encodings = json.load(f)

    exported_encodings = {
        enc["name"]: enc
        for enc in itertools.chain(
            encodings["activation_encodings"], encodings["param_encodings"]
        )
    }

    # sanity check
    if block_size:
        enc_type = "LPBQ" if lpbq else "PER_BLOCK"
        assert any(enc["enc_type"] == enc_type for enc in exported_encodings.values())

    """
    Then: If the Bias is in exported encodings, then the weight also must be in exported encodings and vice-versa
          If both are not present in the list, assertion passes.
    """
    for op in sim.connected_graph.get_all_ops().values():
        weight, bias = sim._get_weight_and_bias(op)
        if bias:
            assert (bias.name in exported_encodings) == (
                weight.name in exported_encodings
            )

    """
    Then: For linear ops such as Conv, ConvTranspose, and Gemm,
          bias encoding should be derived analytically from input and weight encodings
    """
    linear_ops_with_bias = {
        op
        for op in sim.connected_graph.get_all_ops().values()
        if op.type in ("Conv", "ConvTranspose", "Gemm")
        and "bias" in [param_type for _, param_type in op.parameters.values()]
    }

    for op in linear_ops_with_bias:
        input, weight, bias = op.inputs
        if bias.name not in exported_encodings:
            continue

        assert all(
            offset == -(2**31) for offset in exported_encodings[bias.name]["offset"]
        )

        weight_scale = np.array(exported_encodings[weight.name]["scale"])

        if exported_encodings[weight.name]["enc_type"] == "PER_BLOCK":
            weight_scale = weight_scale.reshape(
                sim.qc_quantize_op_dict[weight.name]._encoding_shape()
            )
            block_axis = 0 if op.type == "ConvTranspose" else 1
            weight_scale = weight_scale.max(axis=block_axis).flatten()

        bias_scale = np.array(exported_encodings[bias.name]["scale"])
        try:
            input_scale = np.array(exported_encodings[input.name]["scale"])
        except KeyError:
            continue  # TODO: Remove this exception. Find input scale more smartly

        assert np.allclose(bias_scale, input_scale * weight_scale)

    """
    Then: For non-linear ops such as BatchNormalization, InstanceNormalization, LayerNormalization,
          and GroupNormalization, bias encoding should be calibrated statistically
    """
    nonlinear_ops_with_bias = {
        op
        for op in sim.connected_graph.get_all_ops().values()
        if op.type
        in (
            "BatchNormalization",
            "InstanceNormalizationLayerNormalizationGroupNormalization",
        )
        and "bias" in [param_type for _, param_type in op.parameters.values()]
    }

    for op in nonlinear_ops_with_bias:
        input, weight, bias, *_ = op.inputs
        if bias.name not in exported_encodings:
            print()
            assert False
        assert all(
            offset == -(2**31) for offset in exported_encodings[bias.name]["offset"]
        )

        weight_scale = np.array(exported_encodings[weight.name]["scale"])
        bias_scale = np.array(exported_encodings[bias.name]["scale"])
        try:
            input_scale = np.array(exported_encodings[input.name]["scale"])
        except KeyError:
            continue

        bias_proto = sim.model.get_initializer(bias.name) or next(
            iter(
                node.attribute[0].t
                for node in sim.model.graph().node
                if node.output == [bias.name]
            )
        )

        bias_value = onnx.numpy_helper.to_array(bias_proto)
        expected_bias_scale = np.maximum(abs(bias_value) / 2**31, _INT32_MINIMUM_SCALE)
        assert np.allclose(bias_scale, expected_bias_scale)


def _parse_type(type_str: str) -> tuple[str, int]:
    if type_str.startswith("int"):
        return "int", int(type_str[3:])
    if type_str.startswith("uint"):
        return "uint", int(type_str[4:])
    if type_str.startswith("float"):
        return "float", int(type_str[5:])
    raise RuntimeError


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize("export_int32_bias_encodings", [False, True])
@pytest.mark.parametrize(
    "param_dtype, activation_dtype",
    [
        ("int4", "uint16"),
        ("int4", "float16"),
        ("int8", "uint8"),
        ("int8", "uint16"),
        ("float16", "float16"),
    ],
)
@pytest.mark.parametrize(
    "model_factory, tolerance",
    [
        (partial(single_residual_model, opset_version=21), 1),
        (partial(transposed_conv_model, opset_version=21), 2),
        # normalization layers tolerance rationale:
        #   * off-by-one in input/output qtzn respectively
        #   * No off-by-one in weight qtzn; weights are exported spot-on
        (partial(standalone_batchnorm, (1, 32, 4096, 10)), 2),
        (partial(standalone_batchnorm_constants, (1, 32, 4096, 10)), 2),
        (partial(standalone_instancenorm, (1, 32, 40960)), 2),
        (partial(standalone_layernorm, (1, 40960, 32)), 2),
    ],
)
def test_to_onnx_qdq(
    model_factory,
    tolerance: int,
    param_dtype: str,
    activation_dtype: str,
    export_int32_bias_encodings: bool,
    prequantize_constants: bool,
    seed: int,
    tmp_path: pathlib.Path,
):
    ort.set_seed(seed)
    np.random.seed(seed)

    model = model_factory()
    input_names = [inp.name for inp in getattr(model, "model", model).graph.input]
    output_names = [out.name for out in getattr(model, "model", model).graph.output]

    param_kind, param_bw = _parse_type(param_dtype)
    activation_kind, activation_bw = _parse_type(activation_dtype)
    sim = QuantizationSimModel(
        model,
        param_type=param_dtype.removeprefix("u"),
        activation_type=activation_dtype.removeprefix("u"),
        config_file="htp_v81",
    )

    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    input = np.random.randn(*input_shape).astype(np.float32)

    """
    When: Create a pure onnx model with sim._to_onnx_qdq()
    """
    sim.compute_encodings([{"input": input}])

    if export_int32_bias_encodings:
        sim._concretize_int32_bias_quantizers()
        # FIXME: Need extra tolerance due to numerical instability of AIMET int32 bias qdq.
        tolerance += 1

    (out_sim,) = sim.session.run(None, {"input": input})

    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx",
        save_as_external_data=True,
        prequantize_constants=prequantize_constants,
    )

    """
    Then: Exported model should preserve the original I/O names
    """
    assert input_names == [inp.name for inp in onnx_qdq_model.graph.input]
    assert output_names == [out.name for out in onnx_qdq_model.graph.output]

    """
    Then: Onnx QDQ model should contain as many DequantizeLinear as the number of of ENABLED QcQuantizers
    """
    assert len(
        [
            node
            for node in onnx_qdq_model.graph.node
            if node.op_type == "DequantizeLinear"
        ]
    ) == len(
        [
            qtzr
            for qtzr in sim.qc_quantize_op_dict.values()
            if qtzr.enabled
            and (qtzr.data_type == QuantizationDataType.int or qtzr.bitwidth < 16)
        ]
    )

    # NOTE: Should disable all ORT graph optimization to circumvent known bugs
    # in CPUExecutionProvider operator fusing.
    # ORT CPUExecutionProvider produces corrupted output after fusing pattern A to B:
    #
    # A:
    #   x -----> QuantizeLinear -> DequantizeLinear -+
    #   W -----> QuantizeLinear -> DequantizeLinear -+-> Conv
    #   b_int32 -----------------> DequantizeLinear -+
    #
    # B:
    #   x -----> QuantizeLinear -+
    #   W -----> QuantizeLinear -+---------------------> QLinearConv
    #   b_int32 -----------------+
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    """
    Then: Output of the pure onnx model should be equal to that of sim.session
    """
    if activation_kind in ("uint", "int"):
        # Allow off-by-N error
        atol = tolerance * sim.qc_quantize_op_dict["output"].get_encodings()[0].delta
    else:
        # Allow off-by-3 error, using float16.eps as a pseudo-scale
        atol = 3 * np.finfo(np.float16).eps

    rtol = 1e-3 * tolerance
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(), sess_options=sess_options
    )
    (out_onnx_qdq,) = sess.run(None, {"input": input})
    assert np.allclose(out_sim, out_onnx_qdq, atol=atol, rtol=rtol)


@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize("input_model_opset", range(9, 22))
@pytest.mark.parametrize(
    "param_bw, act_bw, per_channel, minimum_required_opset",
    [
        (4, 8, False, 21),
        (4, 16, False, 21),
        (8, 8, False, 10),
        (8, 16, False, 21),
        (16, 16, False, 21),
        (4, 16, True, 21),
        (8, 8, True, 13),
        (8, 16, True, 21),
        (16, 16, True, 21),
        (8, 12, True, -1),
    ],
)
def test_onnx_qdq_opset_compatibility(
    input_model_opset: int,
    param_bw: int,
    act_bw: int,
    per_channel: bool,
    minimum_required_opset: int,
    prequantize_constants: bool,
    tmp_path: pathlib.Path,
):
    ort.set_seed(1)
    np.random.seed(1)

    input_shape = (1, 3, 32, 32)
    model = single_residual_model(opset_version=input_model_opset)
    input_names = [inp.name for inp in model.graph().input]
    output_names = [out.name for out in model.graph().output]

    config_file = "htp_v81" if per_channel else get_path_for_per_tensor_config()
    sim = QuantizationSimModel(
        model,
        param_type=qtype.int(param_bw),
        activation_type=qtype.int(act_bw),
        config_file=config_file,
    )
    input = np.random.randn(*input_shape).astype(np.float32)
    sim.compute_encodings([{"input": input}])

    if minimum_required_opset < 0:
        with pytest.raises(RuntimeError):
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=prequantize_constants
            )
        return

    (out_sim,) = sim.session.run(None, {"input": input})

    """
    When: Create a pure onnx model with sim._to_onnx_qdq()
    Then:
      1. Onnx opset should be upgraded to minimum required opset if needed
      2. Should pass onnx checker
    """
    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )
    output_model_opset = onnx_qdq_model.opset_import[0].version
    assert output_model_opset == max(input_model_opset, minimum_required_opset)
    onnx.checker.check_model(onnx_qdq_model)

    op_map = {node.name: node for node in onnx_qdq_model.graph.node}
    output_to_op_map = dict()
    for node in op_map.values():
        tensor_name = node.output[0]
        output_to_op_map[tensor_name] = node

    param_names = set(
        param.name
        for op in sim.connected_graph.get_all_ops().values()
        for param, _ in op.parameters.values()
    )
    q_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    expected_output_dtypes = {
        q.output[0]: getattr(onnx.TensorProto, f"INT{param_bw}")
        if q.input[0] in param_names
        else getattr(onnx.TensorProto, f"UINT{act_bw}")
        for q in q_nodes
    }

    """
    Then: Exported model should preserve the original I/O names
    """
    assert input_names == [inp.name for inp in onnx_qdq_model.graph.input]
    assert output_names == [out.name for out in onnx_qdq_model.graph.output]

    """
    Then: Model input/outputs should be associated with QDQ
    """
    input_names = set(inp.name for inp in onnx_qdq_model.graph.input)
    output_names = set(out.name for out in onnx_qdq_model.graph.output)

    for node in onnx_qdq_model.graph.node:
        if node.input and node.input[0] in input_names:
            assert node.op_type == "QuantizeLinear"
            input_names.remove(node.input[0])
        if node.output and node.output[0] in output_names:
            assert node.op_type == "DequantizeLinear"
            output_names.remove(node.output[0])

    assert not input_names
    assert not output_names

    """
    When: Infer output dtype of QuantizeLinear
    Then: Output dtype should match expected param/activatoin dtype
    """
    onnx_qdq_model = onnx.shape_inference.infer_shapes(onnx_qdq_model)

    for val in onnx_qdq_model.graph.value_info:
        if val.name in expected_output_dtypes:
            expected_dtype = expected_output_dtypes[val.name]
            assert val.type.tensor_type.elem_type == expected_dtype

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    """
    Then: Output of the pure onnx model should be equal to that of sim.session
    """
    # Allow off-by-1 error
    atol = sim.qc_quantize_op_dict["output"].get_encodings()[0].delta
    rtol = 1e-3
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(), sess_options=sess_options
    )
    (out_onnx_qdq,) = sess.run(None, {"input": input})
    assert np.allclose(out_sim, out_onnx_qdq, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "model_factory",
    [
        model_with_split_matmul,
        conv_relu,
    ],
)
def test_insert_data_movement_op_quantizers(model_factory, tmp_path):
    if model_factory == conv_relu:
        pytest.skip(reason="Need another PR to pass this case")

    model = model_factory()
    sim = QuantizationSimModel(model)
    input_name = sim.model.model.graph.input[0].name
    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    inputs = {input_name: np.random.randn(*input_shape).astype(np.float32)}
    sim.compute_encodings(lambda session: session.run(None, inputs))
    onnx_qdq_before = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )

    """
    When: Call _insert_data_movement_op_output_quantizers before _to_onnx_qdq()
    Then:
      1. All node outputs should fed into QuantizeLinear
      2. All node inputs should be an output of DequantizeLinear
      3. Model output should be EQUAL with/without data movement op output QDQ
    """
    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_after = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )

    q_nodes = [
        node for node in onnx_qdq_after.graph.node if node.op_type == "QuantizeLinear"
    ]
    all_outputs = itertools.chain(
        *(
            node.output
            for node in onnx_qdq_after.graph.node
            if node.op_type not in ("QuantizeLinear", "DequantizeLinear")
        )
    )
    for output in all_outputs:
        assert any(output == q.input[0] for q in q_nodes)

    dq_nodes = [
        node for node in onnx_qdq_after.graph.node if node.op_type == "DequantizeLinear"
    ]
    all_inputs = itertools.chain(
        node.input[0]
        for node in onnx_qdq_after.graph.node
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear")
    )
    for input in all_inputs:
        assert any(input == dq.output[0] for dq in dq_nodes)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_before = ort.InferenceSession(
        onnx_qdq_before.SerializeToString(), sess_options=sess_options
    )
    sess_after = ort.InferenceSession(
        onnx_qdq_after.SerializeToString(), sess_options=sess_options
    )
    for _ in range(10):
        inputs = {input_name: np.random.randn(*input_shape).astype(np.float32)}
        outputs_before = sess_before.run(None, inputs)
        outputs_after = sess_after.run(None, inputs)
        for out_before, out_after in zip(outputs_before, outputs_after):
            assert np.all(out_before == out_after)


@pytest.mark.parametrize(
    "model_factory", [model_with_split_matmul, reshape_with_multiple_consumers]
)
def test_insert_data_movement_op_edge_case(model_factory, tmp_path):
    """
    Given: Model with edge case scenarios

      model_with_split_matmul:

                         +--> Quantize
          input -> Split +--> ...
                         +--> ...

      reshape_with_multiple_consumers:

                           +--> Quantize
          input -> Reshape +
                           +--> ...
    """
    model = model_factory()
    with patch("aimet_onnx.quantsim.op_outputs_to_ignore", []):
        sim = QuantizationSimModel(model)

    # Disable all quantizers...
    for qtzr in sim.qc_quantize_op_dict.values():
        qtzr.enabled = False

    # Except output of Reshape and Split
    for node in sim.model.model.graph.node:
        if node.op_type in ("Reshape", "Split"):
            for output in node.output:
                sim.qc_quantize_op_dict[output].enabled = True

    input_name = sim.model.model.graph.input[0].name
    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    inputs = {input_name: np.random.randn(*input_shape).astype(np.float32)}
    sim.compute_encodings(lambda session: session.run(None, inputs))
    onnx_qdq_before = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )

    """
    When: Call _insert_data_movement_op_output_quantizers before _to_onnx_qdq()
    Then: Output encoding should NOT be reused for input quantization
    """
    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_after = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )
    assert onnx_qdq_before == onnx_qdq_after


@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_to_onnx_qdq_lpbq(seed: int, prequantize_constants: bool, tmp_path):
    ort.set_seed(seed)
    np.random.seed(seed)

    model = standalone_gemm(in_channels=16, out_channels=16)
    sim = QuantizationSimModel(
        model,
        param_type="int4",
        activation_type="int16",
        config_file="htp_v81",
    )

    set_grouped_blockwise_quantization_for_weights(
        sim,
        op_types=("MatMul", "Conv", "Gemm"),
        bitwidth=4,
        decompressed_bw=8,
        block_size=4,
        strict=False,
    )

    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    input = np.random.randn(*input_shape).astype(np.float32)

    """
    When: Create a pure onnx model with sim._to_onnx_qdq()
    """
    sim.compute_encodings([{"input": input}])

    (out_sim,) = sim.session.run(None, {"input": input})

    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )

    """
    Then: Onnx QDQ model should contain as many DequantizeLinear as the number of of enabled QcQuantizers
    """
    num_dq = len(
        [
            node
            for node in onnx_qdq_model.graph.node
            if node.op_type == "DequantizeLinear"
        ]
    )
    expected_num_dq = sum(
        # GroupedBlockQuantizeDequantize is mapped to two DequantizeLinears when exported
        2 if isinstance(qtzr, GroupedBlockQuantizeDequantize) else 1
        for qtzr in sim.qc_quantize_op_dict.values()
        if qtzr.enabled
        and (qtzr.data_type == QuantizationDataType.int or qtzr.bitwidth < 16)
    )
    assert num_dq == expected_num_dq

    """
    Then: Output of the pure onnx model should be equal to that of sim.session
    """
    # Allow off-by-1 error
    atol = 1 * sim.qc_quantize_op_dict["output"].get_encodings()[0].delta
    sess = ort.InferenceSession(onnx_qdq_model.SerializeToString())
    (out_onnx_qdq,) = sess.run(None, {"input": input})
    assert np.allclose(out_sim, out_onnx_qdq, atol=atol)


class TestDynamicWeightSymmetryMapping:
    def _assert_uint_activation(self, model: onnx.ModelProto):
        model = onnx.shape_inference.infer_shapes(model)

        dtypes = {
            val.name: val.type.tensor_type.elem_type for val in model.graph.value_info
        }
        param_names = set(init.name for init in model.graph.initializer)
        q_nodes = [
            node for node in model.graph.node if node.op_type == "QuantizeLinear"
        ]

        for q in q_nodes:
            output_dtype = dtypes[q.output[0]]
            if q.input[0] not in param_names:
                assert output_dtype in (
                    onnx.TensorProto.UINT4,
                    onnx.TensorProto.UINT8,
                    onnx.TensorProto.UINT16,
                    onnx.TensorProto.UINT32,
                )

    @pytest.mark.parametrize("default_symmetry", [True, False, None])
    @pytest.mark.parametrize("matmul_op_symmetry", [True, False, None])
    def test_dynamic_matmul_symmetry(
        self, default_symmetry, matmul_op_symmetry, tmp_path
    ):
        model = models_for_tests.dynamic_matmul_model(batch_size=1)
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        # Expected symmetry is
        #  - Default symmetry, if present
        #  - Op_Type param symmetry, if present
        expected_symmetry = False
        if default_symmetry is not None:
            quantsim_config["defaults"]["params"]["is_symmetric"] = str(
                default_symmetry
            )
            expected_symmetry = default_symmetry

        if matmul_op_symmetry is not None:
            op_symmetry = {
                "params": {"weight": {"is_symmetric": str(matmul_op_symmetry)}}
            }
            quantsim_config["op_type"]["MatMul"] = op_symmetry
            expected_symmetry = matmul_op_symmetry

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                default_activation_bw=16,
            )

            assert sim.qc_quantize_op_dict["/linear/Gemm_output_0"].enabled
            assert (
                sim.qc_quantize_op_dict["/linear/Gemm_output_0"].use_symmetric_encodings
                == expected_symmetry
            )

            """
            When: Export to onnx QDQ
            Then: All activation quantizers must be uint
            """
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=False
            )
            self._assert_uint_activation(onnx_qdq_model)

    def test_dynamic_conv_symmetry(self, tmp_path):
        model = models_for_tests.dynamic_conv_model()
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                default_activation_bw=16,
            )

            assert sim.qc_quantize_op_dict["dynamic_conv.weight"].enabled
            assert sim.qc_quantize_op_dict[
                "dynamic_conv.weight"
            ].use_symmetric_encodings

            """
            When: Export to onnx QDQ
            Then: All activation quantizers must be uint
            """
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=False
            )
            self._assert_uint_activation(onnx_qdq_model)

    @pytest.mark.parametrize("conv_transpose", [True, False])
    def test_dynamic_conv_symmetry(self, conv_transpose, tmp_path):
        model = models_for_tests.dynamic_conv_model(conv_transpose=conv_transpose)
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                default_activation_bw=16,
            )

            assert sim.qc_quantize_op_dict["dynamic_conv.weight"].enabled
            assert sim.qc_quantize_op_dict[
                "dynamic_conv.weight"
            ].use_symmetric_encodings

            """
            When: Export to onnx QDQ
            Then: All activation quantizers must be uint
            """
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=False
            )
            self._assert_uint_activation(onnx_qdq_model)

    @pytest.mark.parametrize("conv_transpose", [True, False])
    def test_dynamic_conv_symmetry(self, conv_transpose, tmp_path):
        model = models_for_tests.dynamic_conv_model(conv_transpose=conv_transpose)
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                default_activation_bw=16,
            )

            assert sim.qc_quantize_op_dict["dynamic_conv.weight"].enabled
            assert sim.qc_quantize_op_dict[
                "dynamic_conv.weight"
            ].use_symmetric_encodings

            """
            When: Export to onnx QDQ
            Then: All activation quantizers must be uint
            """
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=False
            )
            self._assert_uint_activation(onnx_qdq_model)

    def test_dynamic_gemm_symmetry(self, tmp_path):
        model = models_for_tests.dynamic_gemm(in_channels=10, out_channels=10)
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "per_channel_quantization": "False",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {"is_output_quantized": "True"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                path=tempdir,
                config_file=os.path.join(tempdir, "quantsim_config.json"),
                default_activation_bw=16,
            )

            assert sim.qc_quantize_op_dict["weights"].enabled
            assert sim.qc_quantize_op_dict["weights"].use_symmetric_encodings

            """
            When: Export to onnx QDQ
            Then: All activation quantizers must be uint
            """
            onnx_qdq_model = sim._to_onnx_qdq(
                tmp_path / "model.onnx", prequantize_constants=False
            )
            self._assert_uint_activation(onnx_qdq_model)

    @pytest.mark.skip
    def test_quantsim_create_speed(self, tmp_path):
        from onnx import helper, TensorProto

        model = models_for_tests.conv_relu_model()

        # Parameters
        graph = model.graph
        num_layers = 3000  # Number of Conv+Relu pairs to add
        input_shape = [1, 3, 224, 224]  # Example input shape
        conv_out_channels = 3
        kernel_shape = [3, 3]

        # Find last output tensor name
        last_output = graph.output[0].name

        for i in range(num_layers):
            # Conv weights
            weight_name = f"conv{i}_weight"
            weight_shape = [conv_out_channels, input_shape[1], *kernel_shape]
            weight_data = np.random.randn(*weight_shape).astype(np.float32)
            weight_tensor = helper.make_tensor(
                weight_name, TensorProto.FLOAT, weight_shape, weight_data.flatten()
            )
            graph.initializer.append(weight_tensor)

            # Conv node
            conv_output = f"conv{i}_out"
            conv_node = helper.make_node(
                "Conv",
                name=f"Conv_{i + 4}",
                inputs=[last_output, weight_name],
                outputs=[conv_output],
                kernel_shape=kernel_shape,
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            )
            graph.node.append(conv_node)

            # Relu node
            relu_output = f"relu{i}_out"
            relu_node = helper.make_node(
                "Relu",
                name=f"Relu_{i + 4}",
                inputs=[conv_output],
                outputs=[relu_output],
            )
            graph.node.append(relu_node)

            last_output = relu_output
            input_shape = [1, conv_out_channels, input_shape[2], input_shape[3]]

        # Update graph output to last relu
        graph.output[0].name = last_output

        start_time = time.time()
        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)

        # qdq_model = sim.to_onnx_qdq()
        # onnx.save(qdq_model, "large_model_qdq.onnx")

        # Make all quantizers point to the same object
        quantizer = sim._get_enabled_quantizer("relu0_out")

        quant_dict = {}
        for i in range(1, num_layers):
            quant_dict[f"relu{i}_out"] = quantizer

        sim.set_quantizers(quant_dict)

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.6f} seconds")


def test_onnx_qdq_export_output_name_swapping(tmp_path):
    """
    Given:

                      +------------------> (output_0)
    x ---> Sigmoid ---+----> MaxPool ----> (output_1)
    """
    model = make_model(
        graph=onnx.helper.make_graph(
            name="model",
            nodes=[
                onnx.helper.make_node(
                    "Sigmoid",
                    name="Sigmoid",
                    inputs=["input"],
                    outputs=["output_0"],
                ),
                onnx.helper.make_node(
                    "AveragePool",
                    name="AveragePool",
                    inputs=["output_0"],
                    outputs=["output_1"],
                    kernel_shape=(3, 3),
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    "input",
                    onnx.helper.make_tensor_type_proto(
                        onnx.TensorProto.FLOAT, (1, 3, 224, 224)
                    ),
                )
            ],
            outputs=[
                onnx.helper.make_value_info(
                    "output_0",
                    onnx.helper.make_tensor_type_proto(
                        onnx.TensorProto.FLOAT, (1, 3, 224, 224)
                    ),
                ),
                onnx.helper.make_value_info(
                    "output_1",
                    onnx.helper.make_tensor_type_proto(
                        onnx.TensorProto.FLOAT, (1, 3, 222, 222)
                    ),
                ),
            ],
        )
    )
    onnx.checker.check_model(model)

    """
    When: Export to onnx QDQ
    """
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    sim = QuantizationSimModel(model)
    sim.compute_encodings(lambda sess: sess.run(None, {"input": x}))
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )

    """
    Then: Exported model should look like this:

                                        +-------------------> (output_0)
            x -> QDQ -> Sigmoid -> QDQ -+-> AveragePool -> QDQ -> (output_1)


    NOT like this:

                                        +---> QDQ ----------> (output_0)
            x -> QDQ -> Sigmoid --------+-> AveragePool -> QDQ -> (output_1)
    """
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=False
    )

    # Assert all inputs/outputs of all nodes are associated with Q/DQ
    q_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    all_outputs = itertools.chain(
        *(
            node.output
            for node in onnx_qdq_model.graph.node
            if node.op_type not in ("QuantizeLinear", "DequantizeLinear")
        )
    )
    for output in all_outputs:
        assert any(output == q.input[0] for q in q_nodes)

    dq_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "DequantizeLinear"
    ]
    all_inputs = itertools.chain(
        node.input[0]
        for node in onnx_qdq_model.graph.node
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear")
    )
    for input in all_inputs:
        assert any(input == dq.output[0] for dq in dq_nodes)


@pytest.mark.parametrize("export_int32_bias_encodings", [False, True])
@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize(
    "param_type, activation_type",
    [
        (aimet_onnx.int4, aimet_onnx.int4),
        (aimet_onnx.int8, aimet_onnx.int8),
        (aimet_onnx.int8, aimet_onnx.int16),
    ],
)
@pytest.mark.parametrize(
    "model_factory",
    [
        partial(single_residual_model, opset_version=21),
        partial(transposed_conv_model, opset_version=21),
        partial(standalone_batchnorm, (1, 32, 4096, 10)),
        partial(standalone_batchnorm_constants, (1, 32, 4096, 10)),
        partial(standalone_instancenorm, (1, 32, 40960)),
        partial(standalone_layernorm, (1, 40960, 32)),
    ],
)
def test_from_onnx_qdq(
    model_factory,
    param_type,
    activation_type,
    prequantize_constants: bool,
    export_int32_bias_encodings: bool,
    tmp_path: pathlib.Path,
):
    """
    Given: onnx QDQ model exported from aimet QuantizationSimModel
    """
    sim = QuantizationSimModel(
        model_factory(),
        param_type=param_type,
        activation_type=activation_type,
        config_file="htp_v81",
    )
    input_name = sim.model.model.graph.input[0].name
    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    inputs = {input_name: np.random.randn(*input_shape).astype(np.float32)}

    sim.compute_encodings([inputs])
    if export_int32_bias_encodings:
        sim._concretize_int32_bias_quantizers()
    qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )

    """
    When: Create sim from onnx QDQ model
    Then: The new sim should be in same state as the original sim
    """
    sim_2 = QuantizationSimModel._from_onnx_qdq(
        sim._to_onnx_qdq(
            tmp_path / "model.onnx", prequantize_constants=prequantize_constants
        ),
        config_file="htp_v81",
    )
    if export_int32_bias_encodings:
        sim_2._concretize_int32_bias_quantizers()
    _assert_sim_equal(sim, sim_2)
    assert np.allclose(
        sim.session.run(None, inputs),
        sim_2.session.run(None, inputs),
    )

    """
    When: Call compute_encodings with new sim
    Then: All states of the new sim should remain unchanged
    """
    sim_2 = QuantizationSimModel._from_onnx_qdq(
        sim._to_onnx_qdq(
            tmp_path / "model.onnx", prequantize_constants=prequantize_constants
        ),
        config_file="htp_v81",
    )
    sim_2.compute_encodings([{key: val * 2 for key, val in inputs.items()}])
    if export_int32_bias_encodings:
        sim_2._concretize_int32_bias_quantizers()
    _assert_sim_equal(sim, sim_2)
    assert np.allclose(
        sim.session.run(None, inputs),
        sim_2.session.run(None, inputs),
    )

    """
    When: Export onnx QDQ from the new sim
    Then: The new onnx QDQ model should be in same state as the original onnx QDQ
    """
    qdq_model_2 = sim_2._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )
    assert qdq_model.graph.input == qdq_model_2.graph.input
    assert qdq_model.graph.output == qdq_model_2.graph.output

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        qdq_model.SerializeToString(), sess_options=sess_options
    )
    sess_2 = ort.InferenceSession(
        qdq_model_2.SerializeToString(), sess_options=sess_options
    )

    assert np.allclose(sess.run(None, inputs), sess_2.run(None, inputs))


@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_from_onnx_qdq_lpbq(seed: int, prequantize_constants: bool, tmp_path):
    ort.set_seed(seed)
    np.random.seed(seed)

    model = standalone_gemm(in_channels=16, out_channels=16)
    sim = QuantizationSimModel(
        model,
        param_type="int4",
        activation_type="int16",
        config_file="htp_v81",
    )

    set_grouped_blockwise_quantization_for_weights(
        sim,
        op_types=("MatMul", "Conv", "Gemm"),
        bitwidth=4,
        decompressed_bw=8,
        block_size=4,
        strict=False,
    )

    input_shape = tuple(
        dim.dim_value
        for dim in sim.model.model.graph.input[0].type.tensor_type.shape.dim
    )
    input = np.random.randn(*input_shape).astype(np.float32)

    """
    When: Create a pure onnx model with sim._to_onnx_qdq()
    """
    sim.compute_encodings([{"input": input}])

    sim._insert_data_movement_op_output_quantizers()
    onnx_qdq_model = sim._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )

    """
    When: Create sim from onnx QDQ model
    Then: The new sim should be in same state as the original sim
    """
    sim_2 = QuantizationSimModel._from_onnx_qdq(
        sim._to_onnx_qdq(
            tmp_path / "model.onnx", prequantize_constants=prequantize_constants
        ),
        config_file="htp_v81",
    )
    _assert_sim_equal(sim, sim_2)
    assert np.allclose(
        sim.session.run(None, {"input": input}),
        sim_2.session.run(None, {"input": input}),
    )

    """
    When: Call compute_encodings with new sim
    Then: All states of the new sim should remain unchanged
    """
    sim_2.compute_encodings([{"input": input * 2}])
    _assert_sim_equal(sim, sim_2)
    assert np.allclose(
        sim.session.run(None, {"input": input}),
        sim_2.session.run(None, {"input": input}),
    )

    """
    When: Export onnx QDQ from the new sim
    Then: The new onnx QDQ model should be in same state as the original onnx QDQ
    """
    onnx_qdq_model_2 = sim_2._to_onnx_qdq(
        tmp_path / "model.onnx", prequantize_constants=prequantize_constants
    )

    assert onnx_qdq_model.graph.input == onnx_qdq_model_2.graph.input
    assert onnx_qdq_model.graph.output == onnx_qdq_model_2.graph.output

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(), sess_options=sess_options
    )
    sess_2 = ort.InferenceSession(
        onnx_qdq_model_2.SerializeToString(), sess_options=sess_options
    )

    assert np.allclose(
        sess.run(None, {"input": input}),
        sess_2.run(None, {"input": input}),
    )


def _assert_sim_equal(sim_1: QuantizationSimModel, sim_2: QuantizationSimModel):
    assert len(sim_1.activation_names) == len(sim_2.activation_names)
    for key1, key2 in zip(
        sorted(sim_1.activation_names), sorted(sim_2.activation_names)
    ):
        assert key1 == key2 or key1 == key2 + "_qdq" or key2 == key1 + "_qdq"

    assert len(sim_1.param_names) == len(sim_2.param_names)
    for key1, key2 in zip(sorted(sim_1.param_names), sorted(sim_2.param_names)):
        assert key1 == key2 or key1 == key2 + "_qdq" or key2 == key1 + "_qdq"

    assert len(sim_1.qc_quantize_op_dict) == len(sim_2.qc_quantize_op_dict)
    for key1, key2 in zip(
        sorted(sim_1.qc_quantize_op_dict), sorted(sim_2.qc_quantize_op_dict)
    ):
        assert key1 == key2 or key1 == key2 + "_qdq" or key2 == key1 + "_qdq"

        qtzr_1 = sim_1.qc_quantize_op_dict[key1]
        qtzr_2 = sim_2.qc_quantize_op_dict[key2]

        assert type(qtzr_1) == type(qtzr_2)
        assert qtzr_1.enabled == qtzr_2.enabled
        assert qtzr_1.tensor_quantizer_params == qtzr_2.tensor_quantizer_params

        if not qtzr_1.enabled:
            continue

        e1 = qtzr_1.export_encodings("2.0.0")
        e2 = qtzr_2.export_encodings("2.0.0")
        assert np.allclose(e1.get("y_scale", 0), e2.get("y_scale", 0))
        assert np.allclose(
            e1.get("per_channel_float_scale", 0), e2.get("per_channel_float_scale", 0)
        )
        assert e1.get("per_block_int_scale") == e2.get("per_block_int_scale")
        assert e1.get("y_zero_point") == e2.get("y_zero_point")
        assert e1.get("output_dtype") == e2.get("output_dtype")
        assert e1.get("axis") == e2.get("axis")
        assert e1.get("block_size") == e2.get("block_size")


def test_from_onnx_qdq_output_dtype():
    """
    Given: onnx QDQ model utilizing "output_dtype" attribute
    """
    model = make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 21)],
        graph=onnx.helper.make_graph(
            name="model",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            initializer=[
                onnx.numpy_helper.from_array(
                    np.array(0.1, dtype=np.float32), name="input_scale"
                ),
                onnx.numpy_helper.from_array(
                    np.array(5, dtype=np.uint8), name="input_zero_point"
                ),
                onnx.numpy_helper.from_array(
                    np.array(0.1, dtype=np.float32), name="output_scale"
                ),
            ],
            nodes=[
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["input", "input_scale", "input_zero_point"],
                    outputs=["input_q"],
                    output_dtype=onnx.TensorProto.UINT8,
                    name="input_q",
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["input_q", "input_scale", "input_zero_point"],
                    outputs=["input_qdq"],
                    name="input_dq",
                ),
                onnx.helper.make_node(
                    "Relu",
                    inputs=["input_qdq"],
                    outputs=["Relu_output_0"],
                    name="relu",
                ),
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["Relu_output_0", "output_scale"],
                    outputs=["output_q"],
                    output_dtype=onnx.TensorProto.UINT8,
                    name="output_q",
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["output_q", "output_scale"],
                    outputs=["output"],
                    name="output_dq",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)

    """
    When: Create sim from onnx QDQ and re-export to QDQ
    Then: Re-exported QDQ model should produce same output as the original model
    """
    model_2 = QuantizationSimModel._from_onnx_qdq(copy.deepcopy(model)).to_onnx_qdq()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options)
    sess_2 = ort.InferenceSession(
        model_2.SerializeToString(), sess_options=sess_options
    )
    input = {"input": np.random.randn(10, 10).astype(np.float32)}
    assert np.equal(sess.run(None, input), sess_2.run(None, input)).all()


def test_from_onnx_qdq_split_op():
    model = model_with_split_matmul()
    sim = QuantizationSimModel(
        model,
        param_type="int8",
        activation_type="int8",
        config_file="htp_v81",
    )
    sim.compute_encodings([make_dummy_input(model)])

    """
    When: Create sim from onnx QDQ model
    Then: The new sim should be in same state as the original sim
    """
    sim_2 = QuantizationSimModel._from_onnx_qdq(
        sim.to_onnx_qdq(),
        config_file="htp_v81",
    )
    _assert_sim_equal(sim, sim_2)


@pytest.mark.parametrize(
    "activation_type",
    [
        aimet_onnx.int8,
        # NOTE: INT16 activation will trigger up-conversion to opset 21
        # during export because INT16 QDQ was only added in opset 21.
        # However, currently this will fail because onnx version converter
        # has a bug with large models. This bug is expected to be fixed in onnx 1.19.
        # TODO (kyunggeu): Uncomment this when onnx 1.19 is released
        # aimet_onnx.int16,
    ],
)
def test_to_onnx_qdq_large_model(
    activation_type,
    tmp_path: pathlib.Path,
):
    seed = 200
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        model = torch.nn.Sequential(
            torch.nn.Linear(2**14, 2**14, bias=False),  # 0.25B parameters = 1GB
            torch.nn.Linear(2**14, 2**14, bias=False),  # 0.25B parameters = 1GB
        )
        torch.onnx.export(
            model,
            torch.zeros(1, 2**14),
            path,
            input_names=["input"],
            output_names=["output"],
        )

        """
        Given: Model that exceeds 2GB
        """
        model = onnx.load_model(path, load_external_data=True)
        sim = QuantizationSimModel(
            model,
            param_type=aimet_onnx.int8,
            activation_type=activation_type,
        )
        input = np.random.randn(1, 2**14).astype(np.float32)
        sim.compute_encodings([{"input": input}])

        for constants_flag in [True, False]:
            """
            When: Export to onnx QDQ with save_as_external_data=True
            Then: Output of the pure onnx model should be equal to that of sim.session
            """
            _ = sim._to_onnx_qdq(
                f=tmpdir + "/model_qdq.onnx",
                save_as_external_data=True,
                prequantize_constants=constants_flag,
            )

            qdq_model = onnx.load_model(
                tmpdir + "/model_qdq.onnx", load_external_data=True
            )
            count = sum(
                1 for node in qdq_model.graph.node if node.op_type == "DequantizeLinear"
            )
            print("Number of dequantizeLinear nodes: ", count)

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            sess = ort.InferenceSession(
                tmpdir + "/model_qdq.onnx",
                sess_options=sess_options,
            )

            (out_onnx_qdq,) = sess.run(None, {"input": input})
            (out_sim,) = sim.session.run(None, {"input": input})
            assert np.allclose(out_sim, out_onnx_qdq, atol=1e-7)
