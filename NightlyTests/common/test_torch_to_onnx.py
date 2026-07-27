# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import json
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch

from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64("transformers is not available on Windows ARM64")

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

import aimet_onnx
from aimet_onnx.utils import make_dummy_input
import aimet_torch
import aimet_torch.v2.nn.transformers


@pytest.fixture
def tmp_dir():
    """
    Pytest fixture to create and yield a temporary directory.
    The directory is automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def add_genai_tests_path(monkeypatch):
    """
    Pytest fixture to add the GenAILab directory to sys.path.
    """
    path = os.path.abspath(os.path.join(Path(__file__).parent, "../../../../"))
    monkeypatch.syspath_prepend(path)


@pytest.mark.parametrize(
    "model_cls, config_cls",
    [
        (Gemma3ForCausalLM, Gemma3TextConfig),
        (LlamaForCausalLM, LlamaConfig),
        (MistralForCausalLM, MistralConfig),
        (Phi3ForCausalLM, Phi3Config),
        (Qwen2ForCausalLM, Qwen2Config),
        (Qwen3ForCausalLM, Qwen3Config),
    ],
)
def test_hf_torch_to_onnx_workflow(
    tmp_dir, add_genai_tests_path, model_cls, config_cls
):
    """
    Given: HF model quantized / exported as onnx QDQ from aimet-torch
    When: Import onnx QDQ into aimet-onnx
    Then: aimet-onnx sim should produce same output as aimet-torch sim
    """

    from GenAILab.qai_hub_lm.models.utils.exportable import (
        ONNXExportableModuleWithCache,
    )

    config = config_cls(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=32,
        num_hidden_layers=1,
        pad_token_id=999,
    )
    model = ONNXExportableModuleWithCache(model_cls(config), input_names=("input_ids",))
    input_ids = torch.randint(0, config.vocab_size, (1, 128))

    torch_sim = aimet_torch.QuantizationSimModel(
        model,
        dummy_input=input_ids,
        config_file="htp_quantsim_config_v81_per_channel_linear.json",
        in_place=True,
    )
    torch_sim.compute_encodings(lambda model: model(input_ids))
    onnx_qdq_model_path = os.path.join(tmp_dir, "model_qdq.onnx")
    aimet_torch.onnx.export(
        torch_sim.model,
        input_ids,
        onnx_qdq_model_path,
        opset_version=21,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_sim = aimet_onnx.QuantizationSimModel.from_onnx_qdq(
        onnx.load(onnx_qdq_model_path),
        config_file="htp_quantsim_config_v81_per_channel_linear.json",
        strict=True,
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    onnx_qdq_sess = ort.InferenceSession(onnx_qdq_model_path, sess_options=sess_options)

    # Allow off-by-three due to inevitable numerical errors
    atol = 3 * torch_sim.model.model.lm_head.output_quantizers[0].get_scale().item()

    for i in range(10):
        np.random.seed(i)
        input = make_dummy_input(onnx_sim.model.model)
        onnx_sim_out, *_ = onnx_sim.session.run(None, input)
        onnx_qdq_out, *_ = onnx_qdq_sess.run(None, input)
        assert np.allclose(onnx_sim_out, onnx_qdq_out, atol=atol)


@pytest.mark.parametrize("encoding_version", ["1.0.0", "2.0.0"])
def test_torch_to_onnx_zero_point_shift(tmp_dir, encoding_version):
    """
    Given: aimet_torch quantized model with shifted weight zero-points
    When: Export to onnx using aimet-torch and import into aimet-onnx
    Then: aimet-onnx sim should produce same output and exported encodings as aimet-torch sim
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
    )
    input_tensor = torch.randn(1, 128)
    torch_sim = aimet_torch.QuantizationSimModel(
        model,
        dummy_input=input_tensor,
        config_file="htp_v81",
        in_place=True,
    )

    # Set param quantizers to 2-bit with zero-point shift of 0.5
    for layer in torch_sim.qmodules():
        if isinstance(layer, torch.nn.Linear):
            quantizer = aimet_torch.quantization.affine.QuantizeDequantize(
                shape=(layer.out_features, 1),
                bitwidth=2,
                symmetric=True,
            )
            quantizer.zero_point_shift = 0.5
            layer.param_quantizers["weight"] = quantizer

    torch_sim.compute_encodings(lambda model: model(input_tensor))
    torch_output = torch_sim.model(input_tensor).detach().numpy()
    torch_export_dir = os.path.join(tmp_dir, "torch_export")
    os.makedirs(torch_export_dir, exist_ok=True)

    torch_export_path = os.path.join(torch_export_dir, "model.onnx")
    encoding_path = os.path.join(torch_export_dir, "model.encodings")
    torch_sim.onnx.export(
        input_tensor,
        torch_export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        dynamo=False,
        encoding_version=encoding_version,
        export_int32_bias=False,
    )

    with open(encoding_path, "r") as f:
        torch_encodings = json.load(f)

    onnx_sim = aimet_onnx.QuantizationSimModel(onnx.load(torch_export_path))
    aimet_onnx.quantsim.load_encodings_to_sim(
        onnx_sim, encoding_path, strict=False, disable_missing_quantizers=True
    )
    onnx_output = onnx_sim.session.run(None, {"input": input_tensor.numpy()})[0]

    for name in onnx_sim.param_names:
        if onnx_sim.qc_quantize_op_dict[name].enabled:
            assert (
                onnx_sim.qc_quantize_op_dict[name]._tensor_quantizer.getZeroPointShift()
                == 0.5
            )
            assert onnx_sim.qc_quantize_op_dict[name].bitwidth == 2

    assert np.allclose(torch_output, onnx_output)

    onnx_export_path = os.path.join(tmp_dir, "onnx_export")
    onnx_encoding_path = os.path.join(onnx_export_path, "model.encodings")
    os.makedirs(onnx_export_path, exist_ok=True)
    onnx_sim.export(
        onnx_export_path,
        "model",
        encoding_version=encoding_version,
        export_int32_bias=False,
    )

    with open(onnx_encoding_path, "r") as f:
        onnx_encodings = json.load(f)

    if encoding_version == "1.0.0":
        onnx_enc = (
            onnx_encodings["param_encodings"] + onnx_encodings["activation_encodings"]
        )
        torch_enc = (
            torch_encodings["param_encodings"] + torch_encodings["activation_encodings"]
        )
    else:
        onnx_enc = onnx_encodings["encodings"]
        torch_enc = torch_encodings["encodings"]

    assert sorted(onnx_enc, key=lambda x: x["name"]) == sorted(
        torch_enc, key=lambda x: x["name"]
    )


def test_transpose_mm_lpbq(tmp_dir):
    """
    Given: QDQ Model with weight -> QDQ (lpbq) -> Transpose -> MatMul sequence
    """
    from aimet_torch.v2.quantsim.config_utils import (
        set_grouped_blockwise_quantization_for_weights,
    )

    model = torch.nn.Sequential(torch.nn.Linear(64, 64))
    x = torch.randn(3, 64, 64)
    torch_sim = aimet_torch.QuantizationSimModel(model, x)
    set_grouped_blockwise_quantization_for_weights(
        torch_sim,
        [torch.nn.Linear],
        bitwidth=4,
        symmetric=True,
        decompressed_bw=8,
        block_size=8,
        block_grouping=8,
    )
    torch_sim.compute_encodings(lambda model: model(x))
    torch_out = torch_sim.model(x).detach().numpy()
    torch_sim.onnx.export(
        x,
        os.path.join(tmp_dir, "transpose_mm.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        encoding_version="2.0.0",
    )

    aimet_torch.onnx.export(
        torch_sim.model,
        x,
        os.path.join(tmp_dir, "transpose_mm_qdq.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=False,
    )

    """
    When: Load encodings to aimet-onnx sim
    Then: Encodings should be loaded normally and outputs should match
    """
    onnx_sim = aimet_onnx.QuantizationSimModel(
        onnx.load(os.path.join(tmp_dir, "transpose_mm.onnx"))
    )
    aimet_onnx.quantsim.load_encodings_to_sim(
        onnx_sim, os.path.join(tmp_dir, "transpose_mm.encodings"), strict=False
    )
    (onnx_out,) = onnx_sim.session.run(None, {"input": x.numpy()})
    assert np.allclose(
        torch_out,
        onnx_out,
        atol=torch_sim.model[0].output_quantizers[0].get_scale().item(),
    )

    onnx_sim = aimet_onnx.QuantizationSimModel.from_onnx_qdq(
        onnx.load(os.path.join(tmp_dir, "transpose_mm_qdq.onnx")), strict=True
    )
    (onnx_out,) = onnx_sim.session.run(None, {"input": x.numpy()})
    assert np.allclose(
        torch_out,
        onnx_out,
        atol=torch_sim.model[0].output_quantizers[0].get_scale().item(),
    )
