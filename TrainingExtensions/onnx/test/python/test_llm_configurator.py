# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import json
import platform
import sys
from unittest import mock

import pytest
import torch

from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64(
    "transformers and onnx_sim is not available on Windows ARM64"
)

from aimet_onnx.utils import make_dummy_input

from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

from aimet_onnx.experimental.llm_configurator import llm_configurator
from aimet_onnx.experimental.llm_configurator.llm_configurator import (
    _apply_int8_kv_cache_tying_and_lm_head,
    _collect_all_projections,
    _set_matmul_second_input_to_8b,
    _get_quantizer_no_split_slice,
    _tie_quantizers_for_kv_cache,
    configure_llm,
)
from aimet_onnx.experimental.llm_topology import analyze_llm_topology
from aimet_onnx.defs import QSpec
import aimet_onnx

import onnx
import os

from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.qc_quantize_op import QcQuantizeOp

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from transformers.cache_utils import DynamicCache

from .models import models_for_tests, style_decoders, transformer_blocks
from .models.style_decoders import STRUCTURAL_MODEL_TYPE

from aimet_onnx.quantsim import QuantizationSimModel


_NUM_LAYERS = 2


def _kv_cache_io_names(num_layers: int = _NUM_LAYERS) -> list[tuple[str, str]]:
    """Returns the (input, output) kv-cache tensor name pairs of the decoder fixtures."""
    return [
        (f"past_{kind}_{layer}_in", f"past_{kind}_{layer}_out")
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]


def _param_precision(sim: QuantizationSimModel, node_name: str):
    """Returns the precision of the weight quantizer of ``node_name``."""
    op = sim.connected_graph.get_all_ops()[node_name]
    _, _, param_quantizers = sim.get_op_quantizers(op)
    return param_quantizers["weight"].precision()


def _all_param_precisions(sim: QuantizationSimModel) -> dict:
    """Returns the precision of every param quantizer, keyed by (node name, param name)."""
    precisions = {}
    for node_name, op in sim.connected_graph.get_all_ops().items():
        _, _, param_quantizers = sim.get_op_quantizers(op)
        for param_name, quantizer in param_quantizers.items():
            precisions[(node_name, param_name)] = quantizer.precision()

    return precisions


def _get_enabled_quantizer_name(quant_sim, tensor_name: str) -> QcQuantizeOp:
    """
    Returns closest enabled quantizer to tensor traversing upwards only through invariant ops

    :param tensor_name: Name of tensor for which to find quantizer
    """
    quantizer = quant_sim.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return tensor_name

    prod_dict = quant_sim.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product == None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
            )
        else:
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
            )

    producer = product.producer

    if producer == None:
        return None

    if not (_is_grid_preserving_op(producer.type)):
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_enabled_quantizer_name(quant_sim, upstream_tensor.name)


def check_config(
    quant_sim: QuantizationSimModel,
    encodings_path: str,
    kv_io_map: dict,
    lm_head_tensor_name: str,
    bw: int,
    is_sym: bool,
    dtype: str,
):
    with open(encodings_path, "r") as f:
        contents = json.load(f)

    activations = contents["activation_encodings"]
    params = contents["param_encodings"]

    for input, output in kv_io_map.items():
        kv_io_map[input] = _get_enabled_quantizer_name(quant_sim, output)

    names = set(list(kv_io_map.keys()) + list(kv_io_map.values()))

    quantizer_map = dict()

    assert len(activations) != 0, f"Activation Encodings are empty!"

    for act in activations:
        if act["name"] in names:
            assert act["bw"] == bw, (
                f"{act['name']} does not have bit width {bw}, has {act['bw']}!"
            )
            assert act["is_sym"] == is_sym, (
                f"{act['name']} does not have symmetry {is_sym}!"
            )
            assert act["dtype"] == dtype, (
                f"{act['name']} does not have data type {dtype}!"
            )

            quantizer_map[act["name"]] = (act["offset"], act["scale"])

    for param in params:
        if param["name"] == lm_head_tensor_name:
            assert param["bw"] == bw, (
                f"LM head {param['name']} does not have bit width {bw}!"
            )

    for input, output in kv_io_map.items():
        assert quantizer_map[input] == quantizer_map[output], (
            f"{input} and {output} quantizers are not tied!"
        )


class ExportableBase(torch.nn.Module):
    N_KEY_VALUE_HEADS = 32

    def base_forward(
        self,
        model_forward,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        kv_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            k_split = [k[i : i + 1] for i in range(32)]
            v_split = [v[i : i + 1] for i in range(self.N_KEY_VALUE_HEADS)]
            k = torch.cat(k_split, axis=1).permute(0, 1, 3, 2)
            v = torch.cat(v_split, axis=1)

            kv_cache.update(k, v, layer_idx, {})  # pyright: ignore [reportArgumentType]

        out = model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
        )

        new_past_key_values = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(new_past_key_values)):
            if hasattr(new_past_key_values, "value_cache"):
                keys = new_past_key_values.key_cache[layer]
                values = new_past_key_values.value_cache[layer]
            elif hasattr(new_past_key_values.layers[layer], "keys"):
                keys = new_past_key_values.layers[layer].keys
                values = new_past_key_values.layers[layer].values
            else:
                keys = new_past_key_values.layers[layer][0]
                values = new_past_key_values.layers[layer][1]
            flat_output_past_key_values += [keys, values]

        return [out["logits"]] + flat_output_past_key_values

    def get_output_names(self, num_hidden_layers: int):
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def get_input_names(self, num_hidden_layers: int):
        output_names = ["input_ids", "attention_mask"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_in")
            output_names.append(f"past_value_{layer}_in")
        return output_names


class ExportableLlama(LlamaForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


class ExportableQwen(Qwen2ForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


class ExportablePhi(Phi3ForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


def apply_to_model(model_id, tmp_path):
    # onnxsim is not available on Windows ARM64
    # Lazy import to avoid import errors on unsupported platforms
    from onnxsim import simplify

    vocab_size = 8
    num_hidden_layers = 2
    hidden_size = 64
    num_attention_heads = 32
    num_key_value_heads = 32
    embed_dim = hidden_size // num_attention_heads // 2
    intermediate_size = 2
    sequence_length = 16
    context_length = 32

    if model_id == "llama":
        llm_config = LlamaConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )

        model = ExportableLlama(config=llm_config)

    elif model_id == "qwen":
        llm_config = Qwen2Config(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )

        model = ExportableQwen(config=llm_config)

    elif model_id == "phi":
        llm_config = Phi3Config(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            pad_token_id=4,
        )

        model = ExportablePhi(config=llm_config)

    checkpoint = tmp_path / str(model_id)
    checkpoint.mkdir()

    onnx_model_path = os.path.join(checkpoint, f"model_cl{context_length}.onnx")

    dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int32)
    dummy_attention_mask = torch.ones(
        (1, 1, sequence_length, context_length), dtype=torch.float32
    )

    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = torch.zeros(
            (num_key_value_heads, 1, embed_dim * 2, context_length - sequence_length),
            dtype=torch.float32,
        )
        past_value = torch.zeros(
            (num_key_value_heads, 1, context_length - sequence_length, embed_dim * 2),
            dtype=torch.float32,
        )
        past_key_values.append(past_key)
        past_key_values.append(past_value)

    example_input = [dummy_input_ids, dummy_attention_mask] + past_key_values

    with torch.no_grad():
        torch.onnx.export(
            model.eval(),
            tuple(example_input),
            onnx_model_path,
            input_names=model.get_input_names(2),
            output_names=model.get_output_names(2),
            opset_version=17,
            dynamo=False,
        )

        onnx_model = onnx.load(onnx_model_path)
        onnx_model, _ = simplify(onnx_model)

    model_name = f"{model_id}_2HL_simplified"

    bw = 8
    is_sym = True
    dtype = "INT"

    kv_io_map = {
        "past_key_0_in": "past_key_0_out",
        "past_key_1_in": "past_key_1_out",
        "past_value_0_in": "past_value_0_out",
        "past_value_1_in": "past_value_1_out",
    }

    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if host_device.type == "cuda" and host_device.index is not None:
        providers = [
            ("CUDAExecutionProvider", {"device_id": host_device.index}),
            "CPUExecutionProvider",
        ]
    elif host_device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    quant_sim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=16,
        default_param_bw=4,
        config_file="htp_v73",
        providers=providers,
    )

    lm_head_tensor_name = None
    for weight in quant_sim.model.model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            dimensions = list(weight.dims)
            if dimensions[-1] == vocab_size:
                lm_head_tensor_name = weight.name

    configured_quant_sim = _apply_int8_kv_cache_tying_and_lm_head(
        quant_sim, kv_io_map, lm_head_tensor_name
    )

    configured_quant_sim.compute_encodings(
        lambda session: session.run(
            None, make_dummy_input(configured_quant_sim.model.model)
        )
    )

    export_dir = checkpoint / f"configured_{model_name}"
    export_dir.mkdir(exist_ok=True)

    configured_quant_sim.export(str(export_dir), f"{model_name}_model")

    encodings_path = export_dir / f"{model_name}_model.encodings"
    check_config(
        configured_quant_sim,
        encodings_path,
        kv_io_map,
        lm_head_tensor_name,
        bw,
        is_sym,
        dtype,
    )


class TestLLMConfigurator:
    """Tests for applying quantsim configuration for LLMs"""

    def test_llm_configurator(self, tmp_path):
        apply_to_model("llama", tmp_path)

        apply_to_model("qwen", tmp_path)

        apply_to_model("phi", tmp_path)

    def test_set_matmul_second_input_to_8b(self):
        model = models_for_tests.model_with_split_matmul()
        sim = QuantizationSimModel(model)

        quantizer = _get_quantizer_no_split_slice(sim, "reshape_output")

        _set_matmul_second_input_to_8b(sim)

        quantizer = sim.qc_quantize_op_dict["reshape_output"]
        assert quantizer.enabled == True
        assert quantizer.bitwidth == 8

    @pytest.mark.parametrize(
        "block_builder",
        [
            transformer_blocks.sha_2_head_block,
            transformer_blocks.sha_2_head_block_native_kvcache,
        ],
    )
    def test_sha_kv_cache_tying(self, block_builder):
        model = block_builder()
        sim = QuantizationSimModel(model, activation_type="int16", param_type="int4")
        kv_io_map = {"past_key_in": "past_key_out", "past_value_in": "past_value_out"}
        _tie_quantizers_for_kv_cache(sim, kv_io_map)

        tied_key_cache_quantizers = {
            name
            for name, q in sim.qc_quantize_op_dict.items()
            if q is sim.qc_quantize_op_dict["past_key_out"]
        }
        tied_value_cache_quantizers = {
            name
            for name, q in sim.qc_quantize_op_dict.items()
            if q is sim.qc_quantize_op_dict["past_value_out"]
        }

        assert "past_key_in" in tied_key_cache_quantizers
        assert "past_value_in" in tied_value_cache_quantizers

        enabled_quantizer_names = set(
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        )

        expected_tied_key_quantizers = {
            name
            for name in enabled_quantizer_names
            if name.startswith("total_key")
            or name.startswith("k_proj_emb")
            or name.startswith("past_key")
        }
        expected_tied_value_quantizers = {
            name
            for name in enabled_quantizer_names
            if name.startswith("total_value")
            or name.startswith("v_proj")
            and "weight" not in name
            or name.startswith("past_value")
        }

        # Additional tensors may be tied by normal concat tying
        assert expected_tied_key_quantizers.issubset(tied_key_cache_quantizers)
        assert expected_tied_value_quantizers.issubset(tied_value_cache_quantizers)

        expected_not_tied_key_quantizers = {
            name
            for name in enabled_quantizer_names
            if name.startswith("q_proj")
            or name.startswith("self_attn_out")
            or name.startswith("v_proj")
            or name.startswith("hidden")
            or name.startswith("k_proj_norm")
            or name.startswith("key_scaled")
        }
        expected_not_tied_value_quantizers = {
            name
            for name in enabled_quantizer_names
            if name.startswith("q_proj")
            or name.startswith("self_attn_out")
            or name.startswith("k_proj")
            or name.startswith("hidden")
        }
        assert not expected_not_tied_key_quantizers.intersection(
            tied_key_cache_quantizers
        )
        assert not expected_not_tied_value_quantizers.intersection(
            tied_value_cache_quantizers
        )


@pytest.fixture(scope="module")
def decoder_model():
    """Decoder stack with paired ``past_{key,value}_{layer}_{in,out}`` kv-cache I/O."""
    return transformer_blocks.sha_gqa_decoder(num_layers=_NUM_LAYERS)


@pytest.fixture
def sim(decoder_model):
    return QuantizationSimModel(decoder_model)


#: Cheap stand-in for a real Qwen3 config, matching the dimensions the spinquant
#: tests use to exercise the headless/embedding-less export variants.
_QWEN3_SMALL = dict(
    num_hidden_layers=_NUM_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=128,
    vocab_size=16,
    hidden_size=64,
)


@pytest.fixture(scope="module")
def qwen3_models():
    """Qwen3 causal LM exports keyed by ``(with_lm_head, with_embedding)``."""
    return {
        (with_lm_head, with_embedding): transformer_blocks.qwen3_causal_lm(
            with_lm_head=with_lm_head, with_embedding=with_embedding, **_QWEN3_SMALL
        )
        for with_lm_head in (True, False)
        for with_embedding in (True, False)
    }


class TestConfigureLlm:
    """Tests for :func:`configure_llm`"""

    def test_ties_kv_cache_quantizers(self, sim):
        """Each kv-cache input shares one quantizer with the output of the same layer."""
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology)

        for input_name, output_name in _kv_cache_io_names():
            # The output quantizer sits upstream of the graph output, so it is
            # reached through the enabled-quantizer walk rather than by name.
            assert sim.qc_quantize_op_dict[input_name] is sim._get_enabled_quantizer(
                output_name
            )

    def test_kv_cache_quantizers_are_tied_per_layer(self, sim):
        """Tying does not merge separate caches into a single quantizer."""
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology)

        # No KV input quantizers are tied to each other
        input_quantizers = [
            sim.qc_quantize_op_dict[input_name]
            for input_name, _ in _kv_cache_io_names()
        ]
        assert len({id(quantizer) for quantizer in input_quantizers}) == len(
            input_quantizers
        )

        # No KV output quantizers are tied to each other
        output_quantizers = [
            sim.qc_quantize_op_dict[output_name]
            for _, output_name in _kv_cache_io_names()
        ]
        assert len({id(quantizer) for quantizer in output_quantizers}) == len(
            output_quantizers
        )

    @pytest.mark.parametrize(
        "precision", [aimet_onnx.int8, aimet_onnx.int16, "int8", "int16"]
    )
    def test_sets_kv_cache_precision(self, sim, precision):
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology, kv_cache_type=precision)

        expected = (
            aimet_onnx.qtype.from_string(precision)
            if isinstance(precision, str)
            else precision
        )
        for input_name, output_name in _kv_cache_io_names():
            assert sim.qc_quantize_op_dict[input_name].precision() == expected
            assert sim._get_enabled_quantizer(output_name).precision() == expected

    def test_sets_projection_weight_precision(self, sim):
        """Every projection of every block is reprecisioned, lm head is not."""
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology, projection_weight_type=aimet_onnx.int4)

        projections = _collect_all_projections(topology)
        assert len(projections) == 24  # 8 qkv + o + gate + up + down, per block
        for node_name in projections:
            assert _param_precision(sim, node_name) == aimet_onnx.int4

        # lm_head is not a block projection, so it keeps the sim default.
        (lm_head,) = topology.lm_head
        assert _param_precision(sim, lm_head) == aimet_onnx.int8

    def test_sets_lm_head_weight_precision(self, sim):
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology, lm_head_weight_type=aimet_onnx.int4)

        (lm_head,) = topology.lm_head
        assert _param_precision(sim, lm_head) == aimet_onnx.int4
        # Block projections are untouched by an lm-head-only call.
        for node_name in _collect_all_projections(topology):
            assert _param_precision(sim, node_name) == aimet_onnx.int8

    @pytest.mark.parametrize(
        "decoder_cls, expected_projections",
        [
            # Unfused q/k/v and gate/up: 3 qkv + o + gate + up + down, per block.
            pytest.param(style_decoders.LlamaStyleDecoder, 14, id="unfused"),
            # Fused qkv_proj/gate_up_proj collapse into one node each
            pytest.param(style_decoders.Phi3StyleDecoder, 8, id="fused"),
        ],
    )
    def test_sets_projection_weight_precision_of_fused_projections(
        self, decoder_cls, expected_projections
    ):
        """Fused read projections are reprecisioned, not silently skipped."""
        model = style_decoders._export_decoder_with_ids(
            decoder_cls(), add_value_input=False
        )
        sim = QuantizationSimModel(model)
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        configure_llm(sim, topology, projection_weight_type=aimet_onnx.int4)

        projections = _collect_all_projections(topology)
        assert len(projections) == expected_projections
        for node_name in projections:
            assert _param_precision(sim, node_name) == aimet_onnx.int4

    def test_accepts_qspec_weight_type(self, sim):
        """A QSpec configures granularity, not just bitwidth."""
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)
        spec = QSpec.lpbq(aimet_onnx.int4, block_size=8)

        configure_llm(sim, topology, projection_weight_type=spec)

        for node_name in _collect_all_projections(topology):
            op = sim.connected_graph.get_all_ops()[node_name]
            _, _, param_quantizers = sim.get_op_quantizers(op)
            quantizer = param_quantizers["weight"]
            assert quantizer.precision() == aimet_onnx.int4
            assert quantizer.quant_info.blockSize == 8
            assert quantizer.quant_info.usePerChannelMode

    def test_leaves_precisions_unchanged_when_no_type_given(self, sim):
        """With no precision argument the call only ties quantizers."""
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)
        before = {
            name: quantizer.precision()
            for name, quantizer in sim.qc_quantize_op_dict.items()
        }

        configure_llm(sim, topology)

        after = {
            name: quantizer.precision()
            for name, quantizer in sim.qc_quantize_op_dict.items()
        }
        assert before == after

    def test_raises_on_unpaired_kv_cache_names(self, sim):
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)
        topology.past_key_output_names.pop()

        with pytest.raises(RuntimeError, match="cache inputs and outputs"):
            configure_llm(sim, topology)

        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)
        topology.past_value_output_names.pop()

        with pytest.raises(RuntimeError, match="cache inputs and outputs"):
            configure_llm(sim, topology)

    @pytest.mark.parametrize("with_lm_head", [True, False])
    @pytest.mark.parametrize("with_embedding", [True, False])
    def test_configures_headless_and_embeddingless_models(
        self, qwen3_models, with_lm_head, with_embedding
    ):
        """
        Exports without an lm head and/or without embed_tokens configure the same.
        """
        sim = QuantizationSimModel(qwen3_models[(with_lm_head, with_embedding)])
        topology = analyze_llm_topology(sim.model.model, "qwen3")
        assert bool(topology.lm_head) == with_lm_head
        assert bool(topology.embed_tokens) == with_embedding

        configure_llm(
            sim,
            topology,
            kv_cache_type=aimet_onnx.int16,
            projection_weight_type=aimet_onnx.int4,
        )

        for input_name, output_name in _kv_cache_io_names():
            quantizer = sim.qc_quantize_op_dict[input_name]
            assert quantizer is sim._get_enabled_quantizer(output_name)
            assert quantizer.precision() == aimet_onnx.int16

        projections = _collect_all_projections(topology)
        assert len(projections) == 7 * _NUM_LAYERS
        for node_name in projections:
            assert _param_precision(sim, node_name) == aimet_onnx.int4

    def test_ignores_lm_head_type_when_model_has_no_lm_head(self, qwen3_models):
        """``lm_head_weight_type`` on a headless model reprecisions nothing."""
        sim = QuantizationSimModel(qwen3_models[(False, True)])
        topology = analyze_llm_topology(sim.model.model, "qwen3")
        before = _all_param_precisions(sim)

        configure_llm(sim, topology, lm_head_weight_type=aimet_onnx.int4)

        assert _all_param_precisions(sim) == before

    @pytest.mark.parametrize(
        "config_file, expected_kv_precision, expect_warning",
        [
            # V69 does not support int16 dynamic matmul inputs, so the exception rules
            # force the requested kv-cache precision back down to int8.
            ("htp_v69", aimet_onnx.int8, True),
            # V73 honors the request, and instead promotes the untouched matmul inputs
            # feeding the kv-cache to int16. That is not an override of the request.
            ("htp_v73", aimet_onnx.int16, False),
        ],
    )
    def test_warns_only_when_backend_overrides_requested_precision(
        self, decoder_model, config_file, expected_kv_precision, expect_warning
    ):
        """Precisions changed by the exception rules only warn if they were requested here."""
        sim = QuantizationSimModel(decoder_model, config_file=config_file)
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        with mock.patch.object(llm_configurator.logger, "warning") as mock_warning:
            configure_llm(
                sim,
                topology,
                kv_cache_type=aimet_onnx.int16,
                projection_weight_type=aimet_onnx.int4,
            )

        for input_name, output_name in _kv_cache_io_names():
            assert sim.qc_quantize_op_dict[input_name].precision() == (
                expected_kv_precision
            )
            assert sim._get_enabled_quantizer(output_name).precision() == (
                expected_kv_precision
            )

        assert mock_warning.called == expect_warning
        if expect_warning:
            message = mock_warning.call_args[0][0] % tuple(
                mock_warning.call_args[0][1:]
            )
            assert "int16 -> int8" in message
            assert "past_key_0_in" in message

    def test_does_not_warn_when_no_type_given(self, decoder_model):
        """Configuring nothing cannot override anything, even on a constrained backend."""
        sim = QuantizationSimModel(decoder_model, config_file="htp_v69")
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        with mock.patch.object(llm_configurator.logger, "warning") as mock_warning:
            configure_llm(sim, topology)

        assert not mock_warning.called

    def test_raises_on_analyzed_topology_with_unpaired_kv_cache_names(self):
        """A decoder with a kv-cache input but no matching output is rejected.

        ``LlamaStyleDecoder`` is exported with a dangling ``past_value_0`` input and
        no kv-cache outputs, so the analyzed topology cannot be paired.
        """
        model = style_decoders._export_decoder_with_ids(
            style_decoders.LlamaStyleDecoder()
        )
        sim = QuantizationSimModel(model)
        topology = analyze_llm_topology(sim.model.model, STRUCTURAL_MODEL_TYPE)

        with pytest.raises(RuntimeError, match="value cache inputs and outputs"):
            configure_llm(sim, topology)
