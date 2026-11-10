# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the blockwise sampler"""

import copy
import os
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnx
import onnx_ir
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from aimet_onnx import QuantizationSimModel, blockwise_sampler
from aimet_onnx.adascale import adascale_optimizer
from aimet_onnx.adascale.activation_sampler import ActivationSampler
from aimet_onnx.adascale.adascale_optimizer import AdaScale
from aimet_onnx.adascale.model_converter import required_block_inputs
from aimet_onnx.blockwise_sampler import BlockwiseSampler, SamplingPlan
from aimet_onnx.common.onnx._utils import _iterate_graphs_recursive_ir
from aimet_onnx.ir_utils import walk_back

from .models.style_decoders import (
    _export_decoder_with_ids,
    _export_gemma4_shared_kv_decoder,
    Gemma4StyleDecoder,
    LlamaStyleDecoder,
)

_S, _D = 3, 4
_CPU = ["CPUExecutionProvider"]


def _toy_model(
    num_blocks=4,
    *,
    glue=False,
    skip_read=None,
    scan_block=None,
    shared_weight=False,
    if_block=None,
    dtype=TensorProto.FLOAT,
):
    """Residual chain ``start_k -> end_k = start_k + start_k @ W_k`` with optional extras.

    :param glue: Put ``start_{k+1} = end_k * g_k`` between blocks instead of
        ``start_{k+1} = end_k``.
    :param skip_read: ``(src, dst)``: block ``dst`` also adds block ``src``'s input.
    :param scan_block: Block whose output also adds a Scan over its MatMul whose body reads
        ``start_0`` and the graph input ``side`` from the enclosing graph.
    :param shared_weight: Blocks 0 and 1 share one weight.
    :param if_block: Block whose MatMul runs in the then_branch of an If on the graph input
        ``cond``, with its weight declared in that branch.
    :return: ``(model, block_end_points)``
    """
    # helper.tensor_dtype_to_np_dtype maps BFLOAT16 to float32
    np_dtype = onnx_ir.DataType(dtype).numpy()

    def _init(array, name):
        return onnx_ir.serde.serialize_tensor(onnx_ir.Tensor(array, name=name))

    rng = np.random.default_rng(0)
    nodes = [helper.make_node("Relu", ["x"], ["start_0"])]
    inits, end_points = [], []
    graph_inputs = [helper.make_tensor_value_info("x", dtype, [_S, _D])]
    for k in range(num_blocks):
        start = f"start_{k}"
        weight = "W_0" if shared_weight and k == 1 else f"W_{k}"
        weight_init = None
        if weight not in {init.name for init in inits}:
            weight_init = _init(
                (rng.standard_normal((_D, _D)) * 0.5).astype(np_dtype), weight
            )
        if if_block == k:
            nodes.append(
                helper.make_node(
                    "If",
                    ["cond"],
                    [f"m_{k}"],
                    then_branch=helper.make_graph(
                        [helper.make_node("MatMul", [start, weight], ["then_m"])],
                        "then",
                        [],
                        [helper.make_tensor_value_info("then_m", dtype, [_S, _D])],
                        [weight_init],
                    ),
                    else_branch=helper.make_graph(
                        [helper.make_node("Identity", [start], ["else_m"])],
                        "else",
                        [],
                        [helper.make_tensor_value_info("else_m", dtype, [_S, _D])],
                    ),
                )
            )
            graph_inputs.append(
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
            )
        else:
            if weight_init is not None:
                inits.append(weight_init)
            nodes.append(helper.make_node("MatMul", [start, weight], [f"m_{k}"]))
        nodes.append(helper.make_node("Add", [start, f"m_{k}"], [f"a_{k}"]))
        out = f"a_{k}"
        if skip_read is not None and skip_read[1] == k:
            nodes.append(
                helper.make_node("Add", [out, f"start_{skip_read[0]}"], [f"skip_{k}"])
            )
            out = f"skip_{k}"
        if scan_block == k:
            body = helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["axes"],
                        value=helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
                    ),
                    helper.make_node(
                        "ReduceMean", ["start_0", "axes"], ["mean"], keepdims=0
                    ),
                    helper.make_node("Add", ["c_in", "s_in"], ["t"]),
                    helper.make_node("Mul", ["t", "mean"], ["t2"]),
                    helper.make_node("Mul", ["t2", "side"], ["c_out"]),
                    helper.make_node("Identity", ["c_out"], ["s_out"]),
                ],
                "body",
                [
                    helper.make_tensor_value_info("c_in", dtype, [_D]),
                    helper.make_tensor_value_info("s_in", dtype, [_D]),
                ],
                [
                    helper.make_tensor_value_info("c_out", dtype, [_D]),
                    helper.make_tensor_value_info("s_out", dtype, [_D]),
                ],
            )
            inits.append(_init(np.zeros(_D, np_dtype), "c0"))
            nodes += [
                helper.make_node(
                    "Scan",
                    ["c0", f"m_{k}"],
                    [f"carry_{k}", f"ys_{k}"],
                    body=body,
                    num_scan_inputs=1,
                ),
                helper.make_node("Add", [out, f"ys_{k}"], [f"scan_{k}"]),
            ]
            out = f"scan_{k}"
            graph_inputs.append(helper.make_tensor_value_info("side", dtype, [_D]))
        end = f"end_{k}"
        nodes.append(helper.make_node("Identity", [out], [end]))
        if glue:
            inits.append(_init(np.full(_D, 0.5, np_dtype), f"g_{k}"))
            nodes.append(helper.make_node("Mul", [end, f"g_{k}"], [f"start_{k + 1}"]))
        else:
            nodes.append(helper.make_node("Identity", [end], [f"start_{k + 1}"]))
        end_points.append((start, end))
    inits.append(_init(rng.standard_normal((_D, _D)).astype(np_dtype), "W_out"))
    nodes.append(helper.make_node("MatMul", [f"start_{num_blocks}", "W_out"], ["out"]))
    graph = helper.make_graph(
        nodes,
        "toy",
        graph_inputs,
        [helper.make_tensor_value_info("out", dtype, [_S, _D])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, end_points


def _toy_inputs(model, num_samples=3, np_dtype=np.float32):
    rng = np.random.default_rng(1)
    return [
        {
            inp.name: np.array(True)
            if inp.type.tensor_type.elem_type == TensorProto.BOOL
            else rng.standard_normal(
                [d.dim_value for d in inp.type.tensor_type.shape.dim]
            ).astype(np_dtype)
            for inp in model.graph.input
        }
        for _ in range(num_samples)
    ]


def _block_inputs(graph, end_points):
    starts = [start for start, _ in end_points]
    return [
        required_block_inputs(graph, starts[: idx + 1], [end])
        for idx, (_, end) in enumerate(end_points)
    ]


def _full_prefix(model, names, inputs, tmp_path, providers=None):
    """Reference: sample ``names`` by running the whole prefix, as today's path does."""
    path = os.path.join(tmp_path, "reference.onnx")
    onnx_ir.save(model, path, external_data="reference.data")
    sampler = ActivationSampler(names, path, providers or _CPU)
    return [sampler.sample_acts(sample) for sample in inputs]


def _assert_samples_close(actual, expected, rtol=1e-5, atol=1e-6):
    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected):
        assert len(actual_sample) == len(expected_sample)
        for a, e in zip(actual_sample, expected_sample):
            np.testing.assert_allclose(
                np.asarray(a, np.float32), np.asarray(e, np.float32), rtol, atol
            )


def _perturb(model, name, scale=1.5):
    initializer = next(
        graph.initializers[name]
        for graph in _iterate_graphs_recursive_ir(model.graph)
        if name in graph.initializers
    )
    initializer.const_value = onnx_ir.Tensor(
        (initializer.const_value.numpy() * scale + 0.1).astype(
            initializer.const_value.numpy().dtype
        )
    )


def _samplers(model_proto, end_points, inputs, providers=None, *, may_update=None):
    """A plan over the blocks of ``model_proto`` and two samplers sharing it, as AdaScale
    builds them: one for a changing copy (quantized) and one for a fixed copy (float)."""
    qt_model = onnx_ir.from_proto(model_proto)
    fp_model = onnx_ir.from_proto(model_proto)
    plan = SamplingPlan(
        qt_model.graph,
        _block_inputs(qt_model.graph, end_points),
        [[end] for _, end in end_points],
    )
    qt_sampler = BlockwiseSampler(
        qt_model,
        plan,
        inputs,
        providers or _CPU,
        may_update=may_update,
        label="quantized",
    )
    fp_sampler = BlockwiseSampler(
        fp_model, plan, inputs, providers or _CPU, label="float"
    )
    return plan, (qt_model, qt_sampler), (fp_model, fp_sampler)


def _run_and_compare(model_proto, end_points, inputs, tmp_path, *, may_update=None):
    """Sample every block with the blockwise sampler, perturbing each block's weight after it
    is sampled (simulated copy-back), and compare with full-prefix sampling of the same models."""
    plan, (qt_model, qt_sampler), (fp_model, fp_sampler) = _samplers(
        model_proto, end_points, inputs, may_update=may_update
    )
    for idx, names in enumerate(plan.block_inputs):
        _assert_samples_close(
            fp_sampler.sample_block(idx),
            _full_prefix(fp_model, names, inputs, tmp_path),
        )
        _assert_samples_close(
            qt_sampler.sample_block(idx),
            _full_prefix(qt_model, names, inputs, tmp_path),
        )
        if any(
            f"W_{idx}" in graph.initializers
            for graph in _iterate_graphs_recursive_ir(qt_model.graph)
        ):
            _perturb(qt_model, f"W_{idx}")
    return plan, qt_sampler, fp_sampler


class TestBlockwiseSampler:
    """Blockwise sampling matches full-prefix sampling of the same, changing, model."""

    @pytest.mark.parametrize("glue", [False, True])
    def test_matches_full_prefix_while_weights_change(self, tmp_path, glue):
        """
        Given: A residual chain of blocks, each block's weight changed right after it is sampled
        When: Every block's inputs are sampled blockwise
        Then: They match sampling the whole prefix of the changed model (no stale values),
              including when ops sit between one block's end and the next block's start
        """
        model, end_points = _toy_model(glue=glue)
        _run_and_compare(model, end_points, _toy_inputs(model), tmp_path)

    def test_cache_holds_only_live_boundaries(self, tmp_path):
        """
        Given: A plain residual chain
        When: Each block is sampled
        Then: The cache holds only that block's residual (needed to advance to the next), sized
              like the block's sampled inputs, and nothing after the last block
        """
        model, end_points = _toy_model()
        inputs = _toy_inputs(model)
        _, (_, qt_sampler), (_, fp_sampler) = _samplers(model, end_points, inputs)
        for idx in range(len(end_points)):
            qsim_inputs = qt_sampler.sample_block(idx)
            fp_inputs = fp_sampler.sample_block(idx)
            expected = [] if idx == len(end_points) - 1 else [f"start_{idx}"]
            for sampler, sampled in (
                (qt_sampler, qsim_inputs),
                (fp_sampler, fp_inputs),
            ):
                assert sampler._cache.names() == expected
                nbytes = sum(a.nbytes for s in sampled for a in s)
                assert sampler.cache_nbytes() == (nbytes if expected else 0)
            # Cached arrays are the ones handed out, not copies
            if expected:
                assert qt_sampler._cache.get(f"start_{idx}", 0) is qsim_inputs[0][0]

    def test_cross_block_read_is_kept_until_last_use(self, tmp_path):
        """
        Given: Block 3 also reads block 1's input (Gemma4-style shared KV)
        When: Blocks are sampled in order
        Then: Block 1's input stays cached across block 2, is evicted after block 3, and all
              inputs match full-prefix sampling
        """
        model, end_points = _toy_model(skip_read=(1, 3))
        plan, qt_sampler, fp_sampler = _run_and_compare(
            model, end_points, _toy_inputs(model), tmp_path
        )
        assert plan.last_use["start_1"] == 3
        assert not qt_sampler._cache.names()
        assert not fp_sampler._cache.names()

    def test_scan_outer_scope_reads(self, tmp_path):
        """
        Given: A block whose Scan body reads block 0's input and a graph input from the
               enclosing graph, which the Scan node does not list as inputs
        When: The block's inputs are computed and sampled blockwise
        Then: Both outer reads are block inputs, and sampling matches the full prefix
        """
        model, end_points = _toy_model(num_blocks=3, scan_block=2)
        graph = onnx_ir.from_proto(model).graph
        assert set(_block_inputs(graph, end_points)[2]) == {
            "start_0",
            "start_2",
            "side",
        }
        plan, _, _ = _run_and_compare(model, end_points, _toy_inputs(model), tmp_path)
        assert plan.last_use["start_0"] == 2

    def test_weight_updated_again_restarts_cache(self, tmp_path):
        """
        Given: Blocks 0 and 1 share a weight the caller may still change at block 1, so
               block 1 may update a weight block 0's cached output was computed from
        When: Block 1 is sampled and the shared weight then changes again
        Then: The quantized sampler warns and restarts, the float sampler (which nothing
              changes) keeps its cache, and block 2's inputs are still correct
        """
        model, end_points = _toy_model(shared_weight=True)
        with (
            patch.object(blockwise_sampler, "_logger") as logger,
            patch.object(
                blockwise_sampler._ActivationCache,
                "clear",
                autospec=True,
                side_effect=blockwise_sampler._ActivationCache.clear,
            ) as clear,
        ):
            _, qt_sampler, fp_sampler = _run_and_compare(
                model,
                end_points,
                _toy_inputs(model),
                tmp_path,
                may_update=lambda name: name == "W_0",
            )
        warnings = [
            call.args[0] % call.args[1:] for call in logger.warning.call_args_list
        ]
        assert any(
            w.startswith("quantized: ") and "owned by an earlier block" in w
            for w in warnings
        )
        cleared = [call.args[0] for call in clear.call_args_list]
        assert qt_sampler._cache in cleared
        assert fp_sampler._cache not in cleared

    def test_cache_limit_evicts_long_lived_tensors(self, tmp_path):
        """
        Given: Block 3 reads block 0's input, and the cache limit is set to zero
        When: Blocks are sampled
        Then: The long-lived tensor is evicted with a warning and recomputed for block 3,
              which still matches full-prefix sampling
        """
        model, end_points = _toy_model(skip_read=(0, 3))
        with (
            patch.object(blockwise_sampler, "_MAX_CACHE_RATIO", 0.0),
            patch.object(blockwise_sampler, "_logger") as logger,
        ):
            _run_and_compare(model, end_points, _toy_inputs(model), tmp_path)
        assert any(
            "evicting ['start_0']" in (call.args[0] % call.args[1:])
            for call in logger.warning.call_args_list
        )

    def test_failed_advance_restarts_cache(self, tmp_path):
        """
        Given: A residual chain whose advance session fails on the second sample of block 1,
               after the first sample's cache already moved past block 0
        When: Block 1 is sampled again
        Then: The chain restarts from the graph inputs, and every block still matches
              full-prefix sampling
        """
        model, end_points = _toy_model()
        inputs = _toy_inputs(model)
        plan, (_, _), (fp_model, sampler) = _samplers(model, end_points, inputs)
        sampler.sample_block(0)
        assert sampler._cache.names() == ["start_0"]
        run = blockwise_sampler._run_session_to_numpy
        calls = []

        def _fail_second(*args):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("ORT failure")
            return run(*args)

        with patch.object(blockwise_sampler, "_run_session_to_numpy", _fail_second):
            with pytest.raises(RuntimeError, match="ORT failure"):
                sampler.sample_block(1)
        assert not sampler._cache.names()
        assert sampler.cache_nbytes() == 0
        for idx in range(1, len(end_points)):
            _assert_samples_close(
                sampler.sample_block(idx),
                _full_prefix(fp_model, plan.block_inputs[idx], inputs, tmp_path),
            )

    def test_weight_declared_in_if_branch(self, tmp_path):
        """
        Given: Block 1's weight is declared in, and read only by, an If branch (MoE-style)
        When: The sampling plan is built and blocks are sampled while the branch weight changes
        Then: Block 1 owns the branch weight, tensors computed from it depend on block 1, and
              sampling matches the full prefix
        """
        model, end_points = _toy_model(if_block=1)
        qt_model = onnx_ir.from_proto(model)
        plan = SamplingPlan(
            qt_model.graph,
            _block_inputs(qt_model.graph, end_points),
            [[e] for _, e in end_points],
        )
        assert plan.owner["W_1"] == 1
        assert "W_1" in plan.block_initializers[1]
        assert plan.dependency["end_1"] == 1
        assert not plan.is_final("end_1", 1)
        assert plan.is_final("end_1", 2)
        _run_and_compare(model, end_points, _toy_inputs(model), tmp_path)

    def test_unsorted_graph_raises(self):
        """
        Given: A graph whose nodes are not in topological order
        When: A sampling plan is built over it
        Then: It raises naming the node that reads a value before its producer, instead of
              misjudging which tensors are safe to reuse
        """
        model, end_points = _toy_model(num_blocks=2)
        graph = onnx_ir.from_proto(model).graph
        block_inputs = _block_inputs(graph, end_points)
        # Move the last node (the output MatMul) to the front
        last = list(graph)[-1]
        graph.remove(last)
        graph.insert_before(list(graph)[0], last)
        with pytest.raises(ValueError, match="topologically sorted"):
            SamplingPlan(graph, block_inputs, [[e] for _, e in end_points])

    def test_model_missing_plan_tensors_raises(self):
        """
        Given: A plan built over one graph, and a model that lacks a block input it names
        When: A sampler is built for that model
        Then: It raises naming the missing tensor, before sampling anything
        """
        model, end_points = _toy_model(num_blocks=2)
        graph = onnx_ir.from_proto(model).graph
        plan = SamplingPlan(
            graph, _block_inputs(graph, end_points), [[e] for _, e in end_points]
        )
        other = onnx_ir.from_proto(model)
        onnx_ir.convenience.create_value_mapping(other.graph)[
            "start_1"
        ].name = "renamed"
        with pytest.raises(ValueError, match=r"float: .*lacks block inputs.*'start_1'"):
            BlockwiseSampler(other, plan, [], _CPU, label="float")

    def test_cache(self):
        """
        Given: An activation cache
        When: Tensors are added, released for one sample, evicted and cleared
        Then: Lookups, names and the byte count follow, and releasing frees a sample's array
              before the tensor is evicted
        """
        cache = blockwise_sampler._ActivationCache("test: ")
        a = [np.zeros(4, np.float32), np.ones(4, np.float32)]
        cache.add("a", a)
        cache.add("b", [np.zeros(2, np.float32)] * 2)
        assert "a" in cache and cache.names() == ["a", "b"]
        assert cache.get("a", 1) is a[1]
        assert cache.nbytes() == 2 * 16 + 2 * 8
        cache.release(["a"], 0)
        assert cache.nbytes() == 16 + 2 * 8
        cache.evict(["a", "missing"])
        assert cache.names() == ["b"]
        cache.clear()
        assert not cache.names() and cache.nbytes() == 0

    @pytest.mark.cuda
    def test_bfloat16(self, tmp_path):
        """
        Given: A bfloat16 residual chain sampled on CUDA
        When: Every block's inputs are sampled blockwise while weights change
        Then: They match full-prefix sampling within bfloat16 tolerance
        """
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")
        model, end_points = _toy_model(dtype=TensorProto.BFLOAT16, glue=True)
        inputs = _toy_inputs(model, np_dtype=ml_dtypes.bfloat16)
        providers = ["CUDAExecutionProvider"]
        plan, (qt_model, qt_sampler), (fp_model, fp_sampler) = _samplers(
            model, end_points, inputs, providers
        )
        for idx, names in enumerate(plan.block_inputs):
            _assert_samples_close(
                fp_sampler.sample_block(idx),
                _full_prefix(fp_model, names, inputs, tmp_path, providers),
                rtol=2e-2,
                atol=2e-2,
            )
            _assert_samples_close(
                qt_sampler.sample_block(idx),
                _full_prefix(qt_model, names, inputs, tmp_path, providers),
                rtol=2e-2,
                atol=2e-2,
            )
            _perturb(qt_model, f"W_{idx}")


def _fake_optimize(records):
    """Stands in for ``optimize_adascale_block``: records the sampled inputs, then
    deterministically changes and freezes the block's weights, as copy-back would."""

    def _optimize(sim_model, quantizer_dict, fp_inputs, qsim_inputs, names, *_, **__):
        records.append(
            (
                [[np.array(a) for a in s] for s in fp_inputs],
                [[np.array(a) for a in s] for s in qsim_inputs],
            )
        )
        mapping = onnx_ir.convenience.create_value_mapping(sim_model.graph)
        _, initializers = walk_back(sim_model.graph, mapping, names[0], names[1])
        for name in initializers:
            quantizer = quantizer_dict.get(name)
            if quantizer is None or not quantizer.enabled:
                continue
            if quantizer.is_encoding_frozen():
                continue
            _perturb(sim_model, name, scale=1.25)
            quantizer.freeze_encodings()

    return _optimize


def _random_inputs(sim, num_samples=2):
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(num_samples):
        feed = {}
        for graph_input in sim.session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in graph_input.shape]
            if "int" in graph_input.type:
                feed[graph_input.name] = rng.integers(0, 16, shape).astype(np.int64)
            else:
                feed[graph_input.name] = rng.standard_normal(shape).astype(np.float32)
        samples.append(feed)
    return samples


def _apply(model, end_points, blockwise):
    """Run ``_apply_adascale`` with a fake optimizer; return the sampled inputs and weights."""
    sim = QuantizationSimModel(
        copy.deepcopy(model), param_type="int4", activation_type="int16"
    )
    sim._compute_param_encodings(overwrite=False)
    inputs = _random_inputs(sim)
    if end_points is None:
        end_points = adascale_optimizer.get_decoder_block_boundaries(sim.model.model)
    records = []
    with (
        patch.object(adascale_optimizer, "_BLOCKWISE_SAMPLING", blockwise),
        patch.object(
            AdaScale, "optimize_adascale_block", side_effect=_fake_optimize(records)
        ),
    ):
        AdaScale._apply_adascale(sim, inputs, end_points, num_iterations=1)
    weights = {
        init.name: numpy_helper.to_array(init)
        for init in sim.model.model.graph.initializer
    }
    return records, weights


def _gemma4_end_points(model):
    nodes = {node.name: node for node in model.graph.node}
    return [
        (
            nodes[f"/block{index}/input_norm/Pow"].input[0],
            nodes[f"/block{index}/Add_6"].output[0],
        )
        for index in range(4)
    ]


@pytest.mark.parametrize("decoder", ["llama", "gemma4_shared_kv"])
def test_apply_adascale_blockwise_matches_full_prefix(decoder):
    """
    Given: A decoder AdaScale runs over, with each block's weights changed after its turn
    When: AdaScale runs with blockwise sampling and with full-prefix sampling
    Then: Every block is handed the same fp and quantized inputs, and the final weights match
    """
    torch.manual_seed(0)
    if decoder == "llama":
        model, end_points = _export_decoder_with_ids(LlamaStyleDecoder()), None
    else:
        model = _export_gemma4_shared_kv_decoder(Gemma4StyleDecoder())
        end_points = _gemma4_end_points(model)

    full_records, full_weights = _apply(model, end_points, blockwise=False)
    blockwise_records, blockwise_weights = _apply(model, end_points, blockwise=True)

    assert len(blockwise_records) == len(full_records) > 1
    for (fp_a, qt_a), (fp_b, qt_b) in zip(blockwise_records, full_records):
        _assert_samples_close(fp_a, fp_b, rtol=1e-4, atol=1e-5)
        _assert_samples_close(qt_a, qt_b, rtol=1e-4, atol=1e-5)
    assert full_weights.keys() == blockwise_weights.keys()
    for name, weight in full_weights.items():
        np.testing.assert_array_equal(blockwise_weights[name], weight)
