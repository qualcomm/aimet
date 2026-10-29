# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
"""
Tests for aimet_onnx.peft (load_lora_adapters).

What LoRA does: a target layer ``y = base(x)`` becomes ``y = base(x) + ((x @ A) * s) @ B``
(``A`` is d x r, ``B`` is r x m, ``s = alpha / r``). ``load_lora_adapters`` inserts that as an
extra branch next to each target layer of an ONNX graph. With several adapters the ``A``s and
``B``s are concatenated along the rank axis, so one branch serves all of them.

Typical use, as exercised by ``TestLoadLoraAdapters``:

    onnx_model = torch.onnx.export(...)                      # base model, exported from PyTorch
    onnx_model = fix_node_names_pass(onnx_model)             # dynamo exports only
    adapters = [("a", "ckpt/a"), ("b", "ckpt/b")]            # order sets the rank slices
    lora_model = load_lora_adapters(onnx_model, adapters, adapter_scales=[("b", 0.5)])

Layout of this file, in the order the pipeline runs:
  1. checkpoint parsing   (TestCheckConfig, TestReadAdapter, TestAdapterLayout)
  2. finding the layers   (TestResolveTargets)
  3. inserting branches   (TestBranchInsertion, TestFindLoraBranches)
  4. the public API, compared against torch peft (TestLoadLoraAdapters)
The torch peft model is the reference: if our ONNX output matches it, the math is right.
"""

import io

import ml_dtypes
import numpy as np
import onnx
import onnx_ir
import onnxruntime as ort
import pytest
from onnx import numpy_helper

peft = pytest.importorskip("peft")
safetensors_numpy = pytest.importorskip("safetensors.numpy")
safetensors_torch = pytest.importorskip("safetensors.torch")
torch = pytest.importorskip("torch")

from aimet_onnx.experimental.lora import set_lora_bitwidth
from aimet_onnx.peft import (
    load_lora_adapters,
    load_lora_adapters_as_inputs,
)
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.peft.config import (
    AdapterLayout,
    AdapterInfo,
    _check_config,
    read_adapter,
)
from aimet_onnx.peft.peft_utils import (
    BranchWeights,
    LoraBranch,
    LoraTensorNames,
    _find_lora_branches,
    _weight_shapes,
    get_lora_tensor_names,
    insert_branches,
    pull_branch_weights_to_inputs,
    resolve_targets,
)
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import static_tensor
from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    fix_node_names_pass,
)


@pytest.fixture(autouse=True)
def _propagate_aimet_logs(monkeypatch):
    """The AIMET area logger does not propagate, so ``caplog`` only sees it if this is on."""
    logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
    monkeypatch.setattr(logger, "propagate", True)


def _lora_config(**kwargs):
    """A small ``LoraConfig`` (r=4, alpha=8) with extra fields from ``kwargs``."""
    return peft.LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], **kwargs)


def _write_weights(ckpt, tensors):
    """Write ``tensors`` as the checkpoint's ``adapter_model.safetensors``."""
    safetensors_numpy.save_file(tensors, str(ckpt / "adapter_model.safetensors"))


def _checkpoint_entries(path, r=4):
    """Zero ``lora_A``/``lora_B`` safetensors entries for the layer at ``path``."""
    return {
        f"base_model.model.{path}.lora_A.weight": np.zeros((r, 16), np.float32),
        f"base_model.model.{path}.lora_B.weight": np.zeros((16, r), np.float32),
    }


class TestCheckConfig:
    """``_check_config``: which LoraConfig settings are accepted."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lora_dropout": 0.1, "task_type": "CAUSAL_LM"},
            {"init_lora_weights": "gaussian"},
        ],
    )
    def test_supported(self, kwargs):
        """Configs that only differ in fields with no effect on inference are accepted."""
        _check_config("a", _lora_config(**kwargs))

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"bias": "all"}, "bias"),
            ({"init_lora_weights": "pissa"}, "init_lora_weights"),
        ],
    )
    def test_unsupported_raises(self, kwargs, field):
        """A config whose math we can't reproduce in ONNX is rejected, naming the field.

        ``bias="all"`` adds trainable biases; ``init_lora_weights="pissa"`` modifies the base
        weights, so the adapter is not relative to the unmodified base model.
        """
        with pytest.raises(ValueError, match=rf"LoraConfig\.{field}="):
            _check_config("a", _lora_config(**kwargs))


class TestReadAdapter:
    """``read_adapter``: reading and validating a checkpoint directory."""

    def test_parse(self, tmp_path):
        """``read_adapter`` returns the name, rank, alpha and the sorted target module paths.

        The paths come from the safetensors keys; the weights themselves are not loaded.
        """
        _lora_config(lora_dropout=0.1).save_pretrained(tmp_path)
        _write_weights(
            tmp_path, {**_checkpoint_entries("b.proj"), **_checkpoint_entries("a.proj")}
        )
        adapter = read_adapter("a", tmp_path)
        assert (adapter.name, adapter.r, adapter.alpha) == ("a", 4, 8)
        assert adapter.module_paths == ("a.proj", "b.proj")

    def test_rank_mismatch_raises(self, tmp_path):
        """A weight whose rank disagrees with ``LoraConfig.r`` is rejected."""
        _lora_config().save_pretrained(tmp_path)
        _write_weights(tmp_path, _checkpoint_entries("a.proj", r=2))
        with pytest.raises(ValueError, match="expected rank 4"):
            read_adapter("a", tmp_path)

    def test_unexpected_key_raises(self, tmp_path):
        """Weights this feature doesn't handle (e.g. embedding LoRA) are rejected."""
        _lora_config().save_pretrained(tmp_path)
        _write_weights(
            tmp_path,
            {"base_model.model.e.lora_embedding_A": np.zeros((4, 4), np.float32)},
        )
        with pytest.raises(ValueError, match="unsupported weight key"):
            read_adapter("a", tmp_path)


class TestAdapterLayout:
    """``AdapterLayout``: how adapters share the rank axis."""

    @staticmethod
    def _layout():
        """Two adapters: "x" (r=4, alpha=8) and "y" (r=2, alpha=2). Both target p1; only "y" p2."""
        return AdapterLayout.build(
            [AdapterInfo("x", 4, 8, ("p1",)), AdapterInfo("y", 2, 2, ("p1", "p2"))]
        )

    def test_offsets_and_rank(self):
        """Adapters are packed one after another along the rank axis.

        At p1 "x" owns ranks [0, 4) and "y" owns [4, 6), so R=6. At p2 only "y" is present.
        """
        layout = self._layout()
        assert layout.adapters == ("x", "y")
        assert [(s.adapter, s.offset, s.r) for s in layout.slices_by_path["p1"]] == [
            ("x", 0, 4),
            ("y", 4, 2),
        ]
        assert [(s.adapter, s.offset, s.r) for s in layout.slices_by_path["p2"]] == [
            ("y", 0, 2)
        ]
        assert (layout.rank("p1"), layout.rank("p2")) == (6, 2)

    def test_offsets_follow_the_input_order(self):
        """Swapping the adapter order swaps the slices; nothing is re-sorted by name."""
        layout = AdapterLayout.build(
            [AdapterInfo("y", 2, 2, ("p1", "p2")), AdapterInfo("x", 4, 8, ("p1",))]
        )
        assert [(s.adapter, s.offset) for s in layout.slices_by_path["p1"]] == [
            ("y", 0),
            ("x", 2),
        ]

    def test_scale_vector(self):
        """The per-rank scale is ``adapter_scales * alpha / r`` for the owning adapter.

        "x" has alpha/r = 2 and "y" has alpha/r = 1; adapter_scales 0.5 halves "y", 0 turns it off.
        """
        layout = self._layout()
        np.testing.assert_array_equal(
            layout.scale_vector("p1", {"x": 1.0, "y": 0.5}), [2, 2, 2, 2, 0.5, 0.5]
        )
        np.testing.assert_array_equal(
            layout.scale_vector("p2", {"x": 1.0, "y": 0.0}), [0, 0]
        )

    def test_pad_rank(self):
        """With ``pad_rank`` every branch has R = 4 + 2 = 6; the slices don't move.

        p2, which only "y" targets, keeps "y" at [0, 2) and gets zero-scale padding [2, 6).
        """
        padded = AdapterLayout.build(
            [AdapterInfo("x", 4, 8, ("p1",)), AdapterInfo("y", 2, 2, ("p1", "p2"))],
            pad_rank=True,
        )
        assert padded.slices_by_path == self._layout().slices_by_path
        assert (padded.rank("p1"), padded.rank("p2")) == (6, 6)
        np.testing.assert_array_equal(
            padded.scale_vector("p2", {"x": 1.0, "y": 0.5}), [0.5, 0.5, 0, 0, 0, 0]
        )
        rekeyed = padded.rekeyed({"p2": "q2"})
        assert rekeyed.rank("q2") == 6


# ---------------------------------------------------------------------------------------------
# Test models: built in PyTorch, exported to ONNX
# ---------------------------------------------------------------------------------------------

# peft target_modules; ONNX paths are e.g. layers.0.q_proj
ALL_LORA_TARGETS = ["q_proj", "v_proj", "lm_head", "features.0"]


class _Layer(torch.nn.Module):
    """A bias-free ``q_proj`` and a ``v_proj`` with bias, added together."""

    def __init__(self, width):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, width, bias=False)
        # exports as MatMul + bias Add
        self.v_proj = torch.nn.Linear(width, width, bias=True)

    def forward(self, x):
        """Sum of the two projections."""
        return self.q_proj(x) + self.v_proj(x)


class _TinyModel(torch.nn.Module):
    """Linear layers in a ModuleList, a conv in a Sequential, and an lm_head feeding the output."""

    def __init__(self, width=16, num_layers=2):
        super().__init__()
        self.width = width
        self.layers = torch.nn.ModuleList([_Layer(width) for _ in range(num_layers)])
        self.lm_head = torch.nn.Linear(width, 8, bias=False)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU()
        )

    def forward(self, x, img):
        """Return ``(logits, feat)``: the layers and ``lm_head`` on ``x``; the conv on ``img``."""
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x), self.features(img)


def _base_model(width=16, num_layers=2, dtype=torch.float32):
    """The same ``_TinyModel`` every call (fixed seed), in eval mode."""
    torch.manual_seed(0)  # every call gives the same base weights
    return _TinyModel(width, num_layers).to(dtype).eval()


class _Wrapper(torch.nn.Module):
    """Holds ``module`` as ``self.model``, so graph paths gain a leading ``model.``."""

    def __init__(self, module):
        super().__init__()
        self.model = module

    def forward(self, *args):
        """Run the wrapped module."""
        return self.model(*args)


class _OneLayerModel(torch.nn.Module):
    """One layer held as the attribute ``name``, so its graph path is exactly ``name``."""

    def __init__(self, layer, name="q_proj"):
        super().__init__()
        self.name = name
        setattr(self, name, layer)

    def forward(self, x):
        """Apply the layer."""
        return getattr(self, self.name)(x)


def _export(module, inputs, output_names=None, dynamo=False, fix_names=False):
    """Export ``module`` to a ModelProto; inputs are named ``x`` and ``img``."""
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            module.eval(),
            inputs,
            buf,
            input_names=["x", "img"][: len(inputs)],
            output_names=output_names,
            dynamo=dynamo,
        )
    except Exception as e:  # pylint: disable=broad-except
        if dynamo:
            pytest.skip(f"dynamo export unavailable: {e}")
        raise
    proto = onnx.load_from_string(buf.getvalue())
    return fix_node_names_pass(proto) if fix_names else proto


def _tiny_proto(model, dynamo=False, fix_names=False, width=16):
    """Export a ``_TinyModel`` (or wrapper): inputs ``x``, ``img``; outputs ``logits``, ``feat``."""
    dtype = next(model.parameters()).dtype
    inputs = (
        torch.zeros(1, 4, width, dtype=dtype),
        torch.zeros(1, 3, 8, 8, dtype=dtype),
    )
    return _export(model, inputs, ["logits", "feat"], dynamo, fix_names)


def _base_graph(dynamo=False, fix_names=False):
    """The base model's onnx_ir graph."""
    return onnx_ir.from_proto(_tiny_proto(_base_model(), dynamo, fix_names)).graph


def _sample_inputs():
    """Fixed random inputs for ``_tiny_proto`` models."""
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((1, 4, 16)).astype(np.float32),
        "img": rng.standard_normal((1, 3, 8, 8)).astype(np.float32),
    }


def _run_onnx(proto, feeds):
    """Run ``proto`` in onnxruntime with graph optimizations off; returns all outputs."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        proto.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


# ---------------------------------------------------------------------------------------------
# PEFT adapters: trained-looking checkpoints plus a torch peft reference
# ---------------------------------------------------------------------------------------------


def _save_adapter(
    tmp_path, name, r, alpha, targets, seed, dtype=torch.float32, base=None
):
    """Save a PEFT checkpoint of ``base`` (default ``_base_model()``) with random weights.

    peft zero-inits ``lora_B``, so it is randomized to make the adapter do something.
    """
    model = peft.get_peft_model(
        base or _base_model(),
        peft.LoraConfig(r=r, lora_alpha=alpha, target_modules=targets),
    )
    torch.manual_seed(seed)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    model.to(dtype).save_pretrained(tmp_path / name)
    return tmp_path / name


def _torch_peft_outputs(ckpts, feeds, adapter_scales=None):
    """Outputs of torch peft with all adapters active, each scaled by its adapter_scales.

    ``ckpts`` and ``adapter_scales`` are lists of pairs, as for ``load_lora_adapters``.
    """
    ckpts = dict(ckpts)
    adapter_scales = dict(adapter_scales or ())
    adapter_scales = {name: adapter_scales.get(name, 1.0) for name in ckpts}
    model = None
    for name, path in ckpts.items():
        if model is None:
            model = peft.PeftModel.from_pretrained(
                _base_model(), path, adapter_name=name
            )
        else:
            model.load_adapter(path, adapter_name=name)
    model.base_model.set_adapter(list(ckpts))
    for module in model.modules():
        if hasattr(module, "set_scale"):
            for name, value in adapter_scales.items():
                if name in module.lora_A:
                    module.set_scale(name, value)
    with torch.no_grad():
        out = model.eval()(torch.from_numpy(feeds["x"]), torch.from_numpy(feeds["img"]))
    return [o.numpy() for o in out]


# ---------------------------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------------------------


def _one_layer_graph(layer, input_shape, name="q_proj", wrap=False):
    """The onnx_ir graph of one exported layer, optionally inside a wrapper module."""
    module = _OneLayerModel(layer, name)
    module = _Wrapper(module) if wrap else module
    return onnx_ir.from_proto(_export(module, (torch.zeros(input_shape),))).graph


class TestResolveTargets:
    """``resolve_targets`` maps a torch module path to the ONNX node that will get a branch."""

    @pytest.mark.parametrize(
        "path, kind, d, m, merges_after_bias",
        [
            pytest.param(
                "layers.0.q_proj", "linear", 16, 16, False, id="linear-no-bias"
            ),
            pytest.param(
                "layers.0.v_proj", "linear", 16, 16, True, id="linear-with-bias"
            ),
            pytest.param("features.0", "conv", 3, 8, False, id="conv"),
        ],
    )
    def test_resolves_supported_layers(self, path, kind, d, m, merges_after_bias):
        """Linear, Linear with bias and Conv2d resolve to the right kind and sizes.

        ``d``/``m`` are the input/output features (Linear) or channels (Conv). The branch output
        is added to ``base_out``: the layer's own output, or the output of its bias Add.
        """
        point = resolve_targets(_base_graph(), [path])[path]

        assert (point.kind, point.in_features, point.out_features) == (kind, d, m)
        if merges_after_bias:
            assert point.base_out.producer().op_type == "Add"
        else:
            # q_proj is followed by a residual Add, which is not its bias
            assert point.base_out is point.node.outputs[0]

    def test_unknown_path_suggests_close_matches(self):
        """A typo in the module path raises and lists the closest real paths."""
        with pytest.raises(ValueError, match=r"Closest paths: \['layers.0.q_proj'"):
            resolve_targets(_base_graph(), ["layers.0.q_prj"])

    def test_adapter_prefix_is_ignored(self):
        """The reverse also works: the adapter was trained with a wrapper that the graph lacks."""
        points = resolve_targets(_base_graph(), ["model.layers.0.q_proj"])

        assert points["layers.0.q_proj"].checkpoint_path == "model.layers.0.q_proj"

    @pytest.mark.parametrize(
        "paths, error",
        [
            pytest.param(
                ["q_proj", "x.model.q_proj"], "different prefixes", id="wrong-model"
            ),
            pytest.param(
                ["model.q_proj", "q_proj"], "same layer", id="same-layer-twice"
            ),
        ],
    )
    def test_inconsistent_prefixes_raise(self, paths, error):
        """Paths that reach the graph through different prefixes are rejected, not guessed at.

        The graph has ``model.q_proj``. Either the paths come from another model, or two adapters
        name one layer differently (with and without ``model.``).
        """
        graph = _one_layer_graph(
            torch.nn.Linear(16, 16, bias=False), (1, 16), wrap=True
        )
        with pytest.raises(ValueError, match=error):
            resolve_targets(graph, paths)

    def test_skipped_layers_that_match_through_a_prefix_are_reported(self, caplog):
        """A top-level exact match turns the shift off; the warning says what it missed."""

        class Mixed(torch.nn.Module):
            """A top-level ``q_proj`` next to ``model.v_proj``."""

            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(16, 16, bias=False)
                self.model = _OneLayerModel(
                    torch.nn.Linear(16, 16, bias=False), "v_proj"
                )

            def forward(self, x):
                """Add the two projections."""
                return self.q_proj(x) + self.model(x)

        graph = onnx_ir.from_proto(_export(Mixed(), (torch.zeros(1, 16),))).graph

        with caplog.at_level("WARNING"):
            points = resolve_targets(graph, ["q_proj", "v_proj"])

        assert list(points) == ["q_proj"]
        assert "leading 'model.'" in caplog.text

    def test_dynamo_export_needs_fix_node_names_pass(self):
        """Dynamo node names must be fixed first; the error says to call ``fix_node_names_pass``."""
        with pytest.raises(ValueError, match="fix_node_names_pass"):
            resolve_targets(_base_graph(dynamo=True), ["layers.0.q_proj"])

    @pytest.mark.parametrize(
        "layer, input_shape",
        [
            pytest.param(
                torch.nn.Conv2d(3, 8, 3, dilation=2), (1, 3, 8, 8), id="dilated-conv"
            ),
            pytest.param(
                torch.nn.Conv2d(8, 8, 3, groups=8), (1, 8, 8, 8), id="grouped-conv"
            ),
        ],
    )
    def test_unsupported_conv_raises(self, layer, input_shape):
        """A single A/B branch can't reproduce a dilated or grouped Conv."""
        graph = _one_layer_graph(layer, input_shape, name="fc")
        with pytest.raises(ValueError, match="not supported"):
            resolve_targets(graph, ["fc"])


# ---------------------------------------------------------------------------------------------
# Branch insertion and finding the branches (A, B and the scale are filled with random values)
# ---------------------------------------------------------------------------------------------


def _attach_random_branches(proto, targets, rank=3, seed=0):
    """Insert branches at ``targets`` with random A, B and scale values."""
    ir_model = onnx_ir.from_proto(proto)
    points = resolve_targets(ir_model.graph, targets)
    rng = np.random.default_rng(seed)

    def random(shape, dtype):
        """A random array in (0.1, 1.0), so the branch contributes."""
        return rng.uniform(0.1, 1.0, shape).astype(dtype)

    weights = {}
    for path, point in points.items():
        a_shape, b_shape, s_shape = _weight_shapes(point, rank)
        dtype = static_tensor(point.weight).dtype.numpy()
        weights[path] = BranchWeights(
            random(a_shape, dtype), random(b_shape, dtype), random(s_shape, dtype)
        )
    insert_branches(ir_model.graph, points, weights)
    return onnx_ir.to_proto(ir_model)


ONE_OF_EACH_KIND = ["layers.0.q_proj", "layers.1.v_proj", "features.0"]


class TestBranchInsertion:
    """``insert_branches``: where the branch goes and what it does to the graph's outputs."""

    @pytest.mark.parametrize(
        "layer, shape",
        [
            (torch.nn.Linear(16, 8, bias=False), (1, 4, 16)),
            (torch.nn.Conv2d(16, 8, 1), (1, 16, 4, 4)),
        ],
    )
    def test_output_name_is_preserved_when_attaching_at_a_graph_output(
        self, layer, shape
    ):
        """Attaching at the final layer keeps the model's output name.

        The Add takes over the graph output name (``logits``); the base output is renamed. The
        numerics of this case are covered by the torch-peft comparison, which targets ``lm_head``.
        """

        module = _OneLayerModel(layer, "lm_head")
        proto = _attach_random_branches(
            _export(module, (torch.zeros(shape),), ["logits"]), ["lm_head"], rank=2
        )
        onnx.checker.check_model(proto)
        assert [o.name for o in proto.graph.output] == ["logits"]

        (branch,) = _find_lora_branches(proto)
        assert (branch.merged_output, branch.base_layer_output) == (
            "logits",
            "/lm_head/lora/base_output_0",
        )
        assert get_lora_tensor_names(proto).merges == (
            ("/lm_head/lora/base_output_0", "logits"),
        )


class TestFindLoraBranches:
    """Finding the branches by name, and the LoRA tensor names callers rely on."""

    def test_names_for_linear_bias_and_conv(self):
        """The literal names for a Linear, a Linear with bias and a Conv2d (a rename fails here).

        With a bias, the base output is the bias Add's output, so the branch merges after the bias.
        """
        proto = _attach_random_branches(_tiny_proto(_base_model()), ONE_OF_EACH_KIND)

        linear, with_bias, conv = _find_lora_branches(proto)

        assert linear == LoraBranch(
            module_path="layers.0.q_proj",
            node_prefix="/layers.0/q_proj",
            kind="linear",
            a_weight="layers.0.q_proj.lora_A.weight",
            b_weight="layers.0.q_proj.lora_B.weight",
            scale_weight="layers.0.q_proj.lora_scale",
            a_proj_node="/layers.0/q_proj/lora_A/MatMul",
            scale_node="/layers.0/q_proj/lora/Mul",
            b_proj_node="/layers.0/q_proj/lora_B/MatMul",
            merge_node="/layers.0/q_proj/lora/Add",
            a_proj_output="/layers.0/q_proj/lora_A/MatMul_output_0",
            scaled_output="/layers.0/q_proj/lora/Mul_output_0",
            b_proj_output="/layers.0/q_proj/lora_B/MatMul_output_0",
            merged_output="/layers.0/q_proj/lora/Add_output_0",
            base_layer_output="/layers.0/q_proj/MatMul_output_0",
        )
        assert with_bias.base_layer_output == "/layers.1/v_proj/Add_output_0"
        assert with_bias.merged_output == "/layers.1/v_proj/lora/Add_output_0"
        assert (conv.module_path, conv.node_prefix, conv.kind) == (
            "features.0",
            "/features/features.0",
            "conv",
        )
        assert conv.a_proj_node == "/features/features.0/lora_A/Conv"
        assert conv.base_layer_output == "/features/features.0/Conv_output_0"

    def test_get_lora_tensor_names(self):
        """The helpers' names, in branch order; the merged outputs are only in ``merges``."""
        proto = _attach_random_branches(_tiny_proto(_base_model()), ONE_OF_EACH_KIND)
        branches = _find_lora_branches(proto)

        names = get_lora_tensor_names(proto)

        assert names == LoraTensorNames(
            params=tuple(n for b in branches for n in (b.a_weight, b.b_weight)),
            activations=tuple(
                n
                for b in branches
                for n in (b.a_proj_output, b.scaled_output, b.b_proj_output)
            ),
            scales=tuple(b.scale_weight for b in branches),
            merges=tuple((b.base_layer_output, b.merged_output) for b in branches),
        )
        assert names.params[:2] == (
            "layers.0.q_proj.lora_A.weight",
            "layers.0.q_proj.lora_B.weight",
        )
        assert names.merges[0] == (
            "/layers.0/q_proj/MatMul_output_0",
            "/layers.0/q_proj/lora/Add_output_0",
        )
        assert names.quantizer_names == names.params + names.activations
        assert not {merged for _, merged in names.merges} & set(names.quantizer_names)

    def test_model_without_branches_has_none(self):
        """A base model has no branches."""
        assert _find_lora_branches(_tiny_proto(_base_model())) == []

    def test_renamed_branch_node_raises(self):
        """A branch node that was renamed is reported, since its name is how it is found."""
        proto = _attach_random_branches(_tiny_proto(_base_model()), ["layers.0.q_proj"])
        mul = next(n for n in proto.graph.node if n.name.endswith("/lora/Mul"))
        mul.name = "renamed"

        with pytest.raises(ValueError, match="lora/Mul"):
            _find_lora_branches(proto)


# ---------------------------------------------------------------------------------------------
# load_lora_adapters (public API)
# ---------------------------------------------------------------------------------------------


class TestLoadLoraAdapters:
    """``load_lora_adapters`` end to end, checked against torch peft."""

    # The scales don't depend on the exporter, so dynamo only runs the default scales
    @pytest.mark.parametrize(
        "adapter_scales, dynamo",
        [
            pytest.param(None, False, id="torchscript"),
            pytest.param(None, True, id="dynamo"),
            pytest.param([("x", 0.5), ("y", 2.0)], False, id="scaled"),
            pytest.param([("x", -1.0), ("y", 0.0)], False, id="negated-and-off"),
        ],
    )
    def test_worked_example_two_adapters(self, tmp_path, adapter_scales, dynamo):
        """WORKED EXAMPLE: read this to learn the flow. Every step is spelled out inline.

        Take a base model exported to ONNX plus two trained LoRA checkpoints, call
        ``load_lora_adapters``, and get the same numbers as torch peft with both adapters active
        at once.

        The user-facing part is steps 3-4. Steps 1-2 and 5-6 are setup and checking:

          1. Export the base model to ONNX (done once, with ``torch.onnx.export``). A dynamo export
             also needs ``fix_node_names_pass``, because layers are found by node name.
          2. Train and save adapters with HF peft. Here "x" (r=4) targets every layer and "y" (r=2)
             only ``v_proj``; peft zero-inits ``lora_B``, so the test randomizes it.
          3. ``load_lora_adapters(model, [("x", dir), ("y", dir)], adapter_scales=...)``. Per target
             layer it reads each adapter's A and B, concatenates them along the rank axis, and adds
             ``y = base(x) + ((x @ A_cat) * s) @ B_cat`` to the graph. ``s`` holds
             ``adapter_scale * alpha / r`` for each rank column.
          4. The result is a plain ``ModelProto``: run it in onnxruntime or load it in QuantSim.
          5. Check the output against torch peft with the same checkpoints and scales.
          6. Check the concatenated ranks: ``q_proj`` has R=4 (only "x"), ``v_proj`` has R=6.
        """
        # --- 1. Export the base model to ONNX (setup; not part of the LoRA API). ---
        base_onnx = _tiny_proto(_base_model(), dynamo=dynamo)
        if dynamo:
            # Dynamo names nodes ``node_*``; LoRA finds layers by ``/<module path>/<op>`` names.
            base_onnx = fix_node_names_pass(base_onnx)

        # --- 2. Trained adapters (setup; a user brings their own checkpoint directories). ---
        adapter_dirs = [
            (
                "x",
                _save_adapter(
                    tmp_path, "x", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
                ),
            ),
            (
                "y",
                _save_adapter(tmp_path, "y", r=2, alpha=2, targets=["v_proj"], seed=2),
            ),
        ]

        # ============================== LoRA API starts here ==============================
        # --- 3. Add both adapters. The input model is not modified; a new one is returned. ---
        lora_model = load_lora_adapters(
            base_onnx, adapter_dirs, adapter_scales=adapter_scales
        )

        # --- 4. ``lora_model`` is a normal ModelProto: run it. ---
        onnx_outputs = _run_onnx(lora_model, _sample_inputs())
        # =============================== LoRA API ends here ===============================

        # --- 5. Torch peft with the same checkpoints and scales is the reference. ---
        torch_outputs = _torch_peft_outputs(
            adapter_dirs, _sample_inputs(), adapter_scales
        )
        for onnx_out, torch_out in zip(onnx_outputs, torch_outputs):
            np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5)

        # --- 6. The adapters share one branch per layer; its rank is the sum of their ranks. ---
        scale_length = {
            i.name: i.dims[0]
            for i in lora_model.graph.initializer
            if i.name.endswith(".lora_scale")
        }
        assert scale_length["layers.0.q_proj.lora_scale"] == 4  # "x" only
        assert scale_length["layers.0.v_proj.lora_scale"] == 6  # "x" (4) + "y" (2)

    @pytest.mark.parametrize("dynamo", [False, True])
    def test_single_adapter_matches_torch_peft(self, tmp_path, dynamo):
        """One adapter: the ONNX model matches the same checkpoint loaded with torch peft.

        Runs for TorchScript and dynamo exports. Covers Linear (with and without bias) and Conv2d.
        """
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )

        lora_model = load_lora_adapters(
            _tiny_proto(_base_model(), dynamo, fix_names=dynamo), [("a", ckpt)]
        )

        expected = _torch_peft_outputs([("a", ckpt)], _sample_inputs())
        for got, ref in zip(_run_onnx(lora_model, _sample_inputs()), expected):
            np.testing.assert_allclose(got, ref, atol=1e-5)
        assert {b.module_path for b in _find_lora_branches(lora_model)} == {
            "layers.0.q_proj",
            "layers.0.v_proj",
            "layers.1.q_proj",
            "layers.1.v_proj",
            "lm_head",
            "features.0",
        }

    @pytest.mark.parametrize(
        "layer, shape",
        [
            pytest.param(torch.nn.Linear(16, 8), (4, 16), id="gemm"),
            pytest.param(torch.nn.Conv1d(3, 8, 3), (1, 3, 8), id="conv1d"),
        ],
    )
    def test_each_layer_kind_matches_torch_peft(self, tmp_path, layer, shape):
        """ONNX forms of a base layer not in ``_TinyModel`` give the same output as torch peft.

        A 2-D Linear input exports as ``Gemm`` (``transB=1``); Conv1d checks the branch handles
        one spatial dim. MatMul with and without bias and Conv2d are covered by the worked example.
        """
        module = _OneLayerModel(layer)
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=["q_proj"], seed=1, base=module
        )
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        base = _export(_OneLayerModel(layer), (torch.zeros(shape),))

        lora_model = load_lora_adapters(base, [("a", ckpt)])

        with torch.no_grad():
            expected = module(torch.from_numpy(x)).numpy()
        (got,) = _run_onnx(lora_model, {"x": x})
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_quantsim_quantizes_the_lora_tensors(self, tmp_path):
        """The model from ``load_lora_adapters`` loads in QuantSim; its LoRA tensors get
        quantizers."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        lora_model = load_lora_adapters(_tiny_proto(_base_model()), [("a", ckpt)])

        sim = QuantizationSimModel(
            lora_model,
            dummy_input=_sample_inputs(),
            param_type="int4",
            activation_type="int16",
        )
        sim.compute_encodings([_sample_inputs()])

        branches = _find_lora_branches(lora_model)
        assert {b.kind for b in branches} == {"linear", "conv"}
        for branch in branches:
            for name in (
                branch.a_weight,
                branch.b_weight,
                branch.scale_weight,
                branch.a_proj_output,
                branch.scaled_output,
                branch.b_proj_output,
                branch.merged_output,
            ):
                assert name in sim.qc_quantize_op_dict

    @pytest.fixture
    def two_adapters(self, tmp_path):
        """ "x" (r=4) targets everything; "y" (r=2) only v_proj, so q_proj has rank 4, v_proj 6."""
        return [
            (
                "x",
                _save_adapter(
                    tmp_path, "x", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
                ),
            ),
            (
                "y",
                _save_adapter(tmp_path, "y", r=2, alpha=2, targets=["v_proj"], seed=2),
            ),
        ]

    def test_pad_rank(self, two_adapters):
        """``pad_rank=True`` gives every branch R = 4 + 2 = 6 and the same output as without it.

        ``q_proj``, ``lm_head`` and ``features.0`` only have "x" (ranks [0, 4)); ranks [4, 6) are
        exactly zero in A, B and the scale. Covers linear, linear with bias and conv branches.
        """
        base = _tiny_proto(_base_model())
        packed = load_lora_adapters(base, two_adapters)
        padded = load_lora_adapters(base, two_adapters, pad_rank=True)

        for got, ref in zip(
            _run_onnx(padded, _sample_inputs()), _run_onnx(packed, _sample_inputs())
        ):
            np.testing.assert_allclose(got, ref, atol=1e-5)

        initializers = {
            i.name: numpy_helper.to_array(i) for i in padded.graph.initializer
        }
        packed_shapes = {i.name: tuple(i.dims) for i in packed.graph.initializer}
        branches = _find_lora_branches(padded)
        assert {b.kind for b in branches} == {"linear", "conv"}
        for branch in branches:
            a = initializers[branch.a_weight]
            b = initializers[branch.b_weight]
            scale = initializers[branch.scale_weight].reshape(-1)
            assert scale.shape == (6,)
            # linear: A [d, R], B [R, m]; conv: A [R, Cin, *k], B [Cout, R, 1, ...]
            a_ranks = a if branch.kind == "conv" else a.T
            b_ranks = b.swapaxes(0, 1) if branch.kind == "conv" else b
            r = packed_shapes[branch.scale_weight][1 if branch.kind == "conv" else 0]
            assert not a_ranks[r:].any() and not b_ranks[r:].any()
            assert not scale[r:].any()
            assert scale[:r].all()

        # The default is unchanged: only v_proj has both adapters
        assert packed_shapes["layers.0.q_proj.lora_scale"] == (4,)
        assert packed_shapes["layers.0.v_proj.lora_scale"] == (6,)

    @pytest.mark.parametrize(
        "adapter_scales, error",
        [
            ([("y", float("nan"))], "must be finite"),
            ([("z", 1.0)], "unknown adapters"),
        ],
    )
    def test_invalid_adapter_scales_raise(self, two_adapters, adapter_scales, error):
        """Scaling must be finite and may only name the given adapters."""
        with pytest.raises(ValueError, match=error):
            load_lora_adapters(
                _tiny_proto(_base_model()), two_adapters, adapter_scales=adapter_scales
            )

    def test_empty_adapters_raise(self):
        """At least one adapter is required."""
        with pytest.raises(ValueError, match="at least one"):
            load_lora_adapters(_tiny_proto(_base_model()), [])

    def test_duplicate_adapters_raise(self, two_adapters):
        """Adapter names must be unique: they key the scales and the error messages."""
        with pytest.raises(ValueError, match="unique"):
            load_lora_adapters(
                _tiny_proto(_base_model()), [two_adapters[0], two_adapters[0]]
            )

    def test_input_model_is_not_modified(self, tmp_path):
        """``load_lora_adapters`` works on a copy: the input ModelProto is unchanged, so it can
        be reused."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        base = _tiny_proto(_base_model())
        before = base.SerializeToString()

        first, second = (
            load_lora_adapters(base, [("a", ckpt)]),
            load_lora_adapters(base, [("a", ckpt)]),
        )

        assert base.SerializeToString() == before
        assert first == second

    def test_lora_model_cannot_be_the_base(self, tmp_path):
        """Passing the output of ``load_lora_adapters`` back in raises; start from the base
        model."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        lora_model = load_lora_adapters(_tiny_proto(_base_model()), [("a", ckpt)])

        with pytest.raises(ValueError, match="already contains LoRA branches"):
            load_lora_adapters(lora_model, [("a", ckpt)])

    def test_adapter_layers_missing_from_base_are_skipped(self, tmp_path, caplog):
        """An adapter trained on more layers than the base has loads on the layers that exist."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        # the adapter also targets layers.1
        one_layer = _tiny_proto(_base_model(num_layers=1))

        with caplog.at_level("WARNING"):
            lora_model = load_lora_adapters(one_layer, [("a", ckpt)])

        names = {i.name for i in lora_model.graph.initializer}
        assert any(
            n.startswith("layers.0.") and n.endswith(".lora_A.weight") for n in names
        )
        assert not any(n.startswith("layers.1.") and ".lora_" in n for n in names)
        assert "layers.1" in caplog.text  # the warning names a skipped layer

    def test_no_matching_layer_raises(self, tmp_path):
        """If none of the adapter's layers are in the base model, it raises (wrong model)."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        other = _export(
            _OneLayerModel(torch.nn.Linear(16, 16), "unrelated"), (torch.zeros(1, 16),)
        )
        with pytest.raises(ValueError, match="No MatMul/Gemm/Conv"):
            load_lora_adapters(other, [("a", ckpt)])

    def test_shape_mismatch_raises(self, tmp_path):
        """An adapter trained for a different width than the base model raises."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        # the adapter was trained at width 16
        narrow = _tiny_proto(_base_model(width=12), width=12)
        with pytest.raises(ValueError, match="doesn't fit"):
            load_lora_adapters(narrow, [("a", ckpt)])

    def test_fp16_base_needs_fp16_adapter(self, tmp_path):
        """Weights are never cast: an fp32 adapter on an fp16 base raises with a fix-it hint.

        An fp16 adapter works and the LoRA initializers are fp16.
        """
        fp16_base = _tiny_proto(_base_model(dtype=torch.float16))
        fp32_ckpt = _save_adapter(
            tmp_path, "f32", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        with pytest.raises(ValueError, match="re-save the adapter"):
            load_lora_adapters(fp16_base, [("a", fp32_ckpt)])

        fp16_ckpt = _save_adapter(
            tmp_path,
            "f16",
            r=4,
            alpha=8,
            targets=ALL_LORA_TARGETS,
            seed=1,
            dtype=torch.float16,
        )
        lora_model = load_lora_adapters(fp16_base, [("a", fp16_ckpt)])
        lora_inits = [i for i in lora_model.graph.initializer if "lora_" in i.name]
        assert lora_inits and all(
            i.data_type == onnx.TensorProto.FLOAT16 for i in lora_inits
        )

    def test_external_data_base(self, tmp_path):
        """Large bases are loaded without their weights; the LoRA initializers are added inline."""
        ckpt = _save_adapter(
            tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
        )
        path = tmp_path / "base.onnx"
        onnx.save(
            _tiny_proto(_base_model()),
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            size_threshold=0,
        )
        lora_model = load_lora_adapters(
            onnx.load(path, load_external_data=False), [("a", ckpt)]
        )
        external = {
            i.name: i.data_location == onnx.TensorProto.EXTERNAL
            for i in lora_model.graph.initializer
        }
        assert any(v for k, v in external.items() if "lora_" not in k)
        assert not any(v for k, v in external.items() if "lora_" in k)

    def test_bfloat16_adapter_on_bfloat16_base(self, tmp_path):
        """A bf16 adapter loads onto a bf16 base, through both loaders, with the
        checkpoint's values kept bit-exact; an fp32 adapter still raises.

        safetensors' numpy reader can't read bf16, so this fails if the checkpoint is read
        through it.
        """
        bf16_base = _tiny_proto(_base_model(dtype=torch.bfloat16))
        ckpt = _save_adapter(
            tmp_path,
            "a",
            r=4,
            alpha=8,
            targets=ALL_LORA_TARGETS,
            seed=1,
            dtype=torch.bfloat16,
        )
        with pytest.raises(ValueError, match="re-save the adapter"):
            load_lora_adapters(
                bf16_base,
                [("f32", _save_adapter(tmp_path, "f32", 4, 8, ALL_LORA_TARGETS, 1))],
            )

        lora_model = load_lora_adapters(bf16_base, [("a", ckpt)])
        _, adapter_inputs = load_lora_adapters_as_inputs(bf16_base, [("a", ckpt)])

        lora_inits = {
            i.name: numpy_helper.to_array(i)
            for i in lora_model.graph.initializer
            if "lora_" in i.name
        }
        assert lora_inits and all(
            i.data_type == onnx.TensorProto.BFLOAT16
            for i in lora_model.graph.initializer
            if "lora_" in i.name
        )
        # The checkpoint's bf16 values land in the graph unchanged, in the graph's layout
        torch_ckpt = safetensors_torch.load_file(
            str(ckpt / "adapter_model.safetensors")
        )
        key = "base_model.model.layers.0.q_proj.lora_A.weight"
        expected = torch_ckpt[key].T.contiguous().view(torch.int16).numpy()
        got = lora_inits["layers.0.q_proj.lora_A.weight"].view(np.int16)
        assert np.array_equal(got, expected)
        inputs = adapter_inputs["a"]
        assert inputs["layers.0.q_proj.lora_A.weight"].dtype == ml_dtypes.bfloat16
        assert np.array_equal(
            inputs["layers.0.q_proj.lora_A.weight"].view(np.int16), expected
        )
        assert inputs["lora_scale"].dtype == ml_dtypes.bfloat16
        assert inputs["lora_scale"].astype(np.float32).tolist() == [2.0]

    @pytest.mark.cuda
    def test_bfloat16_matches_torch_peft(self, tmp_path):
        """A bf16 LoRA model gives torch peft's bf16 output, up to bf16 rounding.

        Runs on CUDA (ORT has few bf16 CPU kernels) and on a base without the conv: ORT has no
        bf16 Conv kernel, with or without LoRA.
        """
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")
        from aimet_onnx.utils import OrtInferenceSession, _run_session_to_numpy

        linear = ["q_proj", "v_proj", "lm_head"]

        class _NoConv(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                tiny = _base_model(dtype=dtype)
                self.layers, self.lm_head = tiny.layers, tiny.lm_head

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        ckpt = _save_adapter(
            tmp_path,
            "a",
            r=4,
            alpha=8,
            targets=linear,
            seed=1,
            dtype=torch.bfloat16,
            base=_NoConv(torch.float32),
        )
        base = _export(
            _NoConv(torch.bfloat16),
            (torch.zeros(1, 4, 16, dtype=torch.bfloat16),),
            ["logits"],
        )
        x = torch.from_numpy(_sample_inputs()["x"]).bfloat16()

        lora_model = load_lora_adapters(base, [("a", ckpt)])
        session = OrtInferenceSession(lora_model, ["CUDAExecutionProvider"])
        (got,) = _run_session_to_numpy(
            session, {"x": x.view(torch.int16).numpy().view(ml_dtypes.bfloat16)}
        )

        reference = peft.PeftModel.from_pretrained(_NoConv(torch.bfloat16), ckpt)
        with torch.no_grad():
            expected = reference.eval()(x).float().numpy()
            base_output = _NoConv(torch.bfloat16)(x).float().numpy()
        # The adapter matters, and it is applied as torch peft does, up to bf16 rounding
        assert np.abs(expected - base_output).max() > 0.1
        np.testing.assert_allclose(got.astype(np.float32), expected, atol=0.02)


# ---------------------------------------------------------------------------------------------
# load_lora_adapters_as_inputs (LoRA v1: the LoRA weights are graph inputs)
# ---------------------------------------------------------------------------------------------


class TestLoraInputs:
    """One model whose LoRA weights are inputs, with every adapter's values for them."""

    @pytest.fixture
    def compatible(self, tmp_path):
        """ "a" and "b" target the same layers with r=4; "b" has another alpha and weights."""
        return [
            (
                "a",
                _save_adapter(
                    tmp_path, "a", r=4, alpha=8, targets=ALL_LORA_TARGETS, seed=1
                ),
            ),
            (
                "b",
                _save_adapter(
                    tmp_path, "b", r=4, alpha=16, targets=ALL_LORA_TARGETS, seed=2
                ),
            ),
        ]

    @pytest.mark.parametrize("dynamo", [False, True])
    def test_each_adapter_matches_load_lora_adapters(self, compatible, dynamo):
        """Feeding an adapter gives what ``load_lora_adapters`` bakes in for it, for any alpha."""
        base = _tiny_proto(_base_model(), dynamo, fix_names=dynamo)
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(base, compatible)
        assert list(adapter_inputs) == ["a", "b"]
        for name, ckpt in compatible:
            expected = _run_onnx(
                load_lora_adapters(base, [(name, ckpt)]), _sample_inputs()
            )
            feeds = {**_sample_inputs(), **adapter_inputs[name]}
            for got, ref in zip(_run_onnx(lora_model, feeds), expected):
                np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_graph(self, compatible):
        """A and B are inputs named like the initializers; one ``[1]`` lora_scale is shared.

        The template's values are not in the model, base names are unchanged, the adapter inputs
        name every LoRA input, and the result passes the ONNX checker.
        """
        base = _tiny_proto(_base_model())
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(base, compatible[:1])
        onnx.checker.check_model(lora_model)

        inputs = {
            i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in lora_model.graph.input
        }
        assert inputs["layers.0.q_proj.lora_A.weight"] == (16, 4)
        assert inputs["layers.0.q_proj.lora_B.weight"] == (4, 16)
        assert inputs["features.0.lora_A.weight"] == (4, 3, 3, 3)
        assert inputs["features.0.lora_B.weight"] == (8, 4, 1, 1)
        assert inputs["lora_scale"] == (1,)
        assert not any("lora" in i.name for i in lora_model.graph.initializer)
        assert [o.name for o in lora_model.graph.output] == ["logits", "feat"]

        names = get_lora_tensor_names(lora_model)
        assert names.scales == ("lora_scale",)
        assert (
            len(names.params) == 2 * 6
        )  # q_proj, v_proj per layer, lm_head, features.0
        lora_inputs = {n for n in inputs if "lora" in n}
        assert set(names.params) | {"lora_scale"} == lora_inputs
        assert set(adapter_inputs["a"]) == lora_inputs
        for name, value in adapter_inputs["a"].items():
            assert value.shape == inputs[name]
        branches = _find_lora_branches(lora_model)
        assert {b.scale_weight for b in branches} == {"lora_scale"}
        assert branches[0].a_proj_node == "/layers.0/q_proj/lora_A/MatMul"

    def test_adapter_inputs_are_in_the_model_layout(self, compatible):
        """Linear A and B are transposed into the graph's layout, conv ones are not; lora_scale is
        ``alpha / r``.
        """
        _, adapter_inputs = load_lora_adapters_as_inputs(
            _tiny_proto(_base_model()), compatible
        )
        b_inputs = adapter_inputs["b"]

        ckpt = safetensors_numpy.load_file(
            str(compatible[1][1] / "adapter_model.safetensors")
        )
        key = "base_model.model.layers.0.q_proj.lora_{}.weight"
        np.testing.assert_array_equal(
            b_inputs["layers.0.q_proj.lora_A.weight"], ckpt[key.format("A")].T
        )
        np.testing.assert_array_equal(
            b_inputs["layers.0.q_proj.lora_B.weight"], ckpt[key.format("B")].T
        )
        conv_a = "base_model.model.features.0.lora_A.weight"
        np.testing.assert_array_equal(
            b_inputs["features.0.lora_A.weight"], ckpt[conv_a]
        )
        assert b_inputs["lora_scale"].tolist() == [16 / 4]
        assert adapter_inputs["a"]["lora_scale"].tolist() == [8 / 4]

    def test_zero_adapter_gives_the_base_output(self, compatible):
        """Zero A and B make the model compute the base model."""
        base = _tiny_proto(_base_model())
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(base, compatible[:1])

        zero = {k: np.zeros_like(v) for k, v in adapter_inputs["a"].items()}
        for got, ref in zip(
            _run_onnx(lora_model, {**_sample_inputs(), **zero}),
            _run_onnx(base, _sample_inputs()),
        ):
            np.testing.assert_array_equal(got, ref)

    def test_wrapper_prefix(self, tmp_path):
        """With a base exported inside a wrapper, the adapter inputs still come from its keys."""
        ckpt = _save_adapter(tmp_path, "a", r=4, alpha=8, targets=["q_proj"], seed=1)
        base = _tiny_proto(_Wrapper(_base_model()))
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(base, [("a", ckpt)])
        assert "model.layers.0.q_proj.lora_A.weight" in adapter_inputs["a"]

        expected = _run_onnx(load_lora_adapters(base, [("a", ckpt)]), _sample_inputs())
        got = _run_onnx(lora_model, {**_sample_inputs(), **adapter_inputs["a"]})
        for g, ref in zip(got, expected):
            np.testing.assert_allclose(g, ref, atol=1e-6)

    @pytest.mark.parametrize(
        "r, targets, error",
        [
            pytest.param(2, ALL_LORA_TARGETS, "same rank", id="other-rank"),
            pytest.param(4, ["q_proj"], "same layers", id="fewer-targets"),
        ],
    )
    def test_incompatible_adapters_raise(self, tmp_path, compatible, r, targets, error):
        """Every adapter in the list must have the first one's layers and rank."""
        other = _save_adapter(tmp_path, "other", r=r, alpha=8, targets=targets, seed=3)
        with pytest.raises(ValueError, match=error):
            load_lora_adapters_as_inputs(
                _tiny_proto(_base_model()), [*compatible, ("other", other)]
            )

    @pytest.mark.parametrize(
        "adapters, error", [([], "at least one"), ("duplicate", "unique")]
    )
    def test_bad_adapter_lists_raise(self, compatible, adapters, error):
        if adapters == "duplicate":
            adapters = [compatible[0], ("a", compatible[1][1])]
        with pytest.raises(ValueError, match=error):
            load_lora_adapters_as_inputs(_tiny_proto(_base_model()), adapters)

    def test_pulling_needs_one_scale_value(self, compatible):
        """Several adapters give per-rank scales that one ``lora_scale`` input can't replace."""
        two = load_lora_adapters(_tiny_proto(_base_model()), compatible)
        with pytest.raises(ValueError, match="several values"):
            pull_branch_weights_to_inputs(two)

    def test_pulling_needs_initializer_branches(self, compatible):
        """A base model has nothing to pull; a model with LoRA inputs can't be pulled again."""
        base = _tiny_proto(_base_model())
        with pytest.raises(ValueError, match="no LoRA branches"):
            pull_branch_weights_to_inputs(base)
        lora_model, _ = load_lora_adapters_as_inputs(base, compatible)
        with pytest.raises(ValueError, match="already inputs"):
            pull_branch_weights_to_inputs(lora_model)

    def test_lora_model_cannot_be_the_base(self, compatible):
        """Neither API accepts a model that already has LoRA inputs."""
        lora_model, _ = load_lora_adapters_as_inputs(
            _tiny_proto(_base_model()), compatible
        )
        with pytest.raises(ValueError, match="already contains LoRA branches"):
            load_lora_adapters_as_inputs(lora_model, compatible)
        with pytest.raises(ValueError, match="already contains LoRA branches"):
            load_lora_adapters(lora_model, compatible[:1])

    def test_input_model_is_not_modified(self, compatible):
        base = _tiny_proto(_base_model())
        before = base.SerializeToString()
        load_lora_adapters_as_inputs(base, compatible)
        assert base.SerializeToString() == before

    def test_quantsim_quantizes_the_lora_inputs(self, compatible):
        """In QuantSim, A, B and lora_scale get activation quantizers.

        A and B take the config's param symmetry, as weights of a MatMul/Conv computed at runtime.
        """
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(
            _tiny_proto(_base_model()), compatible
        )
        feeds = {**_sample_inputs(), **adapter_inputs["a"]}

        sim = QuantizationSimModel(
            lora_model, dummy_input=feeds, param_type="int4", activation_type="int16"
        )
        sim.compute_encodings([feeds])

        names = get_lora_tensor_names(lora_model)
        for name in names.params + names.scales:
            assert name not in sim.param_names
            assert sim.qc_quantize_op_dict[name].enabled
        for name in names.params:
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings
