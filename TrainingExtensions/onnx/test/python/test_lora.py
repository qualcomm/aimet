# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
"""
End-to-end LoRA quantization of a tiny Llama: the LoRA v1, v2 and v3 flows.

Every flow starts the same way: export the base model to ONNX, then ``load_lora_adapters``
attaches PEFT checkpoints as one concatenated branch per target layer (``test_onnx_peft.py``
covers that step).
The flows differ in which quantization encodings are shared and which are recomputed:

=======  ===================  ===================================================================
Flow     Adapters at a time   Encodings
=======  ===================  ===================================================================
LoRA v1  one                  one set for every adapter, base and LoRA tensors alike
LoRA v2  one                  a full set per adapter; base encodings are recomputed too
LoRA v3  one or more          base encodings computed once and frozen; LoRA encodings computed
                              per concurrency (a set of adapters that are active together)
=======  ===================  ===================================================================

v1 and v2 switch between compatible adapters (same rank and target layers): every adapter gives
the same graph, and only the LoRA weights change. v3 also combines incompatible ones. v1 builds
its model with ``load_lora_adapters_as_inputs`` instead: the LoRA weights are graph inputs, so
one model and one sim serve every adapter, switched through those inputs.

QuantSim places its quantizers when it is built, so the LoRA tensors get quantizers only if
``load_lora_adapters`` runs before ``QuantizationSimModel``. v3 therefore uses two sims, one on
the base model and one on the LoRA model. They are joined by tensor names:
``load_lora_adapters`` keeps every base tensor name, so the base encodings load into the LoRA sim
as they are.
"""

import copy
import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest

peft = pytest.importorskip("peft")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors")

from aimet_onnx import int4, int8, int16
from aimet_onnx.experimental.lora import (
    disable_lora_quantizers,
    enable_lora_calibration,
    freeze_base_activation_quantizers,
    freeze_base_model,
    freeze_base_param_quantizers,
    get_lora_encodings,
    get_lora_node_names_for_exclusion,
    set_lora_bitwidth,
    set_lora_encodings,
    unfreeze_lora_quantizers,
)
from aimet_onnx.peft import (
    get_lora_tensor_names,
    load_lora_adapters,
    load_lora_adapters_as_inputs,
)
from aimet_onnx.peft.peft_utils import _find_lora_branches
from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    fix_node_names_pass,
)
from aimet_onnx.qc_quantize_op import _EncodingMismatchInfo
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    load_encodings_to_sim,
)

SEQ_LEN = 8
VOCAB_SIZE = 128


# ---------------------------------------------------------------------------------------------
# Base model and adapters
# ---------------------------------------------------------------------------------------------


def _tiny_llama():
    """The same 2-layer Llama every call (fixed seed), with 4 query and 2 key/value heads."""
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=False,
    )
    return transformers.LlamaForCausalLM(config).eval()


class _LogitsOnly(torch.nn.Module):
    """``(input_ids, attention_mask) -> logits``: the prefill graph a deployment exports.

    The mask is passed in 4-D so HF uses it as is instead of building one. Holding the HF model as
    ``self.model`` gives every graph path a leading ``model.`` that the adapters (saved from the
    bare HF model) don't have; ``load_lora_adapters`` detects that shared prefix, as for real
    exports.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


@pytest.fixture(scope="module", params=["torchscript", "dynamo"])
def base_model(request, tmp_path_factory):
    """The tiny Llama in ONNX (TorchScript or dynamo export); output ``logits``.

    Dynamo exports also need ``fix_node_names_pass``.
    """
    dynamo = request.param == "dynamo"
    path = tmp_path_factory.mktemp("base") / "base.onnx"
    inputs = (
        torch.zeros(1, SEQ_LEN, dtype=torch.int64),
        torch.zeros(1, 1, SEQ_LEN, SEQ_LEN),
    )
    try:
        with torch.no_grad():
            torch.onnx.export(
                _LogitsOnly(_tiny_llama()),
                inputs,
                str(path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=18,
                dynamo=dynamo,
            )
    except Exception as e:  # pylint: disable=broad-except
        if dynamo:
            pytest.skip(f"dynamo export unavailable: {e}")
        raise
    model = onnx.load(str(path))
    return fix_node_names_pass(model) if dynamo else model


def _save_adapter(path, r, alpha, targets, seed, b_std):
    """Save a PEFT checkpoint for the tiny Llama with random A and B (peft zero-inits B).

    ``b_std`` sets the adapter's strength. Like a trained adapter, it should change the layer
    outputs by a fraction of the base output: v3 keeps the base encodings computed without LoRA, so
    a far stronger adapter would push the activations out of their range.
    """
    model = peft.get_peft_model(
        _tiny_llama(),
        peft.LoraConfig(r=r, lora_alpha=alpha, target_modules=targets),
    )
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if "lora_A" in name:
            torch.nn.init.normal_(param, std=0.1)
        elif "lora_B" in name:
            torch.nn.init.normal_(param, std=b_std)
    # peft also saves the base lm_head when it is a target unless told not to
    model.save_pretrained(path, save_embedding_layers=False)
    return path


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    """Three PEFT checkpoints for the tiny Llama.

    "adapter_1" and "adapter_2" are compatible (r=4 on q_proj and v_proj): either one gives the
    same graph, so v1 and v2 switch between them. "adapter_2" has larger weights, so a range that
    only saw "adapter_1" does not cover it. "adapter_3" (r=2 on v_proj, o_proj and lm_head) is
    compatible with neither and can only join them in a v3 concurrency.
    """
    root = tmp_path_factory.mktemp("adapters")
    qv = ["q_proj", "v_proj"]
    return {
        "adapter_1": _save_adapter(
            root / "adapter_1", r=4, alpha=8, targets=qv, seed=1, b_std=0.03
        ),
        "adapter_2": _save_adapter(
            root / "adapter_2", r=4, alpha=8, targets=qv, seed=2, b_std=0.06
        ),
        "adapter_3": _save_adapter(
            root / "adapter_3",
            r=2,
            alpha=4,
            targets=["v_proj", "o_proj", "lm_head"],
            seed=3,
            b_std=0.03,
        ),
    }


# ---------------------------------------------------------------------------------------------
# Data, QuantSim and encodings
# ---------------------------------------------------------------------------------------------


def _batches(seed, num_batches=4):
    """Random token batches with a causal mask (-1e4 above the diagonal)."""
    rng = np.random.default_rng(seed)
    mask = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e4, np.float32), k=1)[None, None]
    return [
        {
            "input_ids": rng.integers(0, VOCAB_SIZE, (1, SEQ_LEN)),
            "attention_mask": mask,
        }
        for _ in range(num_batches)
    ]


@pytest.fixture(scope="module")
def calibration_data():
    """Batches to compute encodings with."""
    return _batches(seed=0)


@pytest.fixture(scope="module")
def eval_data():
    """Batches to compare models on, apart from the calibration batches."""
    return _batches(seed=1)


def _propagate_lora_add(sim, lora_tensors):
    """Make each merged output reuse its base layer's output quantizer, as a v3 user would.

    Neither the merge Add nor the ops after it then need encodings of their own. Run it before
    loading the base encodings: the Add's quantizer carries settings QuantSim derives from its
    consumers (e.g. attention makes its V input symmetric), and ``_merge_constraints`` copies them
    onto the base quantizer so the loaded encodings still match.
    """
    shared = {}
    for base_output, merged_output in lora_tensors.merges:
        base_qtzr = sim.qc_quantize_op_dict[base_output]
        # pylint: disable-next=protected-access
        base_qtzr._merge_constraints(sim.qc_quantize_op_dict[merged_output])
        shared[merged_output] = base_qtzr
    sim.set_quantizers(shared)


def _export_encodings(sim, path, prefix):
    """Export only the encodings of ``sim`` and return them as tensor name -> encoding."""
    sim.export(str(path), prefix, export_model=False, encoding_version="1.0.0")
    return _read_encodings(path / f"{prefix}.encodings")


def _read_encodings(path):
    """Tensor name -> encoding of a 1.0.0 encodings file."""
    with open(path, encoding="utf-8") as f:
        encodings = json.load(f)
    entries = encodings["param_encodings"] + encodings["activation_encodings"]
    return {e["name"]: e for e in entries}


def _tensor_names(model):
    """Names of every node output and initializer in ``model``."""
    outputs = {o for node in model.graph.node for o in node.output}
    return outputs | {init.name for init in model.graph.initializer}


def _sqnr_db(expected, actual):
    """Signal-to-quantization-noise ratio of ``actual`` against ``expected``, in dB."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    return 10 * np.log10(np.sum(expected**2) / np.sum((expected - actual) ** 2))


def _logits(session, data):
    """Logits of every batch in ``data``."""
    return [session.run(None, batch)[0] for batch in data]


def _float_session(model):
    """An onnxruntime session on ``model``, without any quantization."""
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


# ---------------------------------------------------------------------------------------------
# The flows
# ---------------------------------------------------------------------------------------------


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestLoraFlows:
    def test_float_lora_model_matches_torch_peft(self, base_model, adapters, eval_data):
        """Before any quantization: the float LoRA model gives the same logits as torch peft.

        Uses the "adapter_1" + "adapter_3" concurrency: adapters of different ranks on overlapping
        layers, grouped-query attention, the ``model.`` prefix shift, and ``lm_head`` feeding the
        graph output.
        """
        concurrency = [(name, adapters[name]) for name in ("adapter_1", "adapter_3")]
        lora_model = load_lora_adapters(base_model, concurrency)

        reference = peft.PeftModel.from_pretrained(
            _tiny_llama(), adapters["adapter_1"], adapter_name="adapter_1"
        )
        reference.load_adapter(adapters["adapter_3"], adapter_name="adapter_3")
        reference.base_model.set_adapter([name for name, _ in concurrency])
        session = _float_session(lora_model)
        for batch in eval_data:
            with torch.no_grad():
                expected = _LogitsOnly(reference.eval())(
                    torch.from_numpy(batch["input_ids"]),
                    torch.from_numpy(batch["attention_mask"]),
                ).numpy()
            (actual,) = session.run(None, batch)
            np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_lora_v1_one_encoding_set_for_all_adapters(
        self, base_model, adapters, tmp_path, calibration_data
    ):
        """LoRA v1: the LoRA weights are model inputs, and every adapter uses one set of encodings.

        1. ``load_lora_adapters_as_inputs`` adds the branches once, with A, B and ``lora_scale``
           as graph inputs, checks "adapter_2" is compatible with "adapter_1" (the template), and
           returns each adapter's values for those inputs.
        2. Zeros for every LoRA input are the "base" adapter.
        3. One sim. Its LoRA weight quantizers are activation quantizers. ``set_lora_bitwidth``
           makes the LoRA weights 8-bit (the sim's activations are 16-bit) and disables the
           ``lora_scale`` quantizer, as for every scale. One ``compute_encodings`` that switches
           adapters per batch collects every adapter's range.
        4. Every LoRA weight range covers both adapters.
        5. One encodings file, with the LoRA weights as input encodings and no ``lora_scale``.
        """
        # --- 1. Add the LoRA inputs once ---
        compatible = ("adapter_1", "adapter_2")
        lora_model, adapter_inputs = load_lora_adapters_as_inputs(
            base_model, [(name, adapters[name]) for name in compatible]
        )

        # --- 2. Inputs per adapter, plus the zero "base" adapter ---
        adapter_inputs["base"] = {
            k: np.zeros_like(v) for k, v in adapter_inputs["adapter_1"].items()
        }

        # --- 3. One sim, one calibration over every adapter ---
        sim = QuantizationSimModel(
            copy.deepcopy(lora_model),
            param_type=int4,
            activation_type=int16,
        )
        lora_tensors = get_lora_tensor_names(lora_model)
        for name in lora_tensors.params:
            assert name not in sim.param_names  # graph inputs: activation quantizers
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings
        set_lora_bitwidth(sim, lora_tensors, param_type="int8", activation_type="int16")
        for name in lora_tensors.params:
            assert sim.qc_quantize_op_dict[name].bitwidth == 8
        assert not sim.qc_quantize_op_dict["lora_scale"].enabled

        def calibrate(session):
            for batch in calibration_data:
                # every adapter per batch: a deterministic stand-in for random.choice
                for name in adapter_inputs:
                    session.run(None, {**batch, **adapter_inputs[name]})

        sim.compute_encodings(calibrate)

        # --- 4. The shared ranges cover both adapters ---
        # Nothing is clipped: quantizing either adapter's weights moves no value by more than a step
        for name in lora_tensors.params:
            quantizer = sim.qc_quantize_op_dict[name]
            step = max(e.delta for e in quantizer.get_encodings())
            for adapter in compatible:
                value = adapter_inputs[adapter][name]
                assert (
                    np.abs(quantizer.quantize_dequantize(value) - value).max() <= step
                )

        # --- 5. One encodings file ---
        encodings = _export_encodings(sim, tmp_path, "lora_v1")
        assert set(lora_tensors.params) <= set(encodings)
        assert "lora_scale" not in encodings

    def test_lora_v2_full_encoding_set_per_adapter(
        self, base_model, adapters, tmp_path, calibration_data
    ):
        """LoRA v2: each adapter gets its own full set of encodings, base tensors included.

        1. Per adapter: attach it, build a sim, and ``set_lora_bitwidth`` (int8 weights, int16
           activations; the weights stay per-channel symmetric like the base ones). Calibrate
           everything and export its encodings.
        2. Both adapters give the same quantizers (one graph; weights and encodings differ).
        3. Base encodings differ between adapters too, e.g. the logits, which LoRA changes.
        """
        # --- 1. Attach, calibrate and export per adapter ---
        encodings = {}
        for adapter in ("adapter_1", "adapter_2"):
            lora_model = load_lora_adapters(base_model, [(adapter, adapters[adapter])])
            sim = QuantizationSimModel(
                copy.deepcopy(lora_model),
                param_type=int4,
                activation_type=int16,
            )
            lora_tensors = get_lora_tensor_names(lora_model)
            set_lora_bitwidth(
                sim, lora_tensors, param_type="int8", activation_type="int16"
            )
            for name in lora_tensors.params:
                quantizer = sim.qc_quantize_op_dict[name]
                assert quantizer.bitwidth == 8
                assert quantizer.use_symmetric_encodings

            sim.compute_encodings(calibration_data)
            encodings[adapter] = _export_encodings(sim, tmp_path, f"lora_v2_{adapter}")

        # --- 2. Same quantizers ---
        first, second = encodings["adapter_1"], encodings["adapter_2"]
        assert first.keys() == second.keys()

        # --- 3. LoRA and base encodings are both per adapter ---
        lora_b_weight = next(n for n in first if n.endswith(".lora_B.weight"))
        assert first[lora_b_weight] != second[lora_b_weight]
        assert first["logits"] != second["logits"]

    def test_lora_v3_frozen_base_lora_encodings_per_concurrency(
        self, base_model, adapters, tmp_path, calibration_data, eval_data
    ):
        """LoRA v3: base encodings computed once and frozen; LoRA encodings per concurrency.

        1. Base sim: quantize the base model alone and export it (float model + encodings).
           Techniques such as AdaScale change base weights at this step, so the adapters are
           attached to the exported model rather than the original.
        2. Per concurrency ("adapter_1" alone, then "adapter_1" + "adapter_3"):
           ``load_lora_adapters`` on the exported base. "adapter_1" and "adapter_3" share v_proj,
           which gets one branch of rank 4 + 2.
        3. LoRA sim with the same config. ``_propagate_lora_add`` gives each merged
           output its base layer's output quantizer (user code, from ``lora_tensors.merges``),
           so the graph has the base model's quantizers in the same places with the same
           settings. Load the base encodings frozen: each one matches its quantizer exactly, and
           only the tensors ``load_lora_adapters`` added are left without one.
           ``set_lora_bitwidth`` sets the LoRA bit widths (int8 weights, int16 activations) and
           disables the scale quantizers.
        4. Calibrate: only the LoRA quantizers update. The adapters keep their effect: the frozen
           base ranges and the shared merge quantizers don't clip it away.
        5. Export the encodings per concurrency; the base encodings in them are unchanged.
        6. The QDQ model is the final artifact: it runs in onnxruntime, and it can't be a base
           for ``load_lora_adapters`` again.
        """
        # --- 1. Base sim, exported once ---
        base_sim = QuantizationSimModel(
            copy.deepcopy(base_model),
            param_type=int4,
            activation_type=int16,
        )
        base_sim.compute_encodings(calibration_data)
        base_sim.export(str(tmp_path), "base", encoding_version="1.0.0")
        exported_base = onnx.load(str(tmp_path / "base.onnx"))
        base_encodings = _read_encodings(tmp_path / "base.encodings")

        concurrencies = {
            "adapter_1": ["adapter_1"],
            "adapter_1_3": ["adapter_1", "adapter_3"],
        }
        for concurrency, adapter_names in concurrencies.items():
            # --- 2. Attach the concurrency's adapters ---
            lora_model = load_lora_adapters(
                exported_base, [(name, adapters[name]) for name in adapter_names]
            )

            # --- 3. LoRA sim with the base encodings frozen ---
            lora_tensors = get_lora_tensor_names(lora_model)
            sim = QuantizationSimModel(
                copy.deepcopy(lora_model),
                param_type=int4,
                activation_type=int16,
            )
            _propagate_lora_add(sim, lora_tensors)
            mismatches = load_encodings_to_sim(
                sim,
                str(tmp_path / "base.encodings"),
                strict=False,
                allow_overwrite=False,
                disable_missing_quantizers=False,
            )
            # Every base encoding matches its quantizer's settings exactly. The only quantizers
            # without one are on tensors load_lora_adapters added.
            assert all(
                m
                == _EncodingMismatchInfo(
                    m.quantizer_name, enabled_mismatch=(True, False)
                )
                for m in mismatches
            )
            without_encoding = {m.quantizer_name for m in mismatches}
            added = _tensor_names(lora_model) - _tensor_names(exported_base)
            assert without_encoding <= added
            set_lora_bitwidth(
                sim, lora_tensors, param_type="int8", activation_type="int16"
            )

            # --- 4. Calibrate the LoRA quantizers ---
            sim.compute_encodings(calibration_data)
            # The sim is much closer to the LoRA model than the base model is
            sqnr = _sqnr_db(
                _logits(_float_session(lora_model), eval_data),
                _logits(sim.session, eval_data),
            )
            lora_effect = _sqnr_db(
                _logits(_float_session(lora_model), eval_data),
                _logits(_float_session(exported_base), eval_data),
            )
            assert sqnr > lora_effect + 6.0, (
                f"v3 {concurrency}: SQNR {sqnr:.1f} dB, LoRA effect {lora_effect:.1f} dB"
            )

            # --- 5. Per-concurrency encodings; base encodings unchanged ---
            encodings = _export_encodings(sim, tmp_path, f"lora_v3_{concurrency}")
            for name, encoding in base_encodings.items():
                assert encodings[name] == encoding
            # Every added tensor has an encoding now, except the scales set_lora_bitwidth disabled
            scales = set(lora_tensors.scales)
            assert without_encoding - scales <= set(encodings)
            assert not scales & set(encodings)

            # --- 6. QDQ model ---
            qdq_model = sim.to_onnx_qdq()
            # Same encodings, so only rounding differences between the two runtimes
            qdq_sqnr = _sqnr_db(
                _logits(sim.session, eval_data),
                _logits(_float_session(qdq_model), eval_data),
            )
            assert qdq_sqnr >= 40.0
            with pytest.raises(ValueError, match="already contains LoRA branches"):
                load_lora_adapters(qdq_model, [("adapter_1", adapters["adapter_1"])])


# ---------------------------------------------------------------------------------------------
# adapter_utils helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lora_model(base_model, adapters):
    """The "adapter_1" + "adapter_3" concurrency.

    Branches of rank 4 (q_proj), 2 (o_proj, lm_head) and 4 + 2 (v_proj).
    """
    return load_lora_adapters(
        base_model, [(n, adapters[n]) for n in ("adapter_1", "adapter_3")]
    )


@pytest.fixture
def sim(lora_model):
    """A fresh LoRA sim with the base output quantizers shared, as in the v3 flow."""
    sim = QuantizationSimModel(
        copy.deepcopy(lora_model),
        param_type=int4,
        activation_type=int16,
    )
    _propagate_lora_add(sim, get_lora_tensor_names(lora_model))
    return sim


@pytest.fixture
def names(lora_model):
    """The LoRA tensors in the form the helpers take."""
    return get_lora_tensor_names(lora_model)


@pytest.fixture
def lora_quantizers(names):
    """The LoRA weight and activation quantizers the helpers configure (not the scales)."""
    return set(names.quantizer_names)


@pytest.fixture
def base_quantizers(sim, names, lora_quantizers):
    """The enabled quantizers that are not LoRA ones, merged outputs included."""
    lora = lora_quantizers | set(names.scales)
    return {
        n for n, q in sim.qc_quantize_op_dict.items() if q.enabled and n not in lora
    }


def _quantizer_encodings(sim, names):
    return {n: sim.qc_quantize_op_dict[n].export_encodings("1.0.0") for n in names}


def _frozen(sim):
    return {n for n, q in sim.qc_quantize_op_dict.items() if q.is_encoding_frozen()}


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestLoraAdapterQuantization:
    def test_set_lora_bitwidth(self, sim, names, lora_quantizers, base_quantizers):
        """LoRA weights and activations get their own bit widths; scale quantizers are disabled.

        The base quantizers keep the sim's (int4 weights, int16 activations).
        """
        base_bitwidths = {
            n: sim.qc_quantize_op_dict[n].bitwidth for n in base_quantizers
        }
        count = set_lora_bitwidth(sim, names, param_type="int16", activation_type=int8)

        assert count == len(lora_quantizers)
        for name in names.params:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16
        for name in names.activations:
            assert sim.qc_quantize_op_dict[name].bitwidth == 8
        for name in names.scales:
            assert not sim.qc_quantize_op_dict[name].enabled
        assert base_bitwidths == {
            n: sim.qc_quantize_op_dict[n].bitwidth for n in base_bitwidths
        }

    def test_set_lora_bitwidth_raises_on_frozen_lora_quantizer(self, sim, names):
        """A frozen quantizer keeps its encoding, so changing its bit width would be ignored."""
        sim.qc_quantize_op_dict[names.params[0]].freeze_encodings()
        with pytest.raises(RuntimeError, match="frozen"):
            set_lora_bitwidth(sim, names, int16, int8)

    def test_freeze_base_param_quantizers(self, sim, names, base_quantizers):
        """Every enabled base weight quantizer is frozen; nothing else is."""
        count = freeze_base_param_quantizers(sim, names)

        expected = set(sim.param_names) & base_quantizers
        assert count == len(expected) > 0
        assert _frozen(sim) == expected

    def test_freeze_base_activation_quantizers(self, sim, names, base_quantizers):
        """Every enabled base activation quantizer is frozen; nothing else is, scales included.

        The merged outputs share their base layer's output quantizer, so they are base
        quantizers too.
        """
        count = freeze_base_activation_quantizers(sim, names)

        expected = set(sim.activation_names) & base_quantizers
        assert {merged for _, merged in names.merges} <= expected
        assert count == len(expected) > 0
        assert _frozen(sim) == expected

    def test_freeze_base_model(
        self, sim, names, lora_quantizers, base_quantizers, calibration_data
    ):
        """Calibration after ``freeze_base_model`` computes only the LoRA encodings."""
        sim.compute_encodings(calibration_data)
        base_encodings = _quantizer_encodings(sim, base_quantizers)
        lora_encodings = _quantizer_encodings(sim, lora_quantizers)
        freeze_base_model(sim, names)

        sim.compute_encodings(_batches(seed=2))

        assert _quantizer_encodings(sim, base_encodings) == base_encodings
        # The LoRA activations saw new data, so their ranges moved
        assert _quantizer_encodings(sim, lora_encodings) != lora_encodings

    def test_unfreeze_lora_quantizers(
        self, sim, names, lora_quantizers, base_quantizers, calibration_data
    ):
        """Frozen LoRA quantizers are recomputed after unfreezing; base ones stay frozen."""
        sim.compute_encodings(calibration_data)
        for name in base_quantizers:
            sim.qc_quantize_op_dict[name].freeze_encodings()
        for name in lora_quantizers:
            sim.qc_quantize_op_dict[name].freeze_encodings()
        base_frozen = _frozen(sim) - lora_quantizers

        count = unfreeze_lora_quantizers(sim, names)

        assert count == len(lora_quantizers)
        assert _frozen(sim) == base_frozen
        for name in lora_quantizers:
            assert not sim.qc_quantize_op_dict[name].is_initialized()
        sim.compute_encodings(calibration_data)
        for name in lora_quantizers:
            assert sim.qc_quantize_op_dict[name].is_initialized()

    def test_get_lora_encodings(self, sim, names, lora_quantizers, calibration_data):
        """One encoding per LoRA weight and activation, the same as the sim exports."""
        sim.compute_encodings(calibration_data)

        encodings = get_lora_encodings(sim, names)

        assert encodings.keys() == lora_quantizers
        for name, encoding in encodings.items():
            assert encoding == sim.qc_quantize_op_dict[name].export_encodings("1.0.0")

    def test_get_lora_encodings_raises_before_calibration(self, sim, names):
        with pytest.raises(RuntimeError, match="has no encoding"):
            get_lora_encodings(sim, names)

    def test_set_lora_encodings(
        self,
        sim,
        names,
        lora_model,
        tmp_path,
        lora_quantizers,
        calibration_data,
        eval_data,
    ):
        """Saved LoRA encodings, set on a fresh sim with the same base encodings, reproduce it.

        This is how a concurrency's LoRA encodings are reused without calibrating again.
        """
        sim.compute_encodings(calibration_data)
        lora_encodings = get_lora_encodings(sim, names)
        # The fresh sim gets the base encodings only
        sim.export(str(tmp_path), "sim", export_model=False, encoding_version="1.0.0")
        with open(tmp_path / "sim.encodings", encoding="utf-8") as f:
            base_encodings = json.load(f)
        for key in ("param_encodings", "activation_encodings"):
            base_encodings[key] = [
                e for e in base_encodings[key] if e["name"] not in lora_quantizers
            ]
        restored = QuantizationSimModel(
            copy.deepcopy(lora_model),
            param_type=int4,
            activation_type=int16,
        )
        _propagate_lora_add(restored, names)
        load_encodings_to_sim(
            restored, base_encodings, strict=False, disable_missing_quantizers=False
        )

        count = set_lora_encodings(restored, lora_encodings)

        assert count == len(lora_encodings)
        assert lora_quantizers <= _frozen(restored)
        for expected, actual in zip(
            _logits(sim.session, eval_data), _logits(restored.session, eval_data)
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_set_lora_encodings_raises_on_unknown_name(self, sim):
        with pytest.raises(ValueError, match="not found in sim"):
            set_lora_encodings(sim, {"not_a_tensor": {}})

    def test_get_lora_node_names_for_exclusion(self, lora_model, names):
        """Exactly the branch nodes: not the merge Add, which carries the base output on."""
        excluded = get_lora_node_names_for_exclusion(lora_model, names)

        assert set(excluded) == {
            n
            for b in _find_lora_branches(lora_model)
            for n in (b.a_proj_node, b.scale_node, b.b_proj_node)
        }

    def test_disable_lora_quantizers(
        self, sim, names, lora_quantizers, base_quantizers
    ):
        """Only the LoRA quantizers are disabled."""
        count = disable_lora_quantizers(sim, names)

        assert count == len(lora_quantizers)
        for name in lora_quantizers | set(names.scales):
            assert not sim.qc_quantize_op_dict[name].enabled
        assert {
            n for n, q in sim.qc_quantize_op_dict.items() if q.enabled
        } == base_quantizers

    def test_enable_lora_calibration(self, sim, names, lora_quantizers):
        """Undoes ``disable_lora_quantizers``, at the LoRA bit widths; scales stay disabled."""
        disable_lora_quantizers(sim, names)

        count = enable_lora_calibration(sim, names, int16, int8)

        assert count == len(lora_quantizers)
        for name in names.params:
            assert sim.qc_quantize_op_dict[name].enabled
            assert sim.qc_quantize_op_dict[name].bitwidth == 16
        for name in names.activations:
            assert sim.qc_quantize_op_dict[name].enabled
            assert sim.qc_quantize_op_dict[name].bitwidth == 8
        for name in names.scales:
            assert not sim.qc_quantize_op_dict[name].enabled
