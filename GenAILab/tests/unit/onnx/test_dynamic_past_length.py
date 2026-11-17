# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""A dynamic-length backbone export accepts a non-empty KV cache."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

from GenAILab.qai_hub_lm.backends.onnx.export_utils import (  # noqa: E402
    export_sample_sequence_length,
)

CONTEXT_LENGTH = 32


class TestExportSampleSequenceLength:
    @pytest.mark.parametrize(
        "sequence_length, expected",
        [
            ([32, 16, 1], 30),  # longest = context: would leave no past entry
            ([31, 1], 30),  # would leave one past entry, also specialized
            ([16, 8, 1], 16),  # leaves past entries: unchanged
            ([30, 1], 30),
            (32, 32),  # a static export has no past axis to keep dynamic
            ([32], 32),
        ],
    )
    def test_leaves_two_past_entries_for_dynamic_exports(
        self, sequence_length, expected
    ):
        """
        Given: The configured sequence length(s) and the context length
        When: The export sample's sequence length is chosen
        Then: A dynamic export (several lengths) leaves at least two past entries, so that
              torch.export doesn't specialize the past length; anything else is unchanged
        """
        assert (
            export_sample_sequence_length(sequence_length, CONTEXT_LENGTH) == expected
        )


def _export(tmp_path, sample_sequence_length):
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        ONNX_OPSET_VERSION,
        _dynamo_export,
        dynamic_axes_to_dynamic_shapes,
    )
    from GenAILab.qai_hub_lm.models.base import LLM
    from GenAILab.qai_hub_lm.models.utils.exportable import (
        ONNXExportableModuleWithCache,
    )
    from GenAILab.qai_hub_lm.models.utils.layer_cache import (
        build_layer_cache_descriptors,
    )

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2 * CONTEXT_LENGTH,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    descriptors = build_layer_cache_descriptors(model.config)
    input_names = LLM.get_backbone_input_names(descriptors)
    wrapped = ONNXExportableModuleWithCache(model, input_names=input_names).eval()
    sample = LLM.get_sample_backbone_inputs(
        wrapped,
        CONTEXT_LENGTH,
        sample_sequence_length,
        layer_cache_descriptors=descriptors,
    )
    path = str(tmp_path / "backbone.onnx")
    with torch.no_grad():
        _dynamo_export(
            wrapped,
            sample,
            path,
            input_names=list(input_names),
            output_names=list(LLM.get_backbone_output_names(descriptors)),
            opset_version=ONNX_OPSET_VERSION,
            dynamic_shapes=dynamic_axes_to_dynamic_shapes(
                input_names, LLM.get_backbone_dynamic_axes(descriptors)
            ),
        )
    return path, sample


def _feed(session, sample, sequence_length):
    np = pytest.importorskip("numpy")
    past = CONTEXT_LENGTH - sequence_length
    feed = {}
    for meta, ref in zip(session.get_inputs(), sample):
        shape = list(ref.shape)
        if meta.name in ("input_ids", "position_ids"):
            shape[1] = sequence_length
        elif "mask" in meta.name:
            shape[2] = sequence_length
        elif meta.name.endswith("_in"):
            shape[2] = past
        array = np.zeros(shape, dtype=ref.numpy().dtype)
        if meta.name == "position_ids":
            array[:] = np.arange(past, past + sequence_length)
        feed[meta.name] = array
    return feed


@pytest.mark.parametrize("sequence_length", [1, 8, 16, CONTEXT_LENGTH])
def test_dynamic_export_runs_with_any_past_length(tmp_path, sequence_length):
    """
    Given: A backbone exported for sequence lengths [context, 16, 1], as the configs that
           list the context length among their sequence lengths do
    When: The graph runs with a KV cache of any length, including none
    Then: It runs; the past length isn't pinned to 0 (which made MMLU's chunked prefill fail
          with "invalid dimensions for input past_key_0_in")
    """
    ort = pytest.importorskip("onnxruntime")
    path, sample = _export(
        tmp_path,
        export_sample_sequence_length([CONTEXT_LENGTH, 16, 1], CONTEXT_LENGTH),
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    logits = session.run(None, _feed(session, sample, sequence_length))[0]
    assert logits.shape[1] == sequence_length
