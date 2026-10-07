# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma4's audio tower through the strict dynamo export path."""

import pytest

torch = pytest.importorskip("torch")
modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

from transformers.models.gemma4.configuration_gemma4 import (  # noqa: E402
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
)

from GenAILab.qai_hub_lm.models.gemma4 import (  # noqa: E402
    Gemma4_VLM,
    _index_based_bidirectional_mask,
)


def _model(attn_implementation="sdpa"):
    text = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=32,
    )
    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        subsampling_conv_channels=[16, 8],
        output_proj_dims=32,
    )
    config = Gemma4Config(text_config=text, audio_config=audio)
    config.audio_config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return modeling_gemma4.Gemma4ForConditionalGeneration(config).eval()


def _inputs(model, padded):
    features, mask = Gemma4_VLM.get_sample_audio_inputs(model.config)
    features = torch.randn_like(features)
    if padded:
        mask = mask.clone()
        mask[:, mask.shape[1] // 2 :] = False
    return features, mask


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
@pytest.mark.parametrize("padded", [False, True])
def test_index_based_mask_matches_stock(attn_implementation, padded):
    """The audio tower builds the same mask, and so the same outputs, as stock."""
    model = _model(attn_implementation)
    tower = model.model.audio_tower
    features, mask = _inputs(model, padded)

    masks = {}
    stock = modeling_gemma4.create_bidirectional_mask
    for name, create in (("stock", stock), ("index", _index_based_bidirectional_mask)):

        def spy(*args, _create=create, _name=name, **kwargs):
            masks[_name] = _create(*args, **kwargs)
            return masks[_name]

        modeling_gemma4.create_bidirectional_mask = spy
        try:
            with torch.no_grad():
                tower(features, mask)
        finally:
            modeling_gemma4.create_bidirectional_mask = stock

    assert masks["stock"] is not None
    assert torch.equal(masks["index"], masks["stock"])


def test_wrapper_restores_the_stock_mask_builder():
    """The swap is scoped to the audio call: the vision tower uses it too."""
    model = _model()
    wrapper = Gemma4_VLM.build_audio_wrapper(model)
    with torch.no_grad():
        wrapper(*_inputs(model, padded=True))
    assert (
        modeling_gemma4.create_bidirectional_mask is not _index_based_bidirectional_mask
    )


def test_strict_export_matches_stock(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        ONNX_OPSET_VERSION,
        _dynamo_export,
    )

    model = _model()
    wrapper = Gemma4_VLM.build_audio_wrapper(model).eval()
    path = str(tmp_path / "audio.onnx")
    with torch.no_grad():
        _dynamo_export(
            wrapper,
            _inputs(model, padded=False),
            path,
            input_names=Gemma4_VLM.get_component_input_names("audio"),
            output_names=Gemma4_VLM.get_component_output_names(
                "audio", config=model.config
            ),
            opset_version=ONNX_OPSET_VERSION,
        )

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    # The validity mask is a graph input, so padding must be honoured at run time
    for padded in (False, True):
        features, mask = _inputs(model, padded)
        got = session.run(None, dict(zip(names, (features.numpy(), mask.numpy()))))
        with torch.no_grad():
            # Stock model, called directly: not through the wrapper's mask swap
            audio_out = model.model.audio_tower(features, mask)
            embeddings = model.model.embed_audio(
                inputs_embeds=audio_out.last_hidden_state
            )
        valid = audio_out.attention_mask
        torch.testing.assert_close(
            torch.from_numpy(got[0]), embeddings, rtol=0, atol=1e-5
        )
        assert torch.equal(torch.from_numpy(got[1]), valid)
