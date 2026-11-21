# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""The recipe cache restores a real aimet-onnx sim, and a failed read leaves it as is.

Loading reads every file of the entry before the sim is touched, so a truncated
or half-purged entry is a miss rather than a sim with some weights replaced.
"""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from GenAILab.bench.profiler import RecipeStepStats  # noqa: E402
from GenAILab.bench.recipe_cache import RecipeCache  # noqa: E402


class _Profiler:
    elapsed_ms = 1.0


def _tiny_sim(seed):
    from aimet_onnx.quantsim import QuantizationSimModel

    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).eval()
    x = torch.randn(1, 8)
    buf = io.BytesIO()
    torch.onnx.export(
        model, (x,), buf, input_names=["x"], output_names=["y"], opset_version=17
    )
    buf.seek(0)
    sim = QuantizationSimModel(
        onnx.load_model(buf),
        dummy_input={"x": x.numpy()},
        providers=["CPUExecutionProvider"],
    )
    sim.compute_encodings(lambda session: session.run(None, {"x": x.numpy()}))
    return sim, x.numpy()


def _weights(sim):
    return {
        init.name: onnx.numpy_helper.to_array(init).copy()
        for init in sim.model.model.graph.initializer
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(RecipeCache, "env_hash", "envhash")
    monkeypatch.setattr(
        "GenAILab.bench.recipe_cache.convert_gpu_meter_to_dict",
        lambda profiler, **kwargs: {"elapsed_ms": profiler.elapsed_ms},
    )
    return RecipeCache(tmp_path / "recipe")


def _save(cache, sim):
    step = RecipeStepStats("SeqMSE", {}, "", {}, _Profiler())
    cache.save("c1", "base", [step], sim, "onnx", model_id="m")


def test_round_trip_restores_weights_and_output(cache):
    source, x = _tiny_sim(seed=0)
    _save(cache, source)
    target, _ = _tiny_sim(seed=1)
    assert not np.allclose(
        target.session.run(None, {"x": x})[0], source.session.run(None, {"x": x})[0]
    )

    stats = cache.load("c1", target, "onnx")

    assert [s.recipe_name for s in stats] == ["SeqMSE"]
    for name, value in _weights(source).items():
        np.testing.assert_array_equal(_weights(target)[name], value)
    np.testing.assert_allclose(
        target.session.run(None, {"x": x})[0],
        source.session.run(None, {"x": x})[0],
        rtol=1e-6,
    )


def test_unreadable_encodings_leave_the_sim_untouched(cache):
    source, x = _tiny_sim(seed=0)
    _save(cache, source)
    (cache.cache_dir / "c1" / "state" / "model.encodings").write_text("{trunc")
    target, _ = _tiny_sim(seed=1)
    before_weights = _weights(target)
    before_out = target.session.run(None, {"x": x})[0]

    with pytest.raises(ValueError):
        cache.load("c1", target, "onnx")

    for name, value in before_weights.items():
        np.testing.assert_array_equal(_weights(target)[name], value)
    np.testing.assert_array_equal(target.session.run(None, {"x": x})[0], before_out)
