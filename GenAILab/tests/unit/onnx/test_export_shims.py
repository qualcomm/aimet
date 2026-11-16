# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""The export shims must not change what gets exported, only how fast."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

import torch.nn as nn  # noqa: E402

from GenAILab.qai_hub_lm.backends.onnx import export_shims  # noqa: E402


class _CondExperts(nn.Module):
    """Two layers of four experts, each expert under its own torch.cond."""

    def __init__(self, d=8, experts=4, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(experts))
            for _ in range(layers)
        )

    def forward(self, x):
        for block in self.blocks:
            out = torch.zeros_like(x)
            for e, expert in enumerate(block):

                def taken(acc, h, _expert=expert):
                    return acc + _expert(h)

                def skipped(acc, h):
                    return acc.clone()

                out = torch.cond(
                    x.flatten()[e % x.numel()] > 0, taken, skipped, (out, x)
                )
            x = x + out
        return x


def _code(prog):
    text = prog.graph_module.print_readable(print_output=False, include_stride=True)
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _stacks(prog):
    return [
        node.meta.get("nn_module_stack")
        for module in prog.graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    ]


def _export(model, x, shims):
    with torch.no_grad(), export_shims.export_shims() if shims else _null():
        prog = torch.export.export(model, (x,), strict=False)
        decomposed = prog.run_decompositions()
    return prog, decomposed


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_every_shim_is_active_on_supported_versions():
    # The suite runs on torch >= 2.14 and onnxscript >= 0.7, where none is fixed yet.
    assert export_shims.active_shims() == [shim.name for shim in export_shims._SHIMS]


def test_exported_program_unchanged():
    torch.manual_seed(0)
    model, x = _CondExperts().eval(), torch.randn(3, 8)
    stock, stock_dec = _export(model, x, shims=False)
    shimmed, shimmed_dec = _export(model, x, shims=True)
    assert _code(shimmed) == _code(stock)
    assert _stacks(shimmed) == _stacks(stock)
    assert _code(shimmed_dec) == _code(stock_dec)
    torch.testing.assert_close(shimmed.module()(x), stock.module()(x), rtol=0, atol=0)


def test_shims_are_restored_on_exit():
    from torch._higher_order_ops import utils as hop_utils
    from torch.fx.experimental import proxy_tensor

    before = (
        hop_utils._hop_compile_and_call,
        hop_utils.potential_input_alias_or_mutation,
        proxy_tensor._ModuleStackTracer.__init__,
        torch.fx._symbolic_trace.Tracer.create_arg,
    )
    with export_shims.export_shims():
        assert hop_utils._hop_compile_and_call is not before[0]
    after = (
        hop_utils._hop_compile_and_call,
        hop_utils.potential_input_alias_or_mutation,
        proxy_tensor._ModuleStackTracer.__init__,
        torch.fx._symbolic_trace.Tracer.create_arg,
    )
    assert after == before


def test_shim_outside_version_range_is_skipped(monkeypatch):
    shim = export_shims._Shim("future", "torch", "99.0", None, lambda: None)
    assert not shim.applies()
    fixed = export_shims._Shim("fixed", "torch", "2.0", "2.1", lambda: None)
    assert not fixed.applies()


def test_alias_check_still_rejects_aliasing_branches():
    """The direct check must keep raising where the re-trace did."""

    def taken(x):
        return x  # returns its input: an input-output alias

    def skipped(x):
        return x.clone()

    class Aliasing(nn.Module):
        def forward(self, x):
            return torch.cond(x.sum() > 0, taken, skipped, (x,))

    x = torch.randn(4)
    with pytest.raises(Exception):
        with export_shims.export_shims():
            torch.export.export(Aliasing(), (x,), strict=False).run_decompositions()
