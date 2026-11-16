# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary runtime fixes for torch / onnxscript costs that make dynamo export of
programs with many ``torch.cond`` calls (predicated MoE) superlinear.

Each shim works around one upstream issue and is removed when the fix ships in the
torch / onnxscript versions GenAI Lab supports. Every shim is output-preserving: the
exported program, its ``nn_module_stack`` metadata and the optimized ONNX graph are
identical with and without it. Enter :func:`export_shims` around ``torch.export.export``
and ``torch.onnx.export``.

Each shim declares the half-open torch (or onnxscript) version range it applies to;
outside that range it is a no-op. All ranges are open-ended (``None`` upper bound) until
the upstream fix lands; set the bound to the first fixed release then.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import importlib
import itertools
import sys
import weakref
from collections.abc import Callable, Iterator

import torch
from packaging.version import Version


@dataclasses.dataclass(frozen=True)
class _Shim:
    name: str
    package: str  # "torch" or "onnxscript"
    min_version: str
    max_version: str | None  # first version that has the upstream fix, exclusive
    factory: Callable[[], contextlib.AbstractContextManager]

    def applies(self) -> bool:
        version = Version(importlib.import_module(self.package).__version__).release
        if version < Version(self.min_version).release:
            return False
        return self.max_version is None or version < Version(self.max_version).release


def _rebind(original: Callable, replacement: Callable) -> Callable[[], None]:
    """Point every module-level binding of ``original`` at ``replacement``.

    torch modules import these helpers by name (``from x import f``), so patching the
    defining module alone misses callers that already hold a reference.
    """
    name = original.__name__
    holders = []
    for module in list(sys.modules.values()):
        try:
            if getattr(module, name, None) is original:
                holders.append(module)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    for module in holders:
        setattr(module, name, replacement)

    def undo() -> None:
        for module in holders:
            setattr(module, name, original)

    return undo


@contextlib.contextmanager
def _rebound(original: Callable, replacement: Callable) -> Iterator[None]:
    undo = _rebind(original, replacement)
    try:
        yield
    finally:
        undo()


@contextlib.contextmanager
def _evict_hop_compile_cache() -> Iterator[None]:
    """Drop each HOP compile's dynamo cache entry once the HOP has been traced.

    Non-strict export runs ``torch.compile`` once per ``torch.cond``, always on the
    same wrapper code object. No entry is ever reused (a new backend closure per call
    fails ``BACKEND_MATCH``, then the predicate's dispatch-key guard fails), so entries
    pile up and every cond first fails the guards of all earlier ones.
    """
    from torch._higher_order_ops import utils as hop_utils

    original = hop_utils._hop_compile_and_call

    def hop_compile_and_call(fn, args, kwargs=None):
        try:
            return original(fn, args, kwargs)
        finally:
            torch._dynamo.reset_code(fn.__code__)

    with _rebound(original, hop_compile_and_call):
        yield


@contextlib.contextmanager
def _share_module_stack_paths() -> Iterator[None]:
    """Build ``_ModuleStackTracer``'s module-path maps once per root.

    Every HOP branch gets a sub-tracer on the same ``scope_root``, and each one walks
    ``named_modules()`` twice to rebuild maps that are only written in ``__init__``.
    """
    from torch.fx.experimental import proxy_tensor

    tracer_cls = proxy_tensor._ModuleStackTracer
    original = tracer_cls.__init__
    built: dict[int, tuple] = {}

    def init(self, scope_root, *args, **kwargs):
        cached = built.get(id(scope_root))
        if cached is None or cached[0] is not scope_root:
            original(self, scope_root, *args, **kwargs)
            built[id(scope_root)] = (
                scope_root,
                self.submodule_paths,
                self.module_id_cache,
                self.enable_attr_proxy,
            )
            return
        # Skip both walks; the rest of __init__ (proxy class, autowrap) still runs.
        scope_root.__dict__["named_modules"] = lambda *a, **k: iter(())
        try:
            original(self, scope_root, *args, **kwargs)
        finally:
            del scope_root.__dict__["named_modules"]
        _, self.submodule_paths, self.module_id_cache, self.enable_attr_proxy = cached

    tracer_cls.__init__ = init
    try:
        yield
    finally:
        tracer_cls.__init__ = original


@contextlib.contextmanager
def _index_call_names() -> Iterator[None]:
    """Look up call-name variants (``item``, ``item@1``...) in ``_assign_attr`` by index.

    ``_assign_attr`` regex-matches every child of the target module per assignment, and
    the root holds two subgraphs per cond.
    """
    # torch.export.unflatten is also a function that shadows the submodule.
    unflatten = importlib.import_module("torch.export.unflatten")
    original = unflatten._assign_attr
    index: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def variants(module: torch.nn.Module, item: str) -> list[torch.nn.Module]:
        children = module._modules
        entry = index.get(module)
        if entry is None or len(children) < entry[0]:
            entry = [0, collections.defaultdict(list)]
            index[module] = entry
        if len(children) > entry[0]:
            for key in itertools.islice(children.keys(), entry[0], None):
                entry[1][key].append(key)
                head, sep, tail = key.rpartition("@")
                if sep and tail.isdecimal():
                    entry[1][head].append(key)
            entry[0] = len(children)
        return [children[k] for k in entry[1].get(item, ()) if k in children]

    def assign_attr(from_obj, to_module, target, attr_kind, persistent=True):
        *prefix, field = target.split(".")
        to_modules = {to_module}
        for item in prefix:
            found: set = set()
            for module in to_modules:
                if not hasattr(module, item):
                    setattr(module, item, torch.nn.Module())
                found.update(variants(module, item))
            to_modules = found
        for module in to_modules:
            original(from_obj, module, field, attr_kind, persistent)

    with _rebound(original, assign_attr):
        yield


@contextlib.contextmanager
def _metadata_hook_on_edited_graph() -> Iterator[None]:
    """Register node-metadata hooks on the graph being edited, not every subgraph.

    Every caller of ``_set_node_metadata_hook`` inserts into ``gm.graph`` only, so a
    hook on a subgraph never fires; callers enter it once per inserted assert node.
    """
    from torch._export.passes import _node_metadata_hook as hook_module

    original = hook_module._set_node_metadata_hook

    @contextlib.contextmanager
    def set_node_metadata_hook(gm, f):
        if not callable(f):
            raise AssertionError("node_metadata_hook must be a callable.")
        is_graph_module = isinstance(gm, torch.fx.GraphModule)
        if is_graph_module:
            gm._register_create_node_hook(f)
        try:
            yield
        finally:
            if is_graph_module:
                gm._unregister_create_node_hook(f)

    with _rebound(original, set_node_metadata_hook):
        yield


@contextlib.contextmanager
def _verify_stack_trace_once() -> Iterator[None]:
    """Run ``_verify_stack_trace`` over the top graph once.

    It loops over every submodule but always checks ``graph_module.graph``, so the
    result equals a single pass over that graph.
    """
    from torch.export import _trace

    original = _trace._verify_stack_trace
    spec_violation = _trace.SpecViolationError

    def verify_stack_trace(graph_module):
        for node in graph_module.graph.nodes:
            stack_trace = node.meta.get("stack_trace", None)
            if node.op in ("call_function", "get_attr"):
                if not (stack_trace is None or isinstance(stack_trace, str)):
                    raise spec_violation(
                        f"Node {node} of type {node.op} has invalid stack_trace metadata, "
                        f"expected a string or None but instead found: {stack_trace}"
                    )
            elif node.op in ("placeholder", "output"):
                if stack_trace:
                    raise spec_violation(
                        f"Node {node} of type {node.op} contains stack_trace metadata, "
                        f"expected None but instead found: {stack_trace}"
                    )

    with _rebound(original, verify_stack_trace):
        yield


@contextlib.contextmanager
def _cache_module_qualnames() -> Iterator[None]:
    """Resolve ``nn.Module`` arguments in ``Tracer.create_arg`` through a cached map.

    HOPs register their subgraphs on the tracer root and pass them as arguments, and
    ``create_arg`` finds each one by scanning ``root.named_modules()``.
    """
    from torch.fx import _symbolic_trace

    original = _symbolic_trace.Tracer.create_arg
    cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def lookup(root: torch.nn.Module, module: torch.nn.Module) -> str | None:
        entry = cache.get(root)
        if entry is None:
            names: dict[int, str] = {}
            for name, m in root.named_modules():
                names.setdefault(id(m), name)
            entry = [len(root._modules), names]
            cache[root] = entry
        children = root._modules
        if id(module) not in entry[1] and len(children) > entry[0]:
            # New subgraphs are registered as direct children of root, in order.
            for key, child in itertools.islice(children.items(), entry[0], None):
                if child is not None:
                    for sub, m in child.named_modules(prefix=key):
                        entry[1].setdefault(id(m), sub)
            entry[0] = len(children)
        name = entry[1].get(id(module))
        if name is None:
            return None
        try:
            if root.get_submodule(name) is module:
                return name
        except AttributeError:
            pass
        cache.pop(root, None)  # stale: rebuild on next lookup, fall back now
        return None

    def create_arg(self, a):
        if isinstance(a, torch.nn.Module) and getattr(self, "root", None) is not None:
            name = lookup(self.root, a)
            if name is not None:
                return self.create_node("get_attr", name, (), {})
        return original(self, a)

    _symbolic_trace.Tracer.create_arg = create_arg
    try:
        yield
    finally:
        _symbolic_trace.Tracer.create_arg = original


@contextlib.contextmanager
def _alias_check_without_retrace() -> Iterator[None]:
    """Check cond branches for aliasing/mutation on the already-traced graph.

    cond's functionalization re-traces each branch with ``make_fx`` before analyzing it,
    even when the branch is already a GraphModule whose placeholders carry fake values
    (as in ``run_decompositions``). Analyze that graph directly; fall back otherwise.
    """
    from torch._higher_order_ops import utils as hop_utils

    original = hop_utils.potential_input_alias_or_mutation

    def potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False, **kwargs):
        if not kwargs and isinstance(gm, torch.fx.GraphModule):
            values = [n.meta.get("val") for n in gm.graph.find_nodes(op="placeholder")]
            if (
                values
                and len(values) == len(inputs)
                and all(v is not None for v in values)
            ):
                inp_inp, inp_out, out_out, mutated = (
                    hop_utils.check_input_alias_and_mutation(gm, values)
                )
                return (inp_inp, inp_out, out_out), mutated
        return original(gm, inputs, pre_dispatch, **kwargs)

    with _rebound(original, potential_input_alias_or_mutation):
        yield


@contextlib.contextmanager
def _without_slices_split_rule() -> Iterator[None]:
    """Drop onnxscript's ``SlicesSplit`` rule from the export-time optimizer.

    It is the rewriter's only multi-output pattern, and the matcher pairs every Slice
    with every other Slice (``itertools.product``), so its cost is quadratic in the
    number of Slices. It only rewrites a half/half split, so on these graphs it never
    fires and the optimized graph is unchanged. microsoft/onnxscript#3154.
    """
    from onnxscript import rewriter

    original = rewriter._DEFAULT_REWRITE_RULES
    rewriter._DEFAULT_REWRITE_RULES = tuple(
        rule for rule in original if getattr(rule, "name", None) != "SlicesSplit"
    )
    try:
        yield
    finally:
        rewriter._DEFAULT_REWRITE_RULES = original


#: In application order. min_version is the oldest release the shim was verified on.
_SHIMS = (
    _Shim("evict_hop_compile_cache", "torch", "2.14", None, _evict_hop_compile_cache),
    _Shim("share_module_stack_paths", "torch", "2.14", None, _share_module_stack_paths),
    _Shim("index_call_names", "torch", "2.14", None, _index_call_names),
    _Shim(
        "metadata_hook_on_edited_graph",
        "torch",
        "2.14",
        None,
        _metadata_hook_on_edited_graph,
    ),
    _Shim("verify_stack_trace_once", "torch", "2.14", None, _verify_stack_trace_once),
    _Shim("cache_module_qualnames", "torch", "2.14", None, _cache_module_qualnames),
    _Shim(
        "alias_check_without_retrace",
        "torch",
        "2.14",
        None,
        _alias_check_without_retrace,
    ),
    _Shim(
        "without_slices_split_rule",
        "onnxscript",
        "0.7",
        None,
        _without_slices_split_rule,
    ),
)


def active_shims() -> list[str]:
    """Names of the shims that apply to the installed torch / onnxscript."""
    return [shim.name for shim in _SHIMS if shim.applies()]


@contextlib.contextmanager
def export_shims() -> Iterator[None]:
    """Apply every shim that matches the installed versions for the duration."""
    with contextlib.ExitStack() as stack:
        for shim in _SHIMS:
            if shim.applies():
                stack.enter_context(shim.factory())
        yield
