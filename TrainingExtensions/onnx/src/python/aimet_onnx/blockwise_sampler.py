# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Sample the inputs of a sequence of blocks, running one block of graph at a time.

For blockwise techniques (e.g. AdaScale) that optimize the blocks of a model in order, each on
the inputs the blocks before it produce. Re-running blocks ``0..k`` to get block ``k``'s inputs
(:class:`~aimet_onnx.adascale.activation_sampler.ActivationSampler`) costs O(N²) blocks of
compute, disk and device memory over a model. This sampler instead keeps, per calibration
sample, the block-boundary tensors that later blocks still read, and produces each block's
inputs by running only the graph between the cached tensors and the ones it needs.

A :class:`SamplingPlan` analyzes the blocks of a graph once, and a :class:`BlockwiseSampler`
samples one model with it. Models whose block tensors have the same names, e.g. a sim and its
unquantized copy, can share a plan.

A block is any subgraph, given by the tensors it reads and writes, so a block can be a decoder
layer or a group of independent layers. The caller may change an initializer, or the encodings
of its quantizer, only while optimizing the block that owns it: the first block whose subgraph
reads it. A cached tensor is therefore only reused while every initializer it was computed from
is final: a tensor computed while sampling block ``t`` is final iff it reads no initializer
owned by block ``t`` or later. Tensors that are not final are recomputed each time they are
needed, never cached.

This is the ONNX counterpart of :class:`aimet_torch.blockwise_sampler.BlockwiseSampler`. That
one chains each block's output into the next block's args. This one plans from the graph which
tensors each block reads, so it also handles reads across blocks and reads from control-flow
bodies, and it checks that cached tensors stay valid instead of requiring that weights don't
change. It keeps the cache in memory, with no disk offload, and doesn't resume from a
checkpoint.
"""

import gc
import os
import tempfile
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx_ir

from aimet_onnx.common.utils import AimetLogger  # pylint: disable=import-error
from aimet_onnx.ir_utils import body_initializers, outer_scope_values, walk_back
from aimet_onnx.utils import OrtInferenceSession, _run_session_to_numpy

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

# A cached tensor whose next use is more than one block away is evicted (and recomputed when
# needed) once the cache grows past this multiple of the largest block's inputs.
_MAX_CACHE_RATIO = 1.5

_NO_BLOCK = -1


class SamplingPlan:
    """Which tensors each block reads, and when a tensor computed for a block can be reused.

    Built once, before any block is optimized, from the graph the blocks are defined on.

    :param graph: Graph the blocks are defined on. Must be topologically sorted.
    :param block_inputs: Per block, the ordered input names of its subgraph (e.g. as returned
        by :func:`~aimet_onnx.adascale.model_converter.required_block_inputs`).
    :param block_outputs: Per block, the names of its outputs.

    :ivar owner: Initializer name -> first block whose subgraph reads it, i.e. the only block
        that may update it.
    :ivar block_initializers: Per block, the initializers its subgraph reads.
    :ivar dependency: Value name -> the highest owner among the initializers it is computed
        from (``-1`` if it reads none).
    :ivar last_use: Cacheable value name -> the last block that reads it, as a block input or
        to compute a later block's input.
    """

    def __init__(
        self,
        graph: onnx_ir.Graph,
        block_inputs: Sequence[Sequence[str]],
        block_outputs: Sequence[Sequence[str]],
    ):
        if len(block_inputs) != len(block_outputs):
            raise ValueError(
                f"Got inputs of {len(block_inputs)} blocks but outputs of "
                f"{len(block_outputs)}"
            )
        self.block_inputs = [list(names) for names in block_inputs]
        self.block_outputs = [list(names) for names in block_outputs]
        name_to_value = onnx_ir.convenience.create_value_mapping(
            graph, include_subgraphs=False
        )
        self.graph_inputs = {value.name for value in graph.inputs}

        self.owner: Dict[str, int] = {}
        self.block_initializers: List[List[str]] = []
        for idx, (inputs, outputs) in enumerate(
            zip(self.block_inputs, self.block_outputs)
        ):
            _, initializers = walk_back(graph, name_to_value, inputs, outputs)
            self.block_initializers.append(initializers)
            for name in initializers:
                self.owner.setdefault(name, idx)

        self.dependency = self._compute_dependency(graph)

        # Simulate sampling with nothing ever evicted to find each tensor's last use. Evicting
        # a tensor after its last use cannot change a later walk: no later walk reached it.
        self.last_use: Dict[str, int] = {}
        cached: set[str] = set()
        for idx, inputs in enumerate(self.block_inputs):
            needed = [name for name in inputs if name not in self.graph_inputs]
            missing = [name for name in needed if name not in cached]
            used = [name for name in needed if name in cached]
            if missing:
                required, _ = walk_back(graph, name_to_value, cached, missing)
                used += [name for name in required if name in cached]
            for name in used:
                self.last_use[name] = idx
            for name in missing:
                if self.is_final(name, idx):
                    cached.add(name)
                    self.last_use[name] = idx

    def _compute_dependency(self, graph: onnx_ir.Graph) -> Dict[str, int]:
        """Value name -> the highest owner among the initializers it is computed from."""
        dependency: Dict[str, int] = {}

        def _values_read(node: onnx_ir.Node) -> List[onnx_ir.Value]:
            """Values of ``graph`` that ``node`` reads, including from its control-flow
            bodies."""
            return [inp for inp in node.inputs if inp is not None] + outer_scope_values(
                node, graph
            )

        def _owner_of(value: onnx_ir.Value) -> int:
            """The highest owner among the initializers ``value`` is computed from."""
            if value.is_initializer():
                return self.owner.get(value.name, _NO_BLOCK)
            return dependency.get(value.name, _NO_BLOCK)

        def _owner_of_bodies(node: onnx_ir.Node) -> int:
            """The highest owner among the initializers that ``node``'s control-flow bodies
            declare themselves, e.g. MoE expert weights."""
            return max(
                (self.owner.get(name, _NO_BLOCK) for name in body_initializers(node)),
                default=_NO_BLOCK,
            )

        # One pass in graph order, which visits every producer before its consumers
        visited = set()
        for node in graph:
            values = _values_read(node)
            for value in values:
                producer = value.producer()
                if producer is not None and producer not in visited:
                    raise ValueError(
                        "The graph must be topologically sorted, but node "
                        f"{node.name!r} reads {value.name!r} before its producer "
                        f"{producer.name!r}"
                    )
            visited.add(node)
            owner = max(
                [_owner_of(value) for value in values] + [_owner_of_bodies(node)]
            )
            for out in node.outputs:
                dependency[out.name] = owner
        return dependency

    def is_final(self, name: str, idx: int) -> bool:
        """Whether ``name``, computed while sampling block ``idx``, is safe to reuse later."""
        return self.dependency.get(name, _NO_BLOCK) < idx


class _ActivationCache:
    """Cached tensors of one model: tensor name -> its array for every calibration sample."""

    def __init__(self, log_prefix: str = ""):
        self._arrays: Dict[str, List[Optional[np.ndarray]]] = {}
        self._log_prefix = log_prefix

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def names(self) -> List[str]:
        """Names of the cached tensors."""
        return list(self._arrays)

    def get(self, name: str, sample_idx: int) -> np.ndarray:
        """The cached array of ``name`` for sample ``sample_idx``."""
        return self._arrays[name][sample_idx]

    def add(self, name: str, arrays: List[np.ndarray]):
        """Cache the array of ``name`` for every sample."""
        self._arrays[name] = arrays
        _logger.debug(
            "%scached %s (%.2f MB)",
            self._log_prefix,
            name,
            sum(array.nbytes for array in arrays) / 1024**2,
        )

    def release(self, names: Collection[str], sample_idx: int):
        """Free the arrays of ``names`` for sample ``sample_idx`` ahead of :meth:`evict`, so
        memory is freed as each sample advances. Until then the cache is inconsistent."""
        for name in names:
            self._arrays[name][sample_idx] = None

    def evict(self, names: Collection[str]):
        """Drop ``names`` from the cache."""
        for name in names:
            if self._arrays.pop(name, None) is not None:
                _logger.debug("%sevicted %s", self._log_prefix, name)

    def clear(self):
        """Drop every cached tensor."""
        if self._arrays:
            _logger.debug("%scleared %s", self._log_prefix, self.names())
        self._arrays.clear()

    def nbytes(self) -> int:
        """Bytes held by the cache."""
        return sum(
            array.nbytes
            for arrays in self._arrays.values()
            for array in arrays
            if array is not None
        )


class BlockwiseSampler:
    """Produces the block inputs of one model by advancing a cache of block-boundary tensors.

    :param model: Model to sample. Read again on every call, so a model the caller changes
        (e.g. a sim) must carry the weights and encodings set while optimizing every block so
        far.
    :param plan: Blocks to sample, and when their tensors can be cached. Every block input it
        names must be a tensor of ``model``.
    :param inputs: Calibration samples, each a dict of every graph input.
    :param providers: ORT providers to sample with.
    :param may_update: Whether the caller may still change an initializer of ``model`` (or its
        encodings). Checked for the initializers a block reads but an earlier block owns,
        which break the reuse of cached tensors computed from them. None means it never does,
        e.g. for a model the caller doesn't change.
    :param label: Prefix for this sampler's log lines, to tell apart samplers that share a
        plan. Has no effect on sampling.
    """

    def __init__(
        self,
        model: onnx_ir.Model,
        plan: SamplingPlan,
        inputs: Sequence[Dict[str, np.ndarray]],
        providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
        may_update: Optional[Callable[[str], bool]] = None,
        label: str = "",
    ):
        self._model = model
        self._plan = plan
        self._inputs = inputs
        self._providers = providers
        self._may_update = may_update
        self._log_prefix = f"{label}: " if label else ""
        self._name_to_value = onnx_ir.convenience.create_value_mapping(
            model.graph, include_subgraphs=False
        )
        absent = sorted(
            {name for names in plan.block_inputs for name in names}
            - self._name_to_value.keys()
        )
        if absent:
            raise ValueError(
                f"{self._log_prefix}The model lacks block inputs the plan names: "
                f"{absent[:5]}"
            )
        self._cache = _ActivationCache(self._log_prefix)
        self._restart = False
        # Block input name -> bytes it takes across all samples
        self._nbytes: Dict[str, int] = {}

    def cache_nbytes(self) -> int:
        """Bytes held by the cache."""
        return self._cache.nbytes()

    def sample_block(self, idx: int) -> List[List[np.ndarray]]:
        """Return block ``idx``'s inputs.

        Same format as sampling with :class:`ActivationSampler`: one list per calibration
        sample, ordered like ``plan.block_inputs[idx]``. Graph inputs are the caller's arrays,
        passed through by reference. Blocks may be skipped, but must be sampled in order.
        """
        if self._restart:
            self._cache.clear()
            self._restart = False

        stale = self._stale_initializers(idx)
        if stale:
            _logger.warning(
                "%sBlock %d can update initializers %s owned by an earlier block, so "
                "blockwise sampling restarts from the graph inputs after this block.",
                self._log_prefix,
                idx,
                stale[:5],
            )
            self._restart = True

        samples = self._advance(idx)
        self._limit_cache(idx, samples)
        return samples

    def _stale_initializers(self, idx: int) -> List[str]:
        """
        Initializers block ``idx`` reads that an earlier block owns and the caller may still
        change. Changing one would make the cached tensors computed from it stale.

        For AdaScale this takes a weight two blocks read whose quantizer the earlier block left
        unfrozen, e.g. because that block reads the weight only through a Gather, so it has no
        layer that trains it.
        """
        if self._may_update is None:
            return []
        return [
            name
            for name in self._plan.block_initializers[idx]
            if self._plan.owner[name] < idx and self._may_update(name)
        ]

    def _advance(self, idx: int) -> List[List[np.ndarray]]:
        plan, cache = self._plan, self._cache
        names = plan.block_inputs[idx]
        missing = [
            name
            for name in names
            if name not in plan.graph_inputs and name not in cache
        ]
        keep = [
            name
            for name in missing
            if plan.is_final(name, idx) and plan.last_use.get(name, idx) > idx
        ]

        session, required, tempdir = None, [], None
        if missing:
            required, _ = walk_back(
                self._model.graph, self._name_to_value, cache.names(), missing
            )
            tempdir = tempfile.TemporaryDirectory()
            session = _build_session(
                self._model,
                self._feed(0, required),
                missing,
                self._providers,
                tempdir.name,
            )
        # Released as each sample advances, so the cache never holds two blocks' worth
        evict = [
            name
            for name in cache.names()
            if name not in keep and plan.last_use.get(name, idx) <= idx
        ]

        samples = []
        kept: Dict[str, List[np.ndarray]] = {name: [] for name in keep}
        try:
            for sample_idx, sample in enumerate(self._inputs):
                computed = {}
                if missing:
                    feed = self._feed(sample_idx, required)
                    computed = dict(
                        zip(missing, _run_session_to_numpy(session, feed, missing))
                    )
                block_sample = []
                for name in names:
                    if name in plan.graph_inputs:
                        block_sample.append(sample[name])
                    elif name in computed:
                        block_sample.append(computed[name])
                    else:
                        block_sample.append(cache.get(name, sample_idx))
                for name in keep:
                    kept[name].append(computed[name])
                cache.release(evict, sample_idx)
                samples.append(block_sample)
            cache.evict(evict)
            for name, arrays in kept.items():
                cache.add(name, arrays)
        except BaseException:
            # The samples before the failing one already released what they evict; restart
            # from the graph inputs
            cache.clear()
            raise
        finally:
            del session
            gc.collect()
            if tempdir is not None:
                tempdir.cleanup()
        return samples

    def _feed(self, sample_idx: int, names: Collection[str]) -> Dict[str, np.ndarray]:
        return {
            name: self._cache.get(name, sample_idx)
            if name in self._cache
            else self._inputs[sample_idx][name]
            for name in names
        }

    def _limit_cache(self, idx: int, samples: List[List[np.ndarray]]):
        """Keep the cache within a multiple of the largest block's inputs, which is what the
        full-prefix sampler holds at its peak."""
        if not samples:
            return
        plan = self._plan
        for name, arr in zip(plan.block_inputs[idx], samples[0]):
            if name not in plan.graph_inputs:
                self._nbytes[name] = arr.nbytes * len(samples)
        if not self._nbytes:
            return
        # Inputs of blocks not sampled yet are assumed to be the average size seen so far
        default = sum(self._nbytes.values()) / len(self._nbytes)
        peak = max(
            sum(
                self._nbytes.get(name, default)
                for name in names
                if name not in plan.graph_inputs
            )
            for names in plan.block_inputs
        )
        cached = self._cache.nbytes()
        if cached <= _MAX_CACHE_RATIO * peak:
            return
        long_lived = [
            name
            for name in self._cache.names()
            if plan.last_use.get(name, idx) > idx + 1
        ]
        if not long_lived:
            return
        _logger.warning(
            "%sBlockwise sampling cache (%.2f GB) exceeds %.1fx the largest block's "
            "inputs (%.2f GB) after block %d; evicting %s, which will be recomputed when "
            "needed.",
            self._log_prefix,
            cached / 1024**3,
            _MAX_CACHE_RATIO,
            peak / 1024**3,
            idx,
            long_lived,
        )
        self._cache.evict(long_lived)


def _build_session(
    model: onnx_ir.Model,
    feed: Dict[str, np.ndarray],
    outputs: List[str],
    providers,
    tempdir: str,
) -> OrtInferenceSession:
    """Session over the part of ``model`` that computes ``outputs`` from the inputs in ``feed``."""
    graph = onnx_ir.convenience.extract(model.graph, list(feed), outputs)
    for value in graph.inputs:
        # Intermediate tensors carry no type unless the model has value_info for them
        if value.type is None:
            array = feed[value.name]
            value.type = onnx_ir.TensorType(onnx_ir.DataType.from_numpy(array.dtype))
            value.shape = onnx_ir.Shape([None] * array.ndim)
    subgraph_model = onnx_ir.Model(
        graph, ir_version=model.ir_version, functions=list(model.functions.values())
    )
    path = os.path.join(tempdir, "subgraph.onnx")
    onnx_ir.save(subgraph_model, path, external_data="subgraph.data")
    return OrtInferenceSession(path, providers)
