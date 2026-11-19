# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-out: let ORT's CUDA EP allocate from torch's caching allocator.

By default every CUDA session grows its own BFC arena, which never hands memory
back to torch and fragments under the per-chunk shape changes of an LLM eval.
For large models (e.g. Qwen3-30B-A3B fp, ~80 GB card) the arena exhausts the
card a few chunks into PPL, whatever ``gpu_mem_limit`` or arena shrinkage is
set to. Routing ORT's allocations through torch's caching allocator (the same
mechanism ORTModule uses) gives ORT and the generator's torch tensors one pool,
so neither can strand memory the other needs.

On by default wherever it can be set up: the first CUDA session build JIT-compiles
a tiny C++ extension (needs ninja, a host compiler and the CUDA headers), and if that
fails, sessions keep ORT's own arena. Numerics are unchanged either way. Opt out with
``GENAILAB_ORT_CUDA_ALLOCATOR=ort``; ``=torch`` makes a failed build an error instead.
"""

import functools
import os

import torch

_ENV_VAR = "GENAILAB_ORT_CUDA_ALLOCATOR"

# Plain-signature wrappers: ORT calls void*(size_t), void(void*), void(). torch's
# emptyCache takes a MempoolId_t argument, so its address must not be handed to
# ORT directly.
_TORCH_ALLOCATOR_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>

static void* torch_alloc(size_t nbytes) { return c10::cuda::CUDACachingAllocator::raw_alloc(nbytes); }
static void torch_free(void* ptr) { c10::cuda::CUDACachingAllocator::raw_delete(ptr); }
static void torch_empty_cache() { c10::cuda::CUDACachingAllocator::emptyCache(); }

int64_t alloc_address() { return reinterpret_cast<int64_t>(&torch_alloc); }
int64_t free_address() { return reinterpret_cast<int64_t>(&torch_free); }
int64_t empty_cache_address() { return reinterpret_cast<int64_t>(&torch_empty_cache); }
"""


@functools.cache
def use_torch_cuda_allocator() -> bool:
    """Whether ORT CUDA sessions should allocate from torch's caching allocator.

    Chosen once per process, so every session and forward agrees, and reported. By
    default, only if the allocator hooks can be built here; with ``=torch``, a failed
    build raises when the session is set up.
    """
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    if value not in ("", "ort", "torch"):
        raise ValueError(
            f"{_ENV_VAR} must be 'torch' (default) or 'ort', got {value!r}"
        )
    if value == "ort":
        ort_arena_reason = f"{_ENV_VAR}=ort"
    elif value == "torch":
        _torch_allocator_addresses()
        ort_arena_reason = None
    else:
        unavailable = _torch_allocator_unavailable()
        ort_arena_reason = (
            f"torch's caching allocator is unavailable: {unavailable}"
            if unavailable
            else None
        )
    if ort_arena_reason:
        print(
            f"  [ort] CUDA sessions allocate from ORT's own arena ({ort_arena_reason})"
        )
        return False
    print(
        "  [ort] CUDA sessions allocate from torch's caching allocator "
        f"({_ENV_VAR}=ort to opt out)"
    )
    return True


def _torch_allocator_unavailable() -> str | None:
    """Why torch's caching allocator cannot back ORT here, or None if it can."""
    if not torch.cuda.is_available():
        return "torch sees no CUDA device"
    # Tried once, by the cached caller: a failed build is not retried by torch, whose
    # next load fails to import the extension instead of reporting what is missing
    try:
        _torch_allocator_addresses()
    except (ImportError, OSError, RuntimeError) as exc:
        # A failed compile appends the whole compiler output
        return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return None


@functools.cache
def _torch_allocator_addresses() -> dict[str, str]:
    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    # torch's caching allocator initializes lazily on first CUDA use; ORT calling
    # raw_alloc before that asserts "Allocator not initialized for device".
    torch.empty(1, device="cuda")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ext = load_inline(
        name="genailab_ort_torch_allocator",
        cpp_sources=_TORCH_ALLOCATOR_SRC,
        functions=["alloc_address", "free_address", "empty_cache_address"],
        extra_include_paths=[os.path.join(CUDA_HOME or "/usr/local/cuda", "include")],
        extra_ldflags=[f"-L{torch_lib}", "-lc10_cuda"],
    )
    return {
        "gpu_external_alloc": str(ext.alloc_address()),
        "gpu_external_free": str(ext.free_address()),
        "gpu_external_empty_cache": str(ext.empty_cache_address()),
    }


def torch_allocator_cuda_options(device_index: int | None) -> dict[str, str]:
    """CUDA EP options that replace ORT's arena with torch's caching allocator.

    One EP-level stream (and one cuBLAS/cuDNN handle) instead of a new stream
    per control-flow subgraph, plus the external allocator hooks.
    """
    # raw_alloc allocates on torch's current device, not the EP's.
    current = torch.cuda.current_device()
    if device_index is not None and device_index != current:
        raise ValueError(
            f"torch's caching allocator needs the ORT device ({device_index}) to be "
            f"torch's current CUDA device ({current}); call torch.cuda.set_device "
            f"first, or set {_ENV_VAR}=ort."
        )
    return {"use_ep_level_unified_stream": "1", **_torch_allocator_addresses()}
