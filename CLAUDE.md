# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMET (AI Model Efficiency Toolkit) is a software toolkit for quantizing trained ML models. It supports PyTorch and ONNX frameworks, providing post-training quantization (PTQ), quantization-aware training (QAT), and model compression techniques.

## Build Commands

`scripts/all/build_and_test.py` is the canonical way to build and test AIMET (used by CI and
documented in `Docs/overview/install/build_from_source.rst`). It manages venv creation, system
dependency installation, and CMake/scikit-build-core configuration -- prefer it over invoking
CMake or pip directly.

```bash
# Build aimet-torch (CPU)
python3 scripts/all/build_and_test.py --task build --package aimet_torch

# Build aimet-torch (GPU/CUDA)
python3 scripts/all/build_and_test.py --task build --package aimet_torch --gpu

# Build aimet-onnx (CPU)
python3 scripts/all/build_and_test.py --task build --package aimet_onnx

# Build aimet-onnx (GPU/CUDA)
python3 scripts/all/build_and_test.py --task build --package aimet_onnx --gpu

# Editable install for local development
python3 scripts/all/build_and_test.py --task build --package aimet_torch --gpu -e

# List every available task
python3 scripts/all/build_and_test.py --task list
```

## Testing

`--task test` builds the package first if needed, then runs its unit test suite:

```bash
# Build (if needed) and run aimet-onnx unit tests
python3 scripts/all/build_and_test.py --task test --package aimet_onnx

# Run tests only, skipping the build step
python3 scripts/all/build_and_test.py --task test --package aimet_torch --only

# Force a rebuild before testing
python3 scripts/all/build_and_test.py --task test --package aimet_torch --force-rebuild
```

For running a single test file/method directly against an already-installed package, pytest
still works as usual:

```bash
python -m pytest TrainingExtensions/torch/test/python/test_<name>.py::TestClass::test_method -v
python -m pytest TrainingExtensions/torch/test/python -m "not cuda"
```

Test directories:
- `TrainingExtensions/torch/test/python/` - PyTorch tests
- `TrainingExtensions/onnx/test/python/` - ONNX tests
- `TrainingExtensions/common/test/` - Common tests

## Linting and Formatting

### Python
- **Formatter**: ruff-format (configured in `.pre-commit-config.yaml`)
- **Linter**: pylint (configured in `.pylintrc`)
- Target Python version: 3.10

### C++
- **Formatter**: clang-format (configured in `.clang-format`)
- Style: Allman braces, 120 column limit, 4 space indent

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit run --all-files
```

## Code Architecture

### Main Python Packages

**aimet_torch** (`TrainingExtensions/torch/src/python/aimet_torch/`)
- Core PyTorch quantization simulation and QAT
- `v2/` - Latest quantization API (QuantizationSimModel)
- `v1/` - Legacy API
- `adaround/` - Adaptive rounding implementation
- `amp/` - Automatic mixed precision
- `experimental/` - Experimental features (omniquant, spinquant, adascale)

**aimet_onnx** (`TrainingExtensions/onnx/src/python/aimet_onnx/`)
- ONNX model quantization and optimization
- `adaround/` - AdaRound for ONNX
- `sequential_mse/` - SeqMSE optimization
- `graph_passes/` - ONNX graph transformations

**aimet_common** (`TrainingExtensions/common/src/python/aimet_common/`)
- Shared utilities across frameworks

### C++ Components

**ModelOptimizations/DlQuantization/**
- Core C++ quantization library with CUDA kernels
- Pybind11 bindings exposed as `libpymo`

### Key Entry Points
- `aimet_torch.quantsim.QuantizationSimModel` - Main quantization simulation class
- `aimet_torch.adaround.adaround_weight.Adaround` - AdaRound API
- `aimet_onnx.quantsim.QuantizationSimModel` - ONNX quantization simulation

## Development Guidelines

### Style Guidelines
- Python: Follow pep8, use snake_case for functions/variables, PascalCase for classes
- C++: Follow Google C++ style guide
- Max line length: 100 characters (Python), 120 characters (C++)

### Commit Requirements
- All commits must be signed with DCO (`git commit -s`)
- Run pre-commit hooks before committing


## Environment Setup

Editable mode (`-e`) is not reliable, so set `PYTHONPATH` explicitly:

```bash
# Set PYTHONPATH after build
export PYTHONPATH=$WORKSPACE/aimet/build/staging/universal/lib/python:$PYTHONPATH

# Or for development from source
export PYTHONPATH=$WORKSPACE/aimet/TrainingExtensions/torch/src/python:$WORKSPACE/aimet/TrainingExtensions/common/src/python:$PYTHONPATH
```

## Generate Documentation
```bash
python3 scripts/all/build_and_test.py --task build --package docs
# Output: wheelhouse/Docs/
```

## Generate Wheel Packages
```bash
python3 scripts/all/build_and_test.py --task build --package aimet_torch
# Output: wheelhouse/
```

## Release Notes

`Docs/release_notes.rst` entries are for end users deciding whether to upgrade, not a commit log. Keep them concise and user-focused:

- **One line per change**, phrased as the user-visible effect, not the implementation. Omit internal rationale, benchmarking detail, review-comment history, or multi-PR narration — that belongs in the commit message, not here.
- **Group by framework**, in this order under each section: ONNX, Torch, Common. Omit a framework subheading entirely if it has no entries for that section.
- **Sections, in order**: `New Features`, `Bug fixes and Improvements`, `Deprecations` (only the sections that have entries).
- Link every entry to its commit(s) via a short 7-char hash reference (e.g. `` (`98862fd`_) ``), with the full `.. _98862fd: https://github.com/qualcomm/aimet/commit/<full-sha>` target listed at the bottom of that release's section. Multiple commits for one line: `` (`abc1234`_, `def5678`_) ``.
- Use double-backtick code formatting for API names, class names, and file names (e.g. ``` ``QuantizationSimModel`` ```).
- Related small commits that land as one logical change (e.g. a feature plus its follow-up fix) collapse into a single bullet with multiple commit links, not one bullet per commit.
- When in doubt about whether a change belongs: ask "would a user upgrading AIMET care about this?" If it's test-only, CI-only, or an internal refactor with no API/behavior change, leave it out.
