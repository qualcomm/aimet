# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMET (AI Model Efficiency Toolkit) is a software toolkit for quantizing trained ML models. It supports PyTorch and ONNX frameworks, providing post-training quantization (PTQ), quantization-aware training (QAT), and model compression techniques.

## Build Commands

### Standard Build (CMake)
```bash
cd aimet
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_ONNX=OFF
make -j8
make install
```

### Build with pip (AIMET 2.0)
```bash
CMAKE_ARGS='-DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_ONNX=OFF' python3 -m pip install --no-build-isolation -e .
```

### CMake Options
- `-DENABLE_CUDA=ON/OFF` - Enable CUDA support
- `-DENABLE_TORCH=ON/OFF` - Enable PyTorch variant
- `-DENABLE_ONNX=ON/OFF` - Enable ONNX variant
- `-DENABLE_TESTS=ON/OFF` - Enable test building

## Testing

### Run All Tests
```bash
cd build
make test
# OR
ctest -V
```

### Run Single Test File with pytest
```bash
python -m pytest TrainingExtensions/torch/test/python/test_<name>.py -v
```

### Run Tests without CUDA
```bash
python -m pytest TrainingExtensions/torch/test/python -m "not cuda"
```

### Run Specific Test
```bash
python -m pytest TrainingExtensions/torch/test/python/test_<name>.py::TestClass::test_method -v
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

```bash
# Set PYTHONPATH after build
export PYTHONPATH=$WORKSPACE/aimet/build/staging/universal/lib/python:$PYTHONPATH

# Or for development from source
export PYTHONPATH=$WORKSPACE/aimet/TrainingExtensions/torch/src/python:$WORKSPACE/aimet/TrainingExtensions/common/src/python:$PYTHONPATH
```

## Generate Documentation
```bash
cd build
make doc
# Output: build/staging/universal/Docs/
```

## Generate Wheel Packages
```bash
cd build
make packageaimet
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
