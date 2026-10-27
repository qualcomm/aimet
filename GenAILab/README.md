# AIMET GenAI Laboratory

AIMET's GenAI laboratory provides an easy way to evaluate large models against quantization techniques provided by both AIMET-ONNX and AIMET-Torch. Define your experiment in a YAML config and the framework handles model loading, quantization, evaluation, and optional ONNX export.

## Prerequisites

- Either AIMET-Torch or AIMET-ONNX
- HuggingFace transformers and datasets
- See `requirements.txt` for full list

## Quick Start

```bash
# Run with Torch
python -m GenAILab --framework torch --config GenAILab/example_config.yaml

# Run with ONNX
python -m GenAILab --framework onnx --config GenAILab/example_config.yaml

# Run both Torch and ONNX sequentially
python -m GenAILab --framework both --config GenAILab/example_config.yaml
```

### Runner Options

```bash
# Cache control
python -m GenAILab --framework torch --config cfg.yaml \
    --fp-cache-dir /path/to/fp/cache       # Custom FP results cache directory
    --clear-fp-cache                        # Clear FP cache before running
    --model-cache-dir /path/to/model/cache  # Custom ONNX model cache directory
    --clear-model-cache                     # Clear model cache before running
    --force-export                          # Force re-export of all artifacts

# Custom output directories
python -m GenAILab --framework torch --config cfg.yaml \
    --export-dir /path/to/exports \
    --results-dir /path/to/results

# Remote execution via GitHub Actions
python -m GenAILab --framework torch --config cfg.yaml --online
python -m GenAILab --framework torch --config cfg.yaml --online --wait
python -m GenAILab --framework torch --config cfg.yaml --download <run_id>
```

Every experiment in the config runs even if an earlier one fails; the run exits non-zero if any failed. For local debugging:

```bash
python -m GenAILab --framework onnx --config cfg.yaml \
    --fail-fast                             # Stop after the first failed experiment
    --pdb                                   # Drop into the debugger on failure
```

## Config File Format

Configs use YAML, and a single file can contain multiple experiments separated by `---`. The full schema — every field, accepted values, defaults, validation rules, and the list of currently-registered recipes/datasets/metrics/adaptations — is documented in [CONFIG.md](CONFIG.md).

## Artifacts and Caching

All artifacts are stored under `GenAILab/artifacts/` by default. Each directory can be overridden via CLI flags.

```
GenAILab/artifacts/
├── results/                   # Profiling output
│   ├── profiling_data.json    # Hierarchical results (model_type → entries)
│   └── profiling_data.csv     # Flat format for database ingestion
├── exports/                   # ONNX export output
│   └── {model_slug}_{timestamp}/
│       ├── config.yaml        # Copy of the input config
│       ├── backbone/
│       │   ├── model.onnx     # Exported ONNX model
│       │   ├── model.data     # External tensor data
│       │   └── model.encodings # Quantization encodings
│       ├── visual/            # (vision models only)
│       │   ├── model.onnx
│       │   └── model.data
│       ├── audio/             # (audio models only)
│       │   ├── model.onnx
│       │   └── model.data
│       └── embedding.pth      # (multi-modal only)
└── cache/
    ├── fp/                    # FP model output cache
    ├── recipe/                # Recipe chain checkpoint cache
    └── model/                 # ONNX model export cache
```

### Results

Each experiment appends to `profiling_data.json` and `profiling_data.csv` in the results directory. The JSON format is hierarchical (keyed by model type), while the CSV is a flat table with JSON-encoded columns suitable for PostgreSQL ingestion. Both files use file locking for process-safe concurrent writes.

Results include model configuration, precision settings, per-step resource utilization (GPU memory, elapsed time), metric scores, and environment metadata (CUDA version, GPU name, pip freeze, git SHA).

### Metric Versions & Reproducibility

Every recorded metric result carries a `scoring_version` alongside its `result` value (`EvaluationMetric.SCORING_VERSION`, threaded through `MetricResult`). This exists because a metric's *scoring semantics* — its prompt template, tokenization, dataset filtering, or aggregation — can change without its name changing. A `48.2` and a `27.4` on `MMMU` are meaningless to compare if they were computed under different scoring semantics; the version makes that explicit instead of silent.

- **Bump `SCORING_VERSION`** on a metric class whenever you change any of the above. Add a one-line changelog entry to the class docstring explaining what changed. [`tests/unit/helpers/test_metric_versions.py`](tests/unit/helpers/test_metric_versions.py) pins each versioned metric's scoring contract with a golden fingerprint — it will fail if you change scoring behavior without bumping the version, as a forcing function.
- **Absent `scoring_version` means version 1.** Results recorded before this field existed (or metrics that have never changed scoring semantics) are implicitly version 1 — no backfill needed.
- **`print_summary` gates comparability.** When a table would mix versions for the same metric, it prints `MMMU (MIXED VERSIONS!)` with a loud warning instead of rendering the numbers side by side as if they were comparable.

**Posted guidance — comparing vs. reproducing:**

- **Comparing a result to a new one:** only compare results with the *same* `scoring_version`. If your baseline was recorded under an older version, re-run *that baseline* under the current version — do not recompute the entire result history. Comparisons are pairwise and local; treat them that way.
- **Reproducing an old number exactly:** old-version results are archival, not regenerable from the current codebase — the old scoring implementation is not kept around once superseded (this repo intentionally does not carry multiple live scorers per metric). To get the exact old scoring code, find the commit before the `SCORING_VERSION` bump (check the metric's docstring changelog for context) and run from there.
- **Dataset drift caveat:** some datasets (e.g. `MMMU/MMMU` on HuggingFace) are not pinned to a fixed revision, so even re-running the same code and version is not guaranteed to reproduce a historical number byte-for-byte. `scoring_version` guards against comparing across different *scoring code*; it does not guard against upstream dataset changes.

### FP Cache

Caches full-precision model outputs (e.g., MMLU logits) so distance metrics can compare quantized vs. FP without re-running the FP model. Entries are stored as `.pt` files (torch tensors) keyed by a hash of the model configuration and metric name. The cache is loaded lazily on first access.

| Flag | Effect |
|------|--------|
| `--fp-cache-dir <path>` | Custom FP cache directory |
| `--clear-fp-cache` | Wipe cache before running |

### Recipe Cache

Caches intermediate quantization state after cacheable recipe steps (currently `SeqMSE` and `AdaScale`). The cache key is a Merkle chain hash: each step's hash extends the previous one, incorporating the recipe name, kwargs, dataset config, model ID, precision, and an environment hash (library versions + AIMET source hashes). This means shared recipe prefixes across different configs hit the same cache entry.

On a cache hit, the recipe chain skips already-computed steps and loads the saved checkpoint directly into the quantsim model. After each new cacheable step, the result is saved for future runs.

| Flag | Effect |
|------|--------|
| `--recipe-cache-dir <path>` | Custom recipe cache directory |
| `--no-recipe-cache` | Disable recipe caching entirely |
| `--clear-recipe-cache` | Wipe cache before running |

### Model Cache (ONNX)

Caches exported ONNX models so that repeated ONNX evaluation runs don't require re-export from Torch. The cache key is a hash of model ID, sequence/context length, adaptations, and precision config. Includes staleness detection: if the HuggingFace hub config changes (e.g., model updated upstream), the cached entry is invalidated.

| Flag | Effect |
|------|--------|
| `--model-cache-dir <path>` | Custom model cache directory |
| `--clear-model-cache` | Wipe cache before running |

## Architecture

```
GenAILab/
├── __main__.py                 # CLI entry point
├── CONFIG.md                   # Full config schema reference
├── bench/                      # The harness: config -> quantize -> evaluate -> report
│   ├── yaml_config_parser.py   # Config parsing and the plugin registry
│   ├── precision.py            # Precision configuration
│   ├── recipe_chain.py         # Multi-step recipe execution
│   ├── datasets.py             # Dataset implementations
│   ├── metrics.py              # Evaluation metrics
│   ├── eval_context.py         # FP/quant result caching
│   ├── profiler.py             # GPU profiling and result output
│   ├── summary.py              # Result tables
│   ├── model_cache.py          # Exported-ONNX cache
│   ├── fp_cache.py             # FP-output cache
│   ├── recipe_cache.py         # Recipe-chain checkpoint cache
│   ├── export.py               # Export utilities
│   ├── runner.py               # Runs each experiment in a config
│   ├── torch/                  # run_genai.py (torch experiment) + quant_recipes.py
│   └── onnx/                   # run_genai.py (ONNX experiment) + quant_recipes.py
├── qai_hub_lm/                 # The model layer
│   ├── schema/                 # Pydantic config schema (stdlib-only import island,
│   │                           #   rsync'd verbatim into AI Hub Models)
│   │   ├── components.py       # Component names, kinds, canonical order
│   │   ├── precision.py        # precision: section
│   │   ├── recipe.py           # recipe: section
│   │   └── dataset.py          # dataset: specs
│   ├── models/                 # Framework-agnostic model classes
│   │   ├── base.py             # LLM / VLM base classes, SimCollection
│   │   ├── components.py       # ComponentSpec registry (visual, audio)
│   │   ├── generator.py        # Generator / VLM_Generator inference
│   │   ├── gemma3.py, gemma4.py, qwen2_vl.py, qwen3_vl.py, qwen3_asr.py, ...
│   │   └── utils/              # Attention masks, KV-cache descriptors, RoPE, export
│   ├── backends/               # Per-framework quantsim construction
│   │   ├── torch/              # llm.py, vlm.py, generator_utils.py
│   │   └── onnx/               # llm.py, vlm.py, export_utils.py, generator_utils.py
│   ├── transforms/             # Model adaptations (SHA, exportable attention, MoE)
│   └── scoring/                # Autograded-prompt and Grace scoring
├── configs/                    # Regression test configs
├── tests/unit/                 # Unit tests
└── dashboard/                  # Result dashboards
```

## How It Works

1. **Config Parsing**: YAML config is parsed into model class, precision, recipe steps, and metrics
2. **Model Instantiation**: Load the HuggingFace model, apply adaptations, wrap in QuantizationSimModel
3. **Precision Configuration**: Apply quantization bitwidths and granularity settings
4. **Recipe Application**: Execute recipe steps in order (e.g., SeqMSE then Calibration), with caching
5. **Evaluation**: Run specified metrics on the quantized model
6. **Export** (optional): Export to ONNX format
7. **ONNX Evaluation** (optional): Re-evaluate the exported ONNX model
8. **Results**: Write profiling data (JSON + CSV) for dashboard ingestion
