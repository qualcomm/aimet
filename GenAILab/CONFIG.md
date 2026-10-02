# GenAILab YAML Configuration Reference

This document describes the YAML config format consumed by the GenAILab harness. It has two parts:

1. [**Guide**](#guide) — a walkthrough that builds a config from minimal to full-featured.
2. [**Reference**](#reference) — exhaustive schema for every field.
3. [**Currently registered values**](#currently-registered-values) — names you can use in `recipe`, `dataset`, `metrics`, `model.adaptations`. Snapshot as of **2026-05-20**; new entries are added by `@YAMLConfigParser.register_*` decorators, so keep this in mind when reading.

The schema is enforced by [yaml_config_parser.py](bench/yaml_config_parser.py) and [precision.py](bench/precision.py).

---

## Guide

### Minimal config

The smallest valid document needs only `model` and `metrics`:

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
metrics:
  - name: TinyMMLU
```

Without a `recipe` section, the model is loaded but not quantized (the harness inserts a default `RemoveQuantization` step). Without `precision`, the harness applies its built-in defaults (W4A16, INT8 lm_head and KV cache, INT16 embedding).

### Adding precision

Override the precision defaults under `precision:`:

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
precision:
  activations: int16
  blocks:
    qtype: int4
    granularity: BQ
    block_size: 32
metrics:
  - name: TinyMMLU
```

`blocks` covers transformer-block weights; `lm_head` and `visual.weight` are configured separately.

### Adding a recipe

A `recipe` section drives the quantization pipeline. The shortest form is a single recipe dict:

```yaml
recipe:
  name: Calibration
  dataset:
    name: Wikitext
    split: train
```

For multi-step pipelines, use a list (treated as the `backbone` chain):

```yaml
recipe:
  - name: SeqMSE
    dataset:
      name: Wikitext
      split: train
  - name: Calibration
    dataset:
      name: Wikitext
      split: train
```

For VLMs, scope each chain to a component:

```yaml
recipe:
  backbone:
    - name: SpinQuant
    - name: Calibration
      dataset: { name: Wikitext, split: train }
  visual:
    - name: SpinQuant
    - name: Calibration
      dataset: { name: Wikitext, split: train }
```

The parser auto-appends a `Calibration` step (with Wikitext/train) to any chain that doesn't end in a *terminal* recipe (`Calibration`, `RemoveQuantization`, `Skip`) and emits a warning. To suppress, add a terminal step explicitly.

### Multi-document files

A single YAML file can hold multiple documents separated by `---`. Each document is validated and run independently:

```yaml
model: { model_id: ..., sequence_length: 2048, context_length: 4096 }
metrics: [{ name: PPL }]
---
model: { model_id: ..., sequence_length: 2048, context_length: 4096 }
precision:
  blocks: { qtype: int4, granularity: PCQ }
recipe:
  name: Calibration
  dataset: { name: Wikitext, split: train }
metrics: [{ name: PPL }, { name: TinyMMLU }]
```

The first document above produces a floating-point baseline (no recipe → `RemoveQuantization`); the second is a quantized run.

---

## Examples

### Minimal LLM

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  backbone:
    - name: Calibration
      dataset:
        name: Wikitext
        split: train
metrics:
  - name: PPL
  - name: TinyMMLU
```

### VLM with multi-step recipe

```yaml
model:
  model_id: Qwen/Qwen2.5-VL-7B-Instruct
  sequence_length: 2048
  context_length: 4096
  image_size: [504, 336]
  adaptations:
    - FastExportable
precision:
  blocks:
    qtype: int4
  activations: int16
  kv_cache: int8
  lm_head:
    qtype: int8
  visual:
    weight:
      qtype: int8
    activations: int16
recipe:
  backbone:
    - name: SeqMSE
      dataset:
        name: Wikitext
        split: train
    - name: Calibration
      num_iterations: 128
      dataset:
        name: Interleaved
        source_datasets:
          - name: Wikitext
            split: train
          - name: AOKVQA
            split: train
  visual:
    - name: Calibration
      num_iterations: 128
      dataset:
        name: AOKVQA
        split: train
metrics:
  - name: PPL
  - name: MMMU
  - name: MultimodalPrompts
```

### Embodied/robotics VLM backbone (e.g. Cosmos-Reason2, `qwen3_vl` model_type)

`ERQA` (multi-image, 4-way MCQ) and `Where2Place` (single-image pointing,
scored by point-in-mask containment) evaluate the embodied-reasoning/spatial-
grounding categories where these checkpoints diverge most from a generic VLM
of the same architecture. Both are eval-only (no `include_answer`/train-split
calibration variant like `AOKVQA`) -- calibration for this model family still
uses the ordinary `visual`/`backbone` recipe blocks above.

```yaml
model:
  model_id: Qwen/Qwen3-VL-2B-Instruct  # or nvidia/Cosmos-Reason2-2B once available
  sequence_length: 2048
  context_length: 4096
  image_size: [504, 336]
precision:
  blocks:
    qtype: int4
  activations: int16
  kv_cache: int8
  lm_head:
    qtype: int8
  visual:
    weight:
      qtype: int8
    activations: int16
recipe:
  backbone:
    - name: Calibration
      num_iterations: 128
      dataset:
        name: Interleaved
        source_datasets:
          - name: Wikitext
            split: train
          - name: AOKVQA
            split: train
  visual:
    - name: Calibration
      num_iterations: 128
      dataset:
        name: AOKVQA
        split: train
metrics:
  - name: ERQA
  - name: Where2Place
```

### Audio (ASR) model

Audio is an input-side sibling of the vision encoder: the audio encoder's
embeddings are scattered into the token sequence at the audio placeholder token,
and the model still emits text logits — so the ordinary text metrics apply, plus
`WER`/`CER`.

Use the `-hf` checkpoints. The non-`-hf` Qwen3-ASR repos publish an Omni-style
`thinker_config` layout that `AutoConfig` silently reads as all-defaults.

`audio_frames` is the padded mel-frame count that fixes the encoder's exported
shape. Mel frames run at 100/s, but what a valid value is — and how many audio
tokens it becomes — depends on the encoder:

| Model | `audio_frames` constraint | Audio tokens | Default |
| ----- | ------------------------- | ------------ | ------- |
| Qwen3-ASR | multiple of `n_window * 2` (= 100); the encoder raises otherwise | 13 per 100 frames (13/s) | 800 (8 s, 104 tokens) |
| Gemma4 | none | frames / 4, from two stride-2 subsampling convs (25/s) | 800 (8 s, 200 tokens) |

Gemma4 additionally publishes checkpoints with `audio_config: null` and no audio
tower; those get no audio component and no `audio:` precision or recipe block,
whatever this file says.

```yaml
model:
  model_id: Qwen/Qwen3-ASR-1.7B-hf
  sequence_length: 128
  context_length: 1024
  audio_frames: 800  # 8 s of audio -> 104 audio tokens (see table above)
precision:
  audio:
    weight:
      qtype: int8
    activations: int16
recipe:
  backbone:
    - name: Calibration
      dataset:
        name: LibriSpeech
        split: validation.clean
        num_samples: 32
  audio:
    - name: Calibration
      dataset:
        name: LibriSpeech
        split: validation.clean
        num_samples: 32
metrics:
  - name: WER
  - name: CER
```

### FP baseline (no quantization)

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  backbone:
    - name: RemoveQuantization
metrics:
  - name: PPL
```

---

## Reference

### Top-level keys

| Key            | Required | Type            | Default | Notes                                            |
| -------------- | -------- | --------------- | ------- | ------------------------------------------------ |
| `model`        | yes      | dict            | —       | Model identification + adaptations.              |
| `metrics`      | yes      | list of dict    | —       | Evaluation metrics.                              |
| `precision`    | no       | dict            | (built-in) | Quantization precision overrides.             |
| `recipe`       | no       | dict or list    | `RemoveQuantization` (or `Skip` if `model.encodings` is set) | Quantization technique pipeline. |
| `dataset`      | no       | dict            | —       | **Deprecated.** Top-level dataset, migrated into the first backbone recipe step. |
| `export`       | no       | bool or str     | `false` | If `true`, auto-generates an artifact path; if a string, uses that path. |
| `eval_in_onnx` | no       | bool            | `false` | Forces `export=true` when set.                   |
| `run_group`    | no       | str or null     | `null`  | Free-form grouping label for results merging.    |
| `profiler`     | no       | dict            | `{}`    | Passed through to the profiler module.           |

Any other top-level key produces `ValueError: Unrecognized sections in config`.

### `model`

| Subkey            | Required | Type           | Default | Notes |
| ----------------- | -------- | -------------- | ------- | ----- |
| `model_id`        | yes      | str            | —       | HuggingFace ID (e.g. `meta-llama/Llama-3.2-1B-Instruct`) or local checkpoint path. Used to detect `model_type` via `transformers.AutoConfig`. |
| `sequence_length` | yes      | int or list[int] | —     | Prompt sequence length. List form supports multi-shape variants (e.g. `[4096, 2048, 1024, 512, 1]`). |
| `context_length`  | yes      | int            | —       | Maximum decoding context. |
| `adaptations`     | no       | list           | `[]`    | See [adaptations](#adaptations). |
| `encodings`       | no       | str            | —       | Path to pre-computed encodings. When present, the default recipe becomes `Skip` instead of `RemoveQuantization`. |
| `image_size`      | no       | list[int]      | —       | VLM-only. `[height, width]` for vision encoder inputs. |
| `audio_frames`    | no       | int            | —       | Audio-model-only. Padded mel-frame count fixing the audio encoder's traced/exported shape — the audio analogue of `image_size`. Mel frames run at 100/s (10 ms hop at 16 kHz), so 800 frames = 8 s. The valid values and the resulting audio-token count are **per model** — see [Audio models](#audio-asr-model). Omitted, each model class uses its own default (800 for both Qwen3-ASR and Gemma4). |
| `dtype`           | no       | str            | —       | Model dtype override (e.g. `float16`, `bfloat16`); resolved via `getattr(torch, dtype)`. |

`model` must be a dict (single model per document). A list raises an error.

#### `model.adaptations`

Each list entry is either:
- a string (`"SHA"`) — adaptation name, no kwargs.
- a single-key dict (`{"AttentionMaskScale": {"layer_multipliers": {0: 0.8}}}`) — name + class-attribute kwargs.

Constraints enforced by the parser:
- An `exclusive` adaptation (e.g. `AIHM`, `SHA`) cannot be combined with any other.
- Adaptations marked `required_for_export` are auto-enforced when the document will export an ONNX artifact, unless an exclusive adaptation owns the pipeline.

### `precision`

`precision` is parsed by `PrecisionConfig.from_dict`. Omitting the section yields:

```yaml
precision:
  activations: int16
  kv_cache: int8
  embedding: int16
  lm_head: { qtype: int8, granularity: PCQ }
  blocks: { default: { qtype: int4, granularity: PCQ } }
```

| Subkey         | Type                       | Default            | Notes |
| -------------- | -------------------------- | ------------------ | ----- |
| `activations`  | int / str (qtype alias)    | `int16`            | Accepts `int4`, `int8`, `int16`, `float16`, `float32`. Setting an FP value disables activation quantizers and forces KV cache + embedding to the same FP type at runtime. |
| `kv_cache`     | int / str                  | `int8`             | Auto-overridden to FP if `activations` is FP. |
| `embedding`    | int / str                  | `int16`            | Plain (non-VLM) LLMs only support `int16`; any other value raises `NotImplementedError` since the embedding isn't wired into the sim. VLM subclasses honor any value. |
| `lm_head`      | int / str / dict           | `{qtype: int8, granularity: PCQ}` | See [WeightPrecision](#weightprecision). FP qtypes are accepted (drop the lm_head weight quantizer). |
| `blocks`       | int / str / flat dict / `{default: dict}` | `{default: {qtype: int4, granularity: PCQ}}` | Per-component block precision. Only `default` is currently accepted as a key. FP qtypes accepted. |
| `visual`       | dict                       | —                  | VLM-only. Sub-keys `weight` (a `WeightPrecision` dict, INT only) and `activations` (qtype, default `int16`). |
| `audio`        | dict                       | —                  | Audio-models-only. Same shape as `visual`: `weight` (`WeightPrecision`, INT only) and `activations` (qtype, default `int16`). Only applied to models that declare an audio component. |

Accepted qtype aliases (strings or shorthand ints): `int2`, `int4`, `int8`, `int16`, `float16`, `float32`. An int value `N` is interpreted as `int{N}`.

#### `WeightPrecision`

Used for `lm_head`, `blocks.default`, and `visual.weight`. Forms accepted:

- **shorthand int**: `8` → `WeightPrecision(qtype=int8)`.
- **shorthand string**: `"int4"` → same as the alias.
- **flat dict**: `{qtype: int4, granularity: BQ, block_size: 32}` (full form).

| Subkey       | Type     | Default       | Notes |
| ------------ | -------- | ------------- | ----- |
| `qtype`      | int / str | `int4` (blocks) / `int8` (lm_head, visual) | Accepts FP for `blocks` and `lm_head`; rejected for `visual.weight`. |
| `granularity` | str     | `PCQ`         | One of `PCQ` (per-channel), `BQ` (blockwise), `LPBQ` (low-precision blockwise). |
| `block_size`  | int     | —             | Required for `BQ` and `LPBQ` granularity (integer qtypes only). Ignored for FP weights. |

When `blocks.qtype` is a floating-point type, the parser rejects any recipe step whose name is not in `{Calibration, SpinQuant, RemoveQuantization, Skip}` — the others (SeqMSE, AdaScale, AdaRound, …) only modify weights, which would be a no-op.

### `recipe`

`recipe` accepts four forms; the parser normalizes them all to `{<component>: [<step>, ...]}`:

| Form                                                | Normalized to |
| --------------------------------------------------- | ------------- |
| Omitted                                             | `{backbone: [{class: RemoveQuantization}]}` plus the same step for each modality component the model declares (`visual` for VLMs, `audio` for audio models) — or `Skip` if `model.encodings` is set. |
| Single dict: `{name: Calibration, ...}`             | `{backbone: [{name: Calibration, ...}]}` |
| List of step dicts: `[{...}, {...}]`                | `{backbone: [{...}, {...}]}` |
| Component dict: `{backbone: [...], visual: [...], audio: [...]}` | (passthrough; component values can be a single dict or a list) |

Component keys are `backbone` plus one per modality encoder the model has: `visual` for VLMs, `audio` for audio models (Qwen3-ASR). A model only ever gets the components it declares — an audio model never acquires a `visual` chain, and vice versa.

Each *step* is a dict with:

| Subkey     | Required | Type | Notes |
| ---------- | -------- | ---- | ----- |
| `name`     | yes      | str  | Recipe class name. See [registered recipes](#recipes). |
| `dataset`  | no       | dict | Inline dataset config (see [registered datasets](#datasets)). Required for recipes that need calibration data. |
| (other)    | no       | (varies) | Additional kwargs are forwarded to the recipe class. See per-recipe details. |

Auto-insertion: if a chain has no terminal recipe (`Calibration`, `RemoveQuantization`, `Skip`), the parser appends a `Calibration` step on `Wikitext/train` and emits a warning. To suppress, end the chain explicitly.

Validation rules:
- For multi-modal models: if `SpinQuant` is in `backbone`, it must also be in every other component chain (`visual` / `audio`), and it must be that chain's first step. Pre-sim techniques act on the whole float model, so the pre-sim prefix must be byte-identical across components.
- For FP `blocks.qtype`: only `Calibration`, `SpinQuant`, `RemoveQuantization`, `Skip` are allowed.
- For `SpinQuant` (onnx): at least one of `enable_r1` / `enable_r2` must be `true`. Setting both to `false` is rejected at parse time.

### `metrics`

A list of dicts (a single dict is accepted and wrapped). Each dict:

| Subkey | Required | Type | Notes |
| ------ | -------- | ---- | ----- |
| `name` | yes      | str  | Metric class name. See [registered metrics](#metrics). |
| (other) | no      | (varies) | Additional kwargs are forwarded to the metric class. |

### `dataset` (top-level, deprecated)

A backward-compatibility shim. If `dataset` is present and the first `backbone` recipe step does not already have `dataset`, the value is migrated into that step. If the first backbone step already has its own `dataset`, the top-level value is silently discarded. Prefer setting `dataset` directly on each recipe step.

### `export` and `eval_in_onnx`

- `export: false` (default) — no ONNX artifact written.
- `export: true` — artifacts written under `GenAILab/artifacts/exports/<auto-generated-path>/`. The full document is also serialized to `<path>/config.yaml`.
- `export: <string>` — artifacts written under that string path.
- `eval_in_onnx: true` — runs evaluation against the exported ONNX. Implies `export=true`; if `export=false`, the parser overrides it and emits a warning.

### `run_group`

A free-form string (or `null`) used by the results merger to group runs. No parser validation.

### `profiler`

A free-form dict, passed through to the profiler module without schema validation. Refer to the profiler implementation for accepted keys.

### Multi-document semantics

Files are loaded with `yaml.safe_load_all`; documents are separated by `---`. Each document is validated and parsed independently — there is no shared state across documents at parse time. Some downstream caches (FP eval cache, ONNX export cache) may key off `model_id` and reuse work across documents, but that's a runtime concern, not part of the schema.

---

## Currently registered values

Snapshot taken **2026-05-20**. To list current registrations yourself, grep for the decorators (e.g., `rg "@YAMLConfigParser.register_recipe"`).

### Recipes

Both backends (torch + onnx) register the same six names. Default values differ between backends where noted.

| Name                  | Backend(s)   | Step kwargs (in addition to `name` / `dataset`) |
| --------------------- | ------------ | ----------------------------------------------- |
| `Calibration`         | torch, onnx  | `num_iterations` (int, default `20`).           |
| `SeqMSE`              | torch, onnx  | inherits `num_iterations` from Calibration.     |
| `AdaScale`            | torch, onnx  | `num_batches` (int — torch default `20`, onnx default `32`); `num_iterations` (int — torch default `1500`, onnx default `64`). |
| `SpinQuant`           | torch, onnx  | `component` (`"backbone"` or `"visual"`, default `"backbone"`). **onnx only:** `enable_r1` (bool, default `true`) — apply the R1 (residual-stream) Hadamard rotation; `enable_r2` (bool, default `false`) — apply the R2 (per-head) Hadamard rotation. At least one must be `true`. R2 is unsupported on architectures with fused QKV (e.g. Phi3). |
| `Calibration` step end-of-chain auto-insertion uses Wikitext/train. |
| `RemoveQuantization`  | torch, onnx  | none.                                            |
| `Skip`                | torch, onnx  | none.                                            |

### Datasets

Defined in [bench/datasets.py](bench/datasets.py).

| Name           | Type        | Step kwargs (in addition to `name`) |
| -------------- | ----------- | ----------------------------------- |
| `Wikitext`     | text        | `split` (str, e.g. `train` / `test`). |
| `TinyMMLU`     | text        | `split` (str, default `test`).      |
| `MMLU`         | text        | `split` (default `test`); `num_fewshot` (int, default `5`); `fewshot_split` (str, default `dev`). |
| `MMMLU`        | text        | `split` (default `default`); `num_fewshot` (int, default `5`). |
| `C4`           | text        | `split` (default `en`); `num_samples` (int, default `2048`). |
| `MMMU`         | multimodal  | `split` (default `validation`); `image_size` (tuple, optional). |
| `AOKVQA`       | multimodal  | `split` (default `train`); `image_size` (tuple, optional). |
| `ERQA`         | multimodal  | `split` (default `test`; the only split, 400 rows). |
| `Where2Place`  | multimodal  | `split` (default `train`; the only split, 100 rows); `image_size` (tuple, optional). |
| `LibriSpeech`  | audio       | `split` (default `test.clean`, or `validation.clean` for calibration); `num_samples` (int, optional); `language` (str, default `English`); `include_reference` (bool, default `false`). |
| `Interleaved`  | multimodal  | `source_datasets` (list of dataset configs, required). |

### Metrics

Defined in [bench/metrics.py](bench/metrics.py).

| Name                          | Step kwargs (in addition to `name`) |
| ----------------------------- | ----------------------------------- |
| `PPL`                         | `batch_size` (int, default `1`); `num_iterations` (int or `null`). |
| `TinyMMLU`                    | none.                                |
| `MMLU`                        | `num_fewshot` (int, default `5`).    |
| `MMLU1000`                    | `num_fewshot` (int, default `5`).    |
| `MMMLU`                       | `split` (str); `num_fewshot` (int, default `5`). |
| `MMLUKLDivergence`            | `num_fewshot` (int, default `5`).    |
| `MMLUReverseKLDivergence`     | `num_fewshot` (int, default `5`).    |
| `MMLUFlips`                   | `num_fewshot` (int, default `5`).    |
| `MMLUJSDivergence`            | `num_fewshot` (int, default `5`).    |
| `MMMU`                        |                                      |
| `MMMUKLDivergence`            |                                      |
| `MMMUReverseKLDivergence`     |                                      |
| `MMMUFlips`                   |                                      |
| `MMMUJSDivergence`            |                                      |
| `ERQA`                        |                                      |
| `ERQAKLDivergence`            |                                      |
| `ERQAReverseKLDivergence`     |                                      |
| `ERQAFlips`                   |                                      |
| `ERQAJSDivergence`            |                                      |
| `Where2Place`                 | Free-space pointing accuracy (%) -- any predicted point in the ground-truth mask; higher is better. |
| `Interactive`                 | none.                                |
| `Prompts`                     | none.                                |
| `MultimodalPrompts`           | none.                                |
| `WER`                         | none. Word error rate (%) on LibriSpeech; lower is better. |
| `CER`                         | none. Character error rate (%) on LibriSpeech, spaces included; lower is better. |
| `TrickyPrompts`               | none.                                |
| `AutogradedPrompts`           | `harness_version` (str, default `v1`). |
| `AutogradedMultimodalPrompts` | `harness_version` (str, default `v1`). |
| `Grace`                       | `num_samples` (int, default `0` = all 100); `max_new_tokens` (int, default `2048`); `seed` (int, default `42`); `deterministic` (bool, default `true`); `grader_model_id` (str, default `Qwen/Qwen3.6-35B-A3B`); `grader_dtype` (`bfloat16`\|`float16`\|`float32`, default `bfloat16`); `grader_device_map` (str, default `auto`); `allow_cpu` (bool, default `false`); `summary` (bool, default `true`); `output_dir` (str, default unset). |

#### Grace

Grace ("Grading Response Accuracy Evaluation") generates one free-form response
per prompt over a built-in set of 10 categories x 10 prompts, then has a grader
LLM rate each on a 0-10 rubric with a one-line rationale, and finishes with a
pass that distils those rationales into the recurring failure modes. The
reported score is total points as a percentage of the maximum.

The result's `details` carries the per-category breakdown, the defect summary
(the dashboard's `Grace` and `Grace Defects` columns), and an `items` array
holding every prompt, its response, and the grade with the grader's reason. So a
score that moved can be explained from `profiling_data.json` alone, without
re-running generation. It is reported under the metric that produced it, at
`accuracy_results->'Grace'->'details'`, so everything a metric reported lives in
one place; queries that only want the score keep projecting `->'result'`.
`output_dir` writes the same two halves as local
`responses.json` and `grader_summary.json` files, which diff more readably.

Scores only compare across runs while the prompt set, the rubric and the
generation path stay identical; a change to any of them needs a `GRACE_VERSION`
bump. The reported name stays `Grace` across versions -- `GRACE_VERSION` is
surfaced as the metric's `scoring_version` instead, so a bump shows up as data
rather than silently renaming the results key out from under existing queries.

Two caveats worth knowing before enabling it:

* The default grader is a 35B MoE. The model under test is evicted to CPU while
  the grader is loaded, but the grader still needs the GPU (or `grader_device_map`
  across several). It refuses to run on CPU unless `allow_cpu` is set, because a
  silent CPU fallback turns seconds into minutes per file while still exiting 0.
* `deterministic` sets `torch.use_deterministic_algorithms(True)` for the
  generation loop, so a greedy decode does not diverge across hosts. If an op in
  the model under test has no deterministic kernel, generation raises; set it to
  `false` to fall back (aggregate scores stay comparable, individual responses
  may not).

The grader always runs deterministically, so the same responses get the same
grades on every run. It uses deterministic torch/cuBLAS kernels and a fixed
seed, and turns off cuDNN attention, which is not repeatable on H100. The job
log shows `Grader deterministic mode: ... cudnn_sdp=False`.

### Models

Plain LLMs use the registered default class (`LLM_Torch` / `LLM_ONNX`). Special models register against their HuggingFace `model_type`:

| `model_type`  | Torch class       | ONNX class        |
| ------------- | ----------------- | ----------------- |
| `qwen2_5_vl`  | `Qwen_25_VL_Torch` | `Qwen_25_VL_ONNX` |
| `qwen3_vl`    | `Qwen_3_VL_Torch` | `Qwen_3_VL_ONNX` |

### Adaptations

Listed under `model.adaptations` in the YAML.

| Name                 | Bound model_type(s)    | `exclusive` | `required_for_export` | Adaptation kwargs |
| -------------------- | ---------------------- | ----------- | --------------------- | ----------------- |
| `SHA`                | `llama`, `qwen3`       | yes         | no                    | none.             |
| `SHA_Conv`           | `llama`, `qwen3`       | yes         | no                    | none.             |
| `FastExportable`     | `qwen2_5_vl`, `qwen3_vl` | no        | yes                   | none.             |
| `AttentionMaskScale` | `*` (all)              | no          | no                    | `layer_multipliers` (dict[int, float], e.g. `{0: 10.0, 5: 25.0}`). |
| `AIHM`               | `*` (all)              | yes         | no                    | none (auto-routes to a supported `qai_hub_models` model). |
