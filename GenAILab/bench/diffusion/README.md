# Text-to-image diffusion (Stable Diffusion)

FP inference of Stable Diffusion 1.x with torch or ONNX Runtime, scored with
CLIPScore. This runs separately from the LLM harness (`python -m GenAILab`) for now:

```bash
# torch
python -m GenAILab.bench.torch.run_diffusion \
    --config GenAILab/bench/diffusion/configs/diffusion_fp.yaml

# ONNX Runtime
python -m GenAILab.bench.onnx.run_diffusion \
    --config GenAILab/bench/diffusion/configs/diffusion_fp.yaml
```

Needs `diffusers` installed. Results are appended to `profiling_data.json`/`.csv`
in `--results-dir` (default `GenAILab/artifacts/results/`), in the same format as
LLM results, and a summary table is printed at the end. `ImageDump` writes its
images under `<results-dir>/images/<model>_<run_group>/`.

| Flag | Effect |
|------|--------|
| `--config <path>` | YAML config; may hold several documents separated by `---` |
| `--results-dir <path>` | Where results and images go |
| `--fp-cache-dir <path>` / `--clear-fp-cache` | FP results cache location / wipe it first |
| `--onnx-cache-dir <path>` / `--clear-onnx-cache` | ONNX only: exported graphs location (default `GenAILab/artifacts/cache/diffusion_onnx/`) / wipe it first |
| `--fail-fast` / `--pdb` | Stop at the first failed document / debug it |

Each result records its `framework` under `model_modifiers`, and the FP results
cache is keyed by framework, so torch and ONNX numbers are never mixed.

## ONNX

The ONNX runner exports each component (`text_encoder`, `unet`, `vae_decoder`)
with the torch dynamo exporter at **opset 18**, with fixed shapes (batch 1, the
configured resolution), and runs it with ONNX Runtime behind the same
`DiffusionGenerator`. On CUDA, inputs and outputs stay in GPU memory between
denoising steps. Exports are cached by model, checkpoint revision, dtype,
resolution, opset, and the torch and diffusers versions, so only the first run
of a configuration pays for them (about half a minute for SD1.5's UNet).

Opset 18 is deliberate. With ONNX Runtime 1.23, newer opsets move Conv, Resize
and normalization onto the CPU, and opset 23's fused `Attention` gives wrong
results on CUDA at SD1.5's head size. Opset 18 cannot express bfloat16 Conv, so
the ONNX runner supports `float32` and `float16` only and rejects `bfloat16`
before loading the model.

## Layout

- [`qai_hub_lm/diffusion/`](../../qai_hub_lm/diffusion/) — model layer, no AIMET dependency:
  tensor-in/tensor-out `text_encoder` / `unet` / `vae_decoder` wrappers, the
  `StableDiffusion` model class, and `DiffusionGenerator`, which runs the
  denoising loop (scheduler, classifier-free guidance, seeding).
- [`config.py`](config.py) — config parsing.
- [`metrics.py`](metrics.py) — `CLIPScore` and `ImageDump`.
- [`experiment.py`](experiment.py) — runs one config document, with torch or ONNX.
- [`onnx_cache.py`](onnx_cache.py) — the cache of exported ONNX graphs.
- [`runner.py`](runner.py) — the loop over a config's documents, shared by the entry points.
- [`../torch/run_diffusion.py`](../torch/run_diffusion.py), [`../onnx/run_diffusion.py`](../onnx/run_diffusion.py) — the command-line entry points.

The ONNX export and the ONNX Runtime session adapter live in
[`qai_hub_lm/diffusion/onnx.py`](../../qai_hub_lm/diffusion/onnx.py).

`DiffusionGenerator` reproduces `StableDiffusionPipeline`, but calls the UNet at
batch 1, once for the conditional and once for the unconditional prediction.

## Config

```yaml
model:
  model_id: stable-diffusion-v1-5/stable-diffusion-v1-5
  dtype: float16
  num_inference_steps: 10
metrics:
  - name: CLIPScore
    num_prompts: 64
  - name: ImageDump
```

Top-level keys: `model` and `metrics` (required); `profiler` and `run_group`.
`precision`, `recipe`, `dataset`, `export`, `eval_in_onnx` and `analysis` are
rejected: only FP inference is supported so far.

| `model` key           | Required | Type  | Default | Notes |
| --------------------- | -------- | ----- | ------- | ----- |
| `model_id`            | yes      | str   | —       | Diffusers pipeline repo or local directory. Its `model_index.json` `_class_name` selects the model class and is reported as the model type. |
| `dtype`               | no       | str   | `float32` | `float32`, `float16` or `bfloat16` (`bfloat16` is torch only). |
| `height`, `width`     | no       | int   | checkpoint's native size (512 for SD1.5) | Multiples of the VAE scale factor (8 for SD). |
| `num_inference_steps` | no       | int   | `10`    | Denoising steps. |
| `guidance_scale`      | no       | float | `7.5`   | Classifier-free guidance; `<= 1` disables the unconditional pass. |
| `negative_prompt`     | no       | str   | `""`    | Unconditional prompt for guidance. |
| `scheduler`           | no       | str   | `DPMSolverMultistepScheduler` | Any diffusers scheduler class name, e.g. `EulerDiscreteScheduler`. SD1.5's own default, PNDM, gives partly garbled images at 10 steps. |
| `seed`                | no       | int   | `0`     | Prompt `i` of a metric is generated with seed `seed + i`. |

LLM-only keys (`sequence_length`, `context_length`, `adaptations`, `encodings`)
are rejected, and only the image-generation metrics below are accepted.

## Metrics

Both generate images for a fixed, category-stratified sample of
[PartiPrompts](https://huggingface.co/datasets/nateraw/parti-prompts) (pinned
revision). Metrics with the same `num_prompts` and `sample_seed` share the
generated images.

| Name        | Kwargs (in addition to `name`) |
| ----------- | ------------------------------ |
| `CLIPScore` | `num_prompts` (int, default `64`); `sample_seed` (int, default `0`); `clip_model` (str, default `openai/clip-vit-large-patch14`). Mean of `100 * max(cos(image, text), 0)`; higher is better. `details` has per-category and per-prompt scores. |
| `ImageDump` | `num_prompts` (int, default `8`); `sample_seed` (int, default `0`); `output_dir` (str, default `<results-dir>/images/<model>_<run_group>/`). Writes one PNG per prompt and `grid.png`; the result is the image count. |
