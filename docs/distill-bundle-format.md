# Distill bundle format, for lobes' Nano backend

This is the contract between `jev distill bundle` (this repo) and a backend that loads the
bundle on an Orin Nano (lobes-cli's `orin-decision-models`). It describes what the code does
today; the code is the authority and `tests/test_bundle_format_doc.py` fails when the bundle
stage writes a file this page does not list. Nothing here edits lobes-cli.

> **State.** The format is built and fixture-tested. No distill run has produced a student
> bundle yet, so no bundle exists to download. Do not describe one as shipped.

Source of truth: `jev_factory/distill/bundle.py` (the stage),
`jev_factory/backbones/pointer/quantized.py` (the quantized artifact, its reader and the
server) and `jev_factory/backbones/pointer/layout.py` (the bf16 layout).

## Two folders, both private

`jev distill bundle` writes one output directory with two upload folders, `model` and
`dataset`. Both are dry-run by default and uploaded private-first. The sealed held-out and
test text are never in either.

### The `model` folder

Written by the stage, in both deployed formats:

| File | Content |
|---|---|
| `bundle.json` | format `jev-distill-bundle/1`, `kind: model`, the frozen training set's sha256 |
| `calibration.json` | the fitted per-kind temperatures (jev-factory's, never upstream's) |
| `gate.json` | the per-kind answer-or-abstain gate, or a record that none was fitted |
| `scorer-train.json` | index of the frozen training set (target source, licence, teacher build; no text) |
| `report.json` | the single final comparison |
| `assemble_manifest.json` | the assemble stage's manifest, byte for byte |
| `distill_provenance.json` | deployed format, pre-registration and calibration sha256, teacher builds, audits, pins, environments |
| `NOTICE` | the attribution strings from the dataset manifest |
| `scan.json` | the pre-upload scan record, written last |

Plus the student itself, one of two layouts, chosen by `distill/quant/format_decision.json`.

**Quantized, `jev-pointer-torchao-wo/2`** (the Nano format, read by
`quantized.read_artifact`):

| File | Content |
|---|---|
| `jev_quant.json` | the manifest: scheme, torchao and torch versions, sha256 of every file, weight byte sizes, packed-embedding record, tensor policy, inventory, source checkpoint hashes |
| `torso_quantized.pt` | the torso state (`torch.save`, loaded with `weights_only=True`) |
| `head.safetensors` | the pointer head, fp32 |
| `strands_decider_config.json` | `use_lora: false`, the pinned `base_revision`, `head_type: pointer`, the fitted `temperature_by_kind` |
| `tokenizer.json` | tokenizer, copied byte for byte from the source checkpoint |
| `tokenizer_config.json` | tokenizer config, copied byte for byte |

The tokenizer files carried when the source checkpoint has them: `special_tokens_map.json`,
`added_tokens.json`, `chat_template.jinja`, `vocab.json`, `merges.txt`, `tokenizer.model`.

**bf16, `hobson-hf-export/1`** (the layout `strands-decider serve` loads; read by
`layout.read_layout`):

| File | Content |
|---|---|
| `lora/adapter_config.json` | LoRA adapter config |
| `lora/adapter_model.safetensors` | the LoRA adapter, unmerged |
| `head.safetensors` | the pointer head |
| `strands_decider_config.json` | the model config with the fitted temperatures |
| `provenance.json` | format, base model, pinned base revision, adapter and head sha256 |
| `MANIFEST.sha256` | sha256 of every other file in the folder |

with the same tokenizer files, and, when present, `train_config.json`, `history.json` and
`chat_template.jinja`. The bf16 layout needs the base model fetched at the pinned revision; the Nano backend
loads the quantized layout.

In both layouts `strands_decider_config.json` carries the fitted `temperature_by_kind`
and a scalar `temperature` of `1.0`. A kind without a fitted value is left uncalibrated.

### The bf16 bundle kind

When the pre-registration's `deployed_format` is `bf16`, the quantize stage (and its heal
round) is recorded skipped with the reason `deployed_format bf16`, and the bundle ships the
train stage's selected candidate instead of a quant build. Nothing else falls back to it: a
`quant` pre-registration whose quantize found no qualifying build still refuses the bundle.

- The student is `distill/select/summary.json`'s `winner`, copied from
  `distill/train/<winner>/bundle` in the bf16 `hobson-hf-export/1` layout above, so
  `strands-decider serve` loads it and `layout.read_layout` verifies it.
- `calibration.json` is `distill/select/calibration.json`, byte for byte; a fitted gate
  must name its sha256.
- The `model` and `dataset` file sets are the ones listed above.

Additive fields, written only in a bf16-kind bundle (a quant bundle is unchanged):

| File | Field | Content |
|---|---|---|
| `distill_provenance.json` | `bundle_kind` | `bf16` |
| `distill_provenance.json` | `deployed_student` | `winner`; `checkpoint` and `predictions` (run-relative paths); `calibration` (`distill/select/calibration.json`); `attestation`, the per-file sha256 of the train-stage folder the student was copied from (what the final and edge attest), taken before the fitted temperatures were installed, with `manifest_sha256` null; `quantize`, why it was skipped |

`deployed_format` in `distill_provenance.json` and `bundle.json` is `bf16`, and
`quant_manifest_prereg_sha256` is null, because no quant manifest exists.

### The `dataset` folder

`bundle.json` (`kind: dataset`), `calibration.json`, `gate.json`, `scorer-train.json`,
`NOTICE` and `scan.json` as above, plus:

| File | Content |
|---|---|
| `train.jsonl` | the frozen training rows whose source licence is Apache-2.0, MIT, or cc-by-4.0 with an attribution |
| `dataset.json` | row counts by licence and target source, rows excluded by licence |

A cc-by-4.0 row without a `NOTICE` refuses the build and the upload.

## Quantization scheme per tensor class

The scheme is recorded in `jev_quant.json` as `scheme`; its fields and defaults
(`QuantScheme`) are:

- `block_bits` 4: weight-only int4 for the text-block linears (8 is the fallback when no int4 kernel runs on the target).
- `embedding_bits` 4: bits of the packed token embedding (4 or 8).
- `block_group_size` 256: weights per scale and zero in the blocks.
- `embedding_group_size` 128: values per scale in the embedding.

The build is torchao weight-only (`Int4WeightOnlyConfig` / `Int8WeightOnlyConfig`, or the
pre-0.10 functions; the one used is recorded as `torchao_api`) after the LoRA is merged into
the torso. By tensor class:

| Class | Modules | Treatment |
|---|---|---|
| Text-block linears | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` under `layers` | weight-only int4 or int8 per `block_bits` |
| Token embedding | `embed_tokens` | full-vocabulary (every row kept), packed: grouped symmetric int4 (two per byte) or int8, buffers `qweight` and `scales`, one scale per `embedding_group_size` values in the model dtype; a lookup is a gather and a multiply |
| Kept unquantized | `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, every `norm` | floating tensors as in the base |
| Dropped | `visual`, `vision_tower`, `vision_model`, `mtp` | removed before saving |
| Pointer head | `head.safetensors` | `float32` (fp32), its own file |

The saved state is checked against this policy (`check_inventory`) before it is written, and
`read_artifact` checks the hashes and the weight byte sum. The weight files
(`torso_quantized.pt` plus `head.safetensors`) must total under 10^9 bytes for
`within_size_budget`; an over-budget build is recorded unavailable by the quantize stage,
never bundled.

## Tokenizer

The tokenizer is the source checkpoint's (Qwen3.5-2B-Base family), copied byte for byte.
`jev_quant.json` records each file's sha256 and the source's; `read_artifact` refuses an
artifact whose tokenizer differs (`unmodified` must be true). The vocabulary is
full-vocabulary: the embedding keeps a row for every tokenizer id
(`check_full_vocabulary`), so no id is unreachable. A pruned-vocabulary build is a
comparison artifact for measuring the cost of the full embedding; it is never bundled and
this format has no field for it.

## Loader and serve entry point

Loader and server are one module, run in the pinned student environment:

```bash
python -m jev_factory.backbones.pointer.quantized serve ARTIFACT --host 127.0.0.1 --port 8000
```

Options: `--device` (default `cuda`), `--model-name`. The host must be loopback
(`127.0.0.1`); anything else is refused. The loader verifies the manifest and every hash,
builds the torso skeleton at the pinned `base_model_revision` with the same modules dropped,
installs the packed embedding, assigns the quantized state over it, loads the fp32 head and
wraps it in upstream's `SystemOneEngine` (strands-decider @ 8b6faa5, Apache-2.0).
`quantized.serve_argv` and `quantized.launch` build the same command and start it detached.
One engine call runs at a time.

### `POST /v1/systemone`

The request is upstream's `SystemOneRequest`: `{model, state, questions}`, with
`questions` keyed by id, each `{type, instructions?, criteria}` and `type` one of `choice`,
`noul`, `score`. The answer gives per-option `probabilities` for `choice` and `score`, and
`noul` as P(true), plus `latency_ms`. Only the probabilities are meant to be read; the
vendor `confidence` field is not a probability and never feeds a gate. Apply the model's own
`gate.json` and `calibration.json` on the caller side. A malformed body is a 422 with
`{"detail": ...}`; an engine `ValueError` is a 422; any other failure is a 500 with a
`detail`.

### `GET /v1/bundle`

The artifact's attestation, computed from the verified manifest at load:
`artifact_format`, `format`, `bits`, `scheme`, `files` (sha256 per file),
`base_model_revision`, `weight_files` (bytes per weight file), `weight_bytes`,
`size_budget_bytes`, `within_size_budget` and `manifest_sha256` (the sha256 of
`jev_quant.json`). A client compares these against the bundle it was given.

### `GET /v1/stats`

Host and process facts, read live from `/proc`, `/sys` and `/etc`: `rss_bytes`,
`mem_available_bytes`, `device_model`, `nv_tegra_release`, `cgroup` (memory limit and
usage as the cgroup reports them) and `accelerator` (the CUDA device name or null). A field
that cannot be read is null, not an error.

### `GET /health`

`status`, `model`, `checkpoint`, `base_model`, `num_slots`, `max_length`, `temperature`,
`device`, `prefix_cache`, `vision` (always false), `quant_format` and `torchao_api`.

An unknown path is a 404 with `{"detail": "Not Found"}`.
