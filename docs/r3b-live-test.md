# r3b live test of the evaluation path

Deviation d13 allowed nvsh's shipped scorer r3b to live-test jev-factory's
evaluation path before the first jev-tool build. This page records that run
of 2026-10-01. **r3b is reference context only.** It is not an acceptance
gate or a parity target for the jev-tool model.

## What ran

- **Model:** nvsh's `scorer-r3b` Q4_K_M build.
  - Published as `jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf`.
  - The GGUF's sha256 is `3568f660…4ea3`, the same as the published bundle's.
  - The tokenizer and chat template come from the bf16 checkpoint it was
    quantized from. Both match the bundle.
- **Serving:** CPU only, through `jev_factory.measure.serve`.
  - Set with `JEV_MEASURE_GPU_LAYERS=0` and nvsh's llama.cpp build.
  - Each decision took about 0.9 s.
- **Domain:** an nvsh operations domain written as a JSON file from nvsh's
  operation table, its escalation reasons and its instruction. It is not
  committed. It sets `control_descriptions` to nvsh's `explain` wording.
  - Prompts rendered by the factory are byte-identical to nvsh's, for the full
    candidate pool and for a missing-candidate pool.
- **Data:** nvsh's v2 validation side only.
  - 204 entries, plus their 84-entry missing-candidate slice.
  - Grounded on nvsh's own snapshot.
  - The probe ran on the 61-entry selection fold.
  - No test or held-out file was opened.
- **Steps:**
  1. `measure` on the full side, then on the missing-candidate slice.
  2. The permutation `probe`, which took 49 minutes on CPU.
  3. The release gate, replaying jev's predictions next to nvsh's own saved
     r3b predictions under the `val` tags.

## Results

Validation side, 204 entries:

| | jev-factory, live | nvsh, saved predictions |
|---|---|---|
| Right proposals, model only | 78/84 | 79/84 |
| Right proposals, r3b's shipped calibration and gate | 76/84 | 76/84 |
| Escalation (abstain) recall | 94.0% | 94.0% |
| False-positive tool calls | 1/120 | 1/120 |
| Wrong mutating | 0 | 0 |
| ECE, model only, then shipped | 0.035, then 0.011 | 0.039, then 0.020 |
| Missing-candidate escalation, model only | 73.8% | 73.8% |
| Missing-candidate escalation, shipped | 79.8% | 78.6% |

- 203 of 204 rows have the same top choice. The median difference in the top
  probability is 0.00003.
- The one row that differs is `q53-draft-v2-eval-op-service_status-003`, a
  near-tie that flipped:
  - nvsh: `service_status` 0.445 against `(escalate)` 0.426;
  - here: `(escalate)` 0.459 against `service_status` 0.397.

  This fits numeric differences between CPU serving here and nvsh's GPU
  serving. r3b's shipped read-only margin of 0.2 declines that row on both
  sides.
- The permutation probe's pooled change rate is **2.58%**:
  - order 1.5%, letters 1.5%, subset 2.8%, paraphrase 3.0%;
  - all four at once: 4.2%.

  This is on the selection fold, so it doesn't compare directly with nvsh's
  test-side probe.

## Gaps the run found, now fixed

- **The explain/escalate prompt text was a constant.** A domain can now carry
  its own text (`Domain.control_descriptions`, part of the surface hash).
- **`measure --serve` always used the GPU.** `measure_gpu_layers` set to `0`
  now serves on the CPU. It hides the GPUs too, because `--device none`
  alone still took about 170 MiB, and it skips the GPU residency guard.
- **The release gate had no validation tag.** It now accepts `val` and
  `val-mc`.
- **The gate report's missing-candidate column was misleading.** It showed
  only the slice's share of rows. It now also shows the slice's escalation
  rate, which is what the 80% bar is measured on.
- **Long steps were silent.** Measurements and probes now write progress
  files, and `jev status --watch` reports them every 30 minutes (deviation
  d15). The 49-minute probe in this run had no progress output.
