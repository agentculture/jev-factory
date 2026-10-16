# Jetson-ops proving run (issue #20, task t23): state at pause

Written 2026-10-16, when the operator paused the work. Read this first in the
next session. It records what ran, what it measured, what is still running,
and how to pick up each open thread. Branch: `spec/jetson-ops-nano-issue-20`
(not pushed at the time of writing).

**Read every number below as UNVERIFIED.** All gold (train, fit, selection,
held-out, test) is model-drafted and no human has approved it (deviation
d10). The run proves the process; it makes no quality or ship claim.

## Where things are

| What | Where |
|---|---|
| Run work dir | `~/jev-runs/jetson-ops-20/work` (config `run.toml`, stages under `distill/`) |
| Shipped artifact | `distill/quant/builds/torch_int4_wo` (un-healed int4, 985,675,969 weight bytes) |
| Bundles (not uploaded) | `distill/bundle/model` (960 MB), `distill/bundle/dataset` (frozen train set); both scans clean |
| Heal evidence | `distill/quant.round1-snapshot`, `quant.round2-snapshot`, `quant.round3-snapshot` |
| First final (Clef 401) | `distill/final.run1` |
| Older checkpoints and bundles | `distill/train-c000.*`, `distill/bundle.scan-pre-0fd1c88` |
| Decisions | `.devague/deliveries/orin-nano-jetson-ops-decider.json` (deviations d7 to d16) |

Every stage from `preregister` to `bundle` is `complete`; `upload` has not
run (private-first: it waits for the operator).

## Results (unverified)

Jetson-ops test side, 248 questions from clean sources:

| Model | Top-1 | Notes |
|---|---|---|
| Clef W4A16 (teacher) | 0.952 | |
| stock 2610 | 0.718 | |
| bf16 student | 0.968 | +25.0 points over stock; 1.6 above Clef |
| int4 (shipped, served on the AGX Orin) | 0.952 | +23.4 over stock; ties Clef; ECE 0.050 (stock 0.060) |

The c17 bars: +10 over stock passes for both builds, within 5 points of Clef
passes for both, weight bytes pass. **One miss:** int4 versus bf16 agreement
on test is 0.960 (bar 0.97, CI 0.935 to 0.984).

The typed-decisions guard is reported, never gated (d14). On its 1,000 test
questions: Clef 0.690, stock 0.596, bf16 0.622, int4 0.545. The int4 build
loses general skill against stock; on Jetson-ops it loses nothing.

Edge on the AGX Orin (d16; the fit fold stands in for the private probe set):
the int4 answers 99.3% of selection questions (877/883) as it does on the
Spark; median latency 0.39 s, p95 1.55 s, through the tunnel; peak RSS
5.81 GB under an 8 GB cgroup cap; no failures. That peak predates the r9 fix
below.

## Why the shipped int4 is un-healed

Three heal rounds ran: d11 (bf16-only targets), d13 (Jetson-ops rows repeated
3x, guard rows 2x). Each one lowered Jetson-ops agreement with the bf16
student, from 0.976 before any heal to 0.961 to 0.964, while Jetson-ops
accuracy held or rose. Only the bf16-target heal recovered much of the guard.
With the rules scoped to the domain (d12), the un-healed int4 passes every
Jetson-ops bar, so the operator chose to ship it (d14).

## Fixed during the run (all committed with tests)

| Commit | What was wrong |
|---|---|
| `6048f3d` | The trainer saved the pointer head as a pickle that export could not read |
| `378b368`, `64fb92c` | `--allow-foreign-gpu` did not reach the scorer, the measurer or the int4 build guard |
| `052831a` | Score answers mapped to levels by position, so any score question read from a saved file was scored against the wrong level |
| `5679a19` | `benchmark` and `final` asked Clef without the bearer key and model header, so every request got HTTP 401 |
| `0fd1c88` | The bundle scan could not read a `torch.save` archive; it now checks the zip layout and walks the pickle without loading it |
| `a17dc91`, `14be91f` | Risk r9: the quant server loaded the bf16 base before swapping in int4; it now fills an empty (meta) torso, answers unchanged (883/883) |

New deviation-gated knobs on `quantize`: `heal_mix`, `gating=clean|clean-report`
and `heal_repeat`. One environment fix: the student venv had an older
strands-decider than the pinned source; the pinned one is installed (backup
under `~/jev-runs/envs/`) and the lockfile regenerated.

## Still running

On the AGX Orin, the shipped int4 serves under a cgroup cap (the operator
keeps it for now):

```bash
ssh orin 'systemctl --user status jev-quant-final'     # port 18092, MemoryMax 8G
ssh orin 'systemctl --user stop jev-quant-final'       # when it is no longer wanted
```

On the Spark, as background tasks of the pausing session (gone when that
session ends): the tunnel to the AGX Orin quant (`127.0.0.1:18092`), the
tunnel to Clef (`127.0.0.1:18090`), the stock 2610 server (`127.0.0.1:18071`)
and the draft review site. To bring the tunnels and stock server back:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18090:127.0.0.1:8000 orin   # Clef
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18092:127.0.0.1:18092 orin  # int4
cd ~/jev-runs/ext/strands-decider && HF_HUB_CACHE=~/jev-runs/hf-cache HF_HUB_OFFLINE=1 \
  PYTHONPATH=src ~/jev-runs/envs/decider/bin/python -m strands_decider.cli serve \
  ~/jev-runs/pin-cache/StrandsAgents__strands-decider-2B-qwen3.5-v1-2610 \
  --host 127.0.0.1 --port 18071 --model-name stock-2610 --strict-window
```

Stages that need Clef run with the key injected by name, never written down:
`grant run --inject DISTILL_TEACHER_KEY=LOBES_GATEWAY_API_KEY -- uv run jev distill <stage> ...`.

## Paused: serving on the Orin Nano

The operator stopped this part, to test on another day. Nothing runs on the
Nano. Staged there under `~/jev-runs`:

- `src/jev-factory` at `14be91f` and `src/strands-decider` at `8b6faa5`
  (`git archive`);
- `hf-cache`: the base model's config and tokenizer only, no weights (the r9
  fix builds the torso from the config, so none are needed);
- `artifacts/jops20-int4-d14`: the shipped int4 (compare its sha256 with the
  Spark copy before use);
- image `jev-quant-nano:r36.4`: the spike image plus `torchao==0.16.0`. The
  base image pins torchao 0.13 through a pip constraints file, so the install
  needs `PIP_CONSTRAINT=`. The wheel is pure Python and imports with the
  image's torch 2.11.

To resume:

1. Re-ship `jev-factory` at the current HEAD if it changed.
2. Start the server under a memory cap (4 GB is the run config's original
   edge cap), for example
   `docker run -d --name jev-quant-nano --runtime nvidia --network host --memory 4g -v ~/jev-runs:/jev-runs:ro -e HF_HUB_CACHE=/jev-runs/hf-cache -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/jev-runs/src/jev-factory:/jev-runs/src/strands-decider:/jev-runs/src/strands-decider/src jev-quant-nano:r36.4 python3 -m jev_factory.backbones.pointer.quantized serve /jev-runs/artifacts/jops20-int4-d14 --host 127.0.0.1 --port 18093 --model-name distill-final-student`.
   Check `/v1/stats` reports the cgroup cap; on a Jetson, GPU allocations may
   not be charged to it, so watch `MemAvailable` too.
3. Tunnel `127.0.0.1:18093` to the Nano, set `distill_edge_url` and
   `distill_edge_memory_cap` in `run.toml` (it now points at the AGX Orin
   server with 8G; a backup sits beside it), and run
   `jev distill edge --knob label=orin-nano --knob served_model=distill-final-student --deviation-id d16 --apply --force`.
4. Things to watch: peak memory (the first board measurement of the r9 fix),
   and latency (the runtime spike saw int4 near 1.9 s per 512 tokens on the
   Nano, risk r6/r13).

## Next steps

1. **PR (operator approved).** Bump the version, update the state text in
   `CLAUDE.md` and the three sibling prompt files (a #20 proving-run model now
   exists, unverified), run `/validate-delivery` and `/summarize-delivery`,
   push the branch and open the PR.
2. **Upload** stays private-first and waits for the operator.
3. **Open threads:** operator review and approval of test gold (for a
   verified headline); the pre-review gold cross-check by two other models;
   the real probe set for `edge`; the Nano run above; the JetPack 7.2 L4T
   topic text (should be R39); lapses l1 to l9 and o18 for the operator to
   adjudicate.
