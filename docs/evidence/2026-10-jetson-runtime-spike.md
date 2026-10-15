# Jetson runtime spike (issue #20, task t1)

Date: 2026-10-15. Board: Jetson Orin Nano Super 8GB. Purpose: settle frame park v1 and plan
risk r1: can the 2610 decider's torch path and a 4-bit weight-only kernel run on JetPack 6.2
(sm_87)? Only the Nano was used; nothing ran on the AGX Orin.

## Board

Read over `ssh nano` on 2026-10-15:

```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
5.15.148-tegra
MemTotal: 7802744 kB   MemAvailable: 6930092 kB   SwapTotal: 67108860 kB
NV Power Mode: 25W
```

## Image

Built on the DGX Spark with jetson-containers (`~/git/jetson-containers-nv` at `c8bad15f`),
cross-built for JetPack 6 and loaded on the Nano with `docker save | zstd | ssh nano docker load`:

```bash
CUDA_ARCH=tegra-aarch64 L4T_VERSION=36.4.4 CUDA_VERSION=12.6 LSB_RELEASE=22.04 PYTHON_VERSION=3.10 \
  ./build.sh --skip-tests=all --name=jev-spike pytorch transformers torchao
# -> jev-spike:r36.4.tegra-aarch64-cp310-cu126-22.04 (12.5 GB, 15 min build)
```

The `--platform=jp6/cu126` preset did not take effect on this host (the simulated build
targeted r38.2 sbsa); exporting the five variables did.

## Run

The base `Qwen/Qwen3.5-2B-Base` at `b1485b2` was copied to the Nano. The script
(`docs/evidence/scripts/2026-10-jetson-runtime-spike.py`) loads only the
text torso (`model.language_model.*`, 24 layers: 18 Gated DeltaNet, 6 full attention),
streaming each tensor onto the GPU in bf16, then runs three timed forward passes of 512 random
token ids. It then rebuilds the torso and applies torchao weight-only int8, then int4
(group 128), to every `nn.Linear` except `in_proj_a` and `in_proj_b`. The cosine compares the
last token's hidden state with bf16's.

```bash
docker run --rm --runtime nvidia --memory 6g --memory-swap 6g -e SPIKE_SEQ=512 \
  -v ~/jev-spike/pin-cache:/data/pin-cache:ro -v ~/jev-spike/spike.py:/spike.py:ro \
  jev-spike:r36.4.tegra-aarch64-cp310-cu126-22.04 python3 /spike.py /data/pin-cache/Qwen__Qwen3.5-2B-Base
```

Output (one JSON line per step, abridged to the measured fields):

```text
facts        python 3.10.22, torch 2.11.0 (CUDA 12.6), transformers 5.19.0, torchao 0.13.0,
             device Orin, capability [8, 7], arch_list ['sm_87']
load_bf16    ok, missing [], unexpected [], layer types full_attention + linear_attention
deltanet     fla importable: false (transformers' torch fallback for the Gated DeltaNet)
forward_bf16 ok, ms [1259.9, 471.4, 440.3], peak_alloc 3684 MiB
forward_int8 ok, ms [633.6, 612.4, 611.2], peak_alloc 2386 MiB, last-token cosine vs bf16 0.99888
forward_int4 ok, ms [1878.3, 1900.2, 1915.9], peak_alloc 3080 MiB, last-token cosine vs bf16 0.93505
```

The first run failed in the script, not the stack: building the torso on the meta device left
its rotary buffers empty (`Cannot copy out of meta tensor`). The loader was changed to build on
the GPU and stream weights; the run above is the second.

## Verdict per format

| Format | Works on sm_87 | Warm forward, 512 tokens | Note |
|---|---|---|---|
| bf16 torso, DeltaNet torch fallback | works | ~440 ms | no `fla` / `causal-conv1d` needed |
| torchao int8 weight-only | works | ~611 ms | slower than bf16; cosine 0.9989 |
| torchao int4 weight-only (group 128) | works | ~1.9 s | ~4x slower than bf16; cosine 0.935 before any heal |

## What this means for the plan

- Plan risk r1 (frame park v1) is answered: the torch path and both torchao weight-only kernels
  run on the Nano under JetPack 6.2. Int4 is not blocked.
- Int4 with torchao 0.13 is slow on the Nano (about 4x bf16), so torchao's int4 kernel likely
  falls back to a generic path on sm_87. Latency is reported, not gated (decision c26), but the
  announcement's "runs fast" is at risk with this kernel.
- The int4 cosine (0.935, random tokens, no heal, every block linear quantized) says the
  quant needs the heal round (c10) and a real agreement measurement on the selection fold.
- Peak allocation during the int4 build was 3.1 GB, with MemAvailable down to 343 MiB inside a
  6 GB cap: building on the board is tight. Build on the Spark or the AGX Orin; the board only
  serves.
- Not measured here: the packed int4 embedding (t10), the pointer head, a real 2610 prompt,
  records up to 1,024 tokens, and the AGX Orin (no run there without the operator's go).
