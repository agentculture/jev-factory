# Training environment

The base install of `jev-factory` has `dependencies = []`. Heavy ML packages
live behind the `train` extra and are imported lazily by the verbs that use
them.

## Pins

The `train` extra mirrors `scripts/lfm-finetune/requirements-train.txt` in
nvsh, except `torch`. Python 3.12.

## torch

`torch` is deliberately not in the extra. Install it from the PyTorch CUDA 13.0
wheel index on the DGX Spark before installing the extra:

```bash
uv pip install --python <venv python> torch==2.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install --python <venv python> "jev-factory[train]"
```

## Evals

The release gate uses the `evals` dependency group (`deepeval==4.2.6`):
`uv sync --group evals`.

## Executor seam

Distill training (`train` and `heal`) runs through `jev_factory/distill/executor.py`.
The run-config key `distill_executor` picks the executor; `local` is the default and
the only one that exists, and any other name is refused. The local executor runs the
same capped GPU call as before, in the environment above.

A container or remote executor plugs in through the `JobSpec`: staged inputs with their
sha256, the expected outputs, the environment as an image digest plus a lockfile hash
(not a tag), the limits and the argv. The seam stages only train-side rows, refuses
protected ids, secrets and private addresses, and re-hashes every output on return.

The environment above is aarch64 with CUDA 13.0 (the GB10). A remote x86 image, for
example on SageMaker, needs its own lockfile and image digest. That is a later session;
no AWS code exists yet.
