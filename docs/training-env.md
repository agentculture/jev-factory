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
