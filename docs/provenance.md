# Provenance of imported nvsh modules

Modules cited (copied) from [nvsh](https://github.com/agentculture/nvsh)
carry a module-level `NVSH_PROVENANCE` dict. There is no shared ledger
file, so import tasks stay file-disjoint. `docs/nvsh-import-provenance.md`
is generated from these headers.

```python
NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/split.py",  # path inside the nvsh repo
    "commit": "9debdc6",  # nvsh commit the copy was taken from
    "adaptations": ["imports rewired to jev_factory seams"],  # non-empty list of str
    "licence": "Apache-2.0",
}
```

The value must be a plain literal (the test reads it with `ast`, it never
imports the module), and `adaptations` must be a non-empty list of strings.

## Which modules need it

- Every non-`__init__.py` module under the imported packages:
  `jev_factory/core/`, `data/`, `measure/`, `release/`,
  `backbones/causal_lm/` and `evals/`.
- Any module elsewhere that defines `NVSH_PROVENANCE` (it is then
  validated the same way).

`tests/test_provenance.py` fails when such a module lacks the dict, has a
malformed one, or names an upstream file that does not exist. The upstream
check needs an nvsh checkout: set `NVSH_ROOT`, or keep it at `../nvsh`.
Without one, that check is skipped and only the shape is enforced.
