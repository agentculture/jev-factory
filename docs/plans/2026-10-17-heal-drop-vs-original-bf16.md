# Build Plan — heal drop vs original bf16

slug: `heal-drop-vs-original-bf16` · status: `exported` · from frame: `heal-drop-vs-original-bf16`

> A healed quant's accuracy drop is gated against the original selected bf16 student, with the healed-bf16 drop and the heal drift recorded beside it as context

## Tasks

### t1 — Rebase the healed quant's drop on the original bf16 student in quant.py (`_against_original`, `heal_result` fields, `_healed_problem` message, manifest bf16 fields) with the new tests

- instruction: uv run pytest -n auto; uv run black --check `jev_factory` tests; uv run flake8 `jev_factory` tests
- covers: c1, h1, c2, h2, c3, h3, c4, h4, c5, h5, c6, h6, c7, h7, c8, h8, c9, h9, c10, h10, c11, h11, c16, h14, c17, h15, c18, h16, c19, h17, c20, h18, c21, h19, c23, h20, c24, h21, c25, h22
- acceptance:
  - tests/`test_distill_quant.py`: the lowers/raises/gated heal tests pass and the lowers/raises ones were red before the change
  - full suite and CI lint pass

### t2 — Record d17 on the #20 plan, update docs/distill.md and the #20 delivery follow-up, file the jev run heal follow-up issue, bump the version

- depends on: t1
- covers: c12, h12, c13, h13
- acceptance:
  - d17 approved by the operator before the code commit
  - markdownlint-cli2 passes on the edited docs
