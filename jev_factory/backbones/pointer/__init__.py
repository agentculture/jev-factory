"""The pointer backbone: Strands Decider, a 2B LoRA torso with a pointer head.

A pointer head scores each named option from its own hidden state, so there is no letter
readout and no letter-token logprob anywhere in this family: a row's distribution is the
per-option ``probabilities`` a served ``strands-decider serve`` returns on /v1/systemone,
read by :mod:`jev_factory.distill.systemone`. strands_decider is never imported in-process;
it runs as a localhost subprocess in the pinned student env (:mod:`.serve`). Bundles are the
``hobson-hf-export/1`` layout (:mod:`.layout`).

Registered as ``pointer`` in :mod:`jev_factory.backbones.registry` (attribute ``BACKBONE``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_factory.backbones.pointer.layout import (
    FORMAT,
    REQUIRED_FILES,
    plan_export,
    read_layout,
    write_layout,
)
from jev_factory.backbones.pointer.serve import (
    PointerClient,
    launch,
    serve_argv,
    wait_ready,
)
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill.pins import DECIDER_REPO

__all__ = [
    "BACKBONE",
    "FORMAT",
    "QUESTION_ID",
    "REQUIRED_FILES",
    "PointerBackbone",
    "PointerClient",
    "launch",
    "plan_export",
    "read_layout",
    "serve_argv",
    "wait_ready",
    "write_layout",
]

#: The one choice question each rendered row asks.
QUESTION_ID = "decide"
INSTRUCTIONS = (
    "Which one of these actions should handle the request? Choose explain to answer in "
    "words, or escalate to hand it off, when no action fits."
)
#: Defaults for the two controls when the domain gives no text of its own.
CONTROL_DEFAULTS = {
    "explain": "Answer the request in words; run no action.",
    "escalate": "Hand the request off; no offered action fits it safely.",
}


def _description(domain: Any, name: str) -> str:
    operation = domain.get(name)
    if operation is not None:
        return operation.description
    if name.startswith("escalate:"):
        text = domain.reason_descriptions().get(name)
        if text is not None:
            return text
    if name in CONTROL_DEFAULTS:
        return domain.control_description(name) or CONTROL_DEFAULTS[name]
    raise CliError(EXIT_USER_ERROR, f"{name!r} is not a candidate of domain {domain.name!r}")


class PointerBackbone:
    """The :class:`~jev_factory.backbones.base.Backbone` for Strands Decider."""

    name = "pointer"

    def render(self, domain: Any, request: str, offered: Sequence[str]) -> dict[str, Any]:
        """One /v1/systemone record: the request as state, one choice over named options."""
        names = [str(n) for n in offered]
        if len(names) < 2 or len(set(names)) != len(names):
            raise CliError(EXIT_USER_ERROR, "a pointer row needs at least 2 distinct candidates")
        criteria = {n: _description(domain, n) for n in names}
        return {
            "id": "",
            "state": request,
            "questions": {
                QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}
            },
        }

    def predict_distributions(
        self, model: Any, rendered: Sequence[Any], offered: Sequence[Sequence[str]]
    ) -> list[Mapping[str, float]]:
        """Per-row probabilities from the served model; *model* is a client or a base URL.

        Any row the server does not answer with a proper distribution fails the call: a
        missing row is never filled with a guess.
        """
        if len(rendered) != len(offered):
            raise CliError(EXIT_USER_ERROR, "rendered and offered differ in length")
        client = model
        if isinstance(model, str):
            try:
                client = PointerClient(model, model=self.name)
            except ValueError as exc:
                raise CliError(EXIT_USER_ERROR, f"refusing the server: {exc}") from exc
        out: list[Mapping[str, float]] = []
        for i, (record, names) in enumerate(zip(rendered, offered)):
            criteria = list(record["questions"][QUESTION_ID]["criteria"])
            if criteria != [str(n) for n in names]:
                raise CliError(EXIT_USER_ERROR, f"row {i}: offered {list(names)} != rendered")
            result = client.ask({**record, "id": record.get("id") or f"row-{i}"})
            answer = {q.qid: q for q in result.questions}.get(QUESTION_ID)
            if answer is None or not answer.ok or answer.distribution is None:
                why = answer.error if answer is not None else result.error
                raise CliError(EXIT_ENV_ERROR, f"row {i}: no distribution from the server: {why}")
            out.append(dict(answer.distribution))
        return out

    def train_plan(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Dry-run: the upstream trainer (``python -m training.train``, run from a
        strands-labs/strands-decider checkout in the trainer env; not part of the wheel)."""
        python = str(config.get("python", "python"))
        argv = [python, "-m", "training.train", "--config", str(config.get("config", "<yaml>"))]
        for key, flag in (("train_file", "--train-file"), ("output_dir", "--output-dir")):
            if config.get(key):
                argv += [flag, str(config[key])]
        return {
            "backbone": self.name,
            "dry_run": True,
            "argv": argv,
            "cwd": config.get("repo", "<strands-decider checkout>"),
            "head_type": "pointer",
            "init_from": str(config.get("init_from", DECIDER_REPO)),
        }

    def merge_verify(self, adapter_dir: str, base: str) -> Mapping[str, Any]:
        """Nothing is merged: the server loads the LoRA unmerged. Verify the bundle and base."""
        if not (isinstance(base, str) and base.strip()):
            raise CliError(
                EXIT_USER_ERROR,
                f"merge_verify needs the expected base model id, got {base!r}",
                "pass the base model the bundle must have been trained on",
            )
        info = read_layout(adapter_dir)
        # Each file is checked on its own: a set comparison let a bundle that records no
        # base at all (both None) "match" an expected base of None.
        recorded = {
            "strands_decider_config.json": info["config"].get("base_model"),
            "provenance.json": info["provenance"].get("base_model"),
        }
        if wrong := {name: value for name, value in recorded.items() if value != base}:
            raise CliError(
                EXIT_USER_ERROR,
                f"bundle base {wrong} does not match the expected base {base!r}",
                "verify against the base the adapter was trained on, or re-export the bundle",
            )
        return {
            "merged": False,
            "reason": "strands-decider serves the LoRA adapter unmerged (PeftModel)",
            "base": base,
            "base_model_revision": info["provenance"].get("base_model_revision"),
            "checked": ["layout", "format", "head_type", "adapter_sha256", "head_sha256", "base"],
        }

    def export(self, model_dir: str, out_dir: str, *, apply: bool = False) -> Mapping[str, Any]:
        """Write a ``hobson-hf-export/1`` bundle (dry-run unless *apply*)."""
        plan = plan_export(model_dir, out_dir)
        if not apply:
            return {
                "apply": False,
                "out": str(Path(out_dir)),
                "would_write": plan["copy"] + plan["write"],
            }
        info = write_layout(model_dir, out_dir)
        return {"apply": True, "out": info["root"], "format": FORMAT, "files": info["files"]}

    def quantize_candidates(self, config: Mapping[str, Any]) -> Sequence[str]:
        """The served build is bf16 (``torch_dtype`` in the config); no GGUF path exists."""
        return ["bf16"]


BACKBONE = PointerBackbone()
