"""``jev ask <request> --bundle <dir>`` -- propose one jev verb; execute nothing.

The bundle is a jev-tool model bundle (:mod:`jev_factory.release.bundle`). The verb
scores the request over the running CLI's own verbs as lettered candidates (one
forward pass, one token), applies the bundle's ``calibration.json``, gates the
calibrated distribution with its ``gate.json`` and prints exactly one of
``propose`` / ``explain`` / ``escalate`` / ``abstain_uncertain`` with its probability.
A proposal's arguments are grounded deterministically from the CLI catalog, never
taken from the model.

``ask`` is read-only in the strongest sense: it starts no process, writes no file
and never builds a ``--apply`` command line. A proposal is text for a human to run.
It also reports when the bundle was trained on a different CLI surface than the one
running (:func:`jev_factory.release.bundle.surface_mismatch`).

Inference is a running ``llama-server`` (``--server``) or, for a bf16 bundle, this
process through transformers; the heavy imports happen only in
:func:`build_scorer`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_diagnostic, emit_result
from jev_factory.core import calibration, gate, metrics
from jev_factory.domain.model import ESCALATE_LABEL, EXPLAIN_LABEL, Domain
from jev_factory.release import bundle as bundles

Render = Callable[[list[dict]], str]


def _load_json(folder: Path, name: str) -> Any:
    path = folder / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise CliError(
            EXIT_USER_ERROR,
            f"{folder} has no readable {name}",
            "pass a jev-tool bundle (it carries calibration.json, gate.json and bundle.json)",
        ) from None
    except ValueError as exc:
        raise CliError(EXIT_USER_ERROR, f"{path} is not valid JSON: {exc}", "") from None


def load_bundle(folder: Path) -> tuple[dict, gate.Thresholds, dict]:
    """``(calibration params, gate thresholds, bundle info)`` of *folder*."""
    if not folder.is_dir():
        raise CliError(
            EXIT_USER_ERROR, f"{folder} is not a folder", "pass --bundle <a jev-tool bundle dir>"
        )
    params = _load_json(folder, "calibration.json")
    thresholds_doc = _load_json(folder, "gate.json")
    info = _load_json(folder, bundles.BUNDLE_FILE)
    temperature = params.get("temperature", 1.0) if isinstance(params, dict) else None
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or temperature <= 0
    ):
        raise CliError(EXIT_USER_ERROR, "calibration.json needs a positive temperature", "")
    try:
        thresholds = gate.Thresholds.from_json(thresholds_doc)
    except gate.GateError as exc:
        raise CliError(EXIT_USER_ERROR, f"gate.json: {exc}", "") from None
    return params, thresholds, info


def build_scorer(args: argparse.Namespace, bundle: Path) -> tuple[sc.TopK, Render]:
    """The top-k source and prompt renderer for *bundle*; heavy imports happen here."""
    if getattr(args, "server", None):
        from transformers import AutoTokenizer  # lazy: never imported by the base install

        tokenizer = AutoTokenizer.from_pretrained(
            str(bundle)
        )  # nosec B615 -- a local bundle folder
        model = args.model or bundle.name
        return sc.served_top_k(args.server, model), lambda m: sc.render_prompt(tokenizer, m)
    if not (bundle / "config.json").is_file():
        raise CliError(
            EXIT_ENV_ERROR,
            f"{bundle} holds no checkpoint to run in this process",
            "serve it with llama-server and pass --server http://127.0.0.1:<port>/v1",
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(bundle))  # nosec B615 -- a local bundle folder
    model = AutoModelForCausalLM.from_pretrained(
        str(bundle), torch_dtype=torch.bfloat16
    )  # nosec B615 -- a local bundle folder
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    top_k = sc.InProcessTopK(tokenizer, sc.transformers_logprobs(model, tokenizer))
    return top_k, lambda m: sc.render_prompt(tokenizer, m)


def ask(
    domain: Domain,
    request: str,
    params: dict,
    thresholds: gate.Thresholds,
    top_k: sc.TopK,
    render: Render,
) -> dict[str, Any]:
    """Score, calibrate, gate and ground one request. Pure: it executes nothing."""
    pool = sc.candidate_pool(domain)
    prompt = render(sc.prompt_messages(domain, request, pool))
    pred = sc.score(domain, top_k, prompt, request, offered=pool)
    result: dict[str, Any] = {
        "request": request,
        "outcome": "abstain_uncertain",
        "operation": None,
        "probability": None,
        "reason": None,
        "arguments": None,
        "grounded": False,
        "grounding": None,
        "executed": False,
    }
    if pred.candidates is None:
        result["reason"] = "no usable distribution from the scorer"
        return result
    calibrated = calibration.apply_scaling(
        pred.candidates, float(params.get("temperature", 1.0)), params.get("vector") or {}
    )
    offered = [Domain.calibration_label(name) for _, name in pred.offered]
    decision = gate.decide(calibrated, offered, thresholds, domain)
    rolled = metrics.rollup_escalate_candidates(calibrated)
    result.update(outcome=decision.outcome, reason=decision.reason)
    result["probability"] = rolled.get(decision.label)
    if decision.outcome in ("propose", "abstain_uncertain") and domain.get(decision.label):
        result["operation"] = decision.label
        result["read_only"] = not domain.is_mutating(decision.label)
    if decision.outcome == "propose":
        found = sc.ground_arguments(domain, domain.get(decision.label), request)
        if isinstance(found, str):
            result["grounding"] = found
        else:
            result.update(arguments=found, grounded=True)
    if decision.outcome == "explain":
        result["label"] = EXPLAIN_LABEL
    if decision.outcome == "escalate":
        result["label"] = ESCALATE_LABEL
    return result


def render_text(doc: dict[str, Any]) -> str:
    p = doc["probability"]
    head = doc["outcome"] + (f"  p={p:.3f}" if p is not None else "")
    lines = [head]
    if doc["operation"]:
        kind = "read-only" if doc.get("read_only") else "mutating: dry-run first, you decide"
        lines.append(f"  verb: {doc['operation']} ({kind})")
    if doc["outcome"] == "propose":
        if doc["grounded"]:
            lines.append(f"  arguments: {json.dumps(doc['arguments'], sort_keys=True)}")
        else:
            lines.append(f"  arguments: not grounded ({doc['grounding']})")
    if doc["reason"]:
        lines.append(f"  reason: {doc['reason']}")
    lines.append("  executed: nothing (ask only proposes)")
    return "\n".join(lines)


def cmd_ask(args: argparse.Namespace) -> int:
    folder = Path(args.bundle).expanduser()
    params, thresholds, _info = load_bundle(folder)
    mismatch = bundles.surface_mismatch(folder)
    if mismatch is not None:
        if args.strict_surface:
            raise CliError(
                EXIT_ENV_ERROR,
                mismatch,
                "rebuild the bundle for this CLI, or drop --strict-surface to ask anyway",
            )
        emit_diagnostic(f"warning: {mismatch}")
    from jev_factory.domains.jev_cli.generate import generate_domain

    top_k, render = build_scorer(args, folder)
    doc = ask(generate_domain(), args.request, params, thresholds, top_k, render)
    doc["surface_mismatch"] = mismatch
    json_mode = bool(args.json)
    emit_result(doc if json_mode else render_text(doc), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ask",
        help="Propose one jev verb for a request from a jev-tool bundle; executes nothing.",
    )
    p.add_argument("request", help="What you want to do, in words.")
    p.add_argument("--bundle", required=True, help="A jev-tool model bundle folder.")
    p.add_argument("--server", help="OpenAI-style llama-server base URL serving the bundle.")
    p.add_argument("--model", help="Model name the server knows (default: the bundle folder).")
    p.add_argument(
        "--strict-surface",
        action="store_true",
        help="Fail instead of warning when the bundle's CLI surface hash differs.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_ask)
