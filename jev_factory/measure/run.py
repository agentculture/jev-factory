"""Measure a candidate scorer on a named split: the predictions file, metrics and a results page.

Usage (one invocation measures every ``--model`` back to back, with identical
settings except the model)::

    python -m jev_factory.measure.run --domain <module|file.json> --run-dir RUN \\
        --split RUN/splits/val.json --scorer in-process \\
        --model RUN/merged --revision <commit> --label val-r1

    # the deployed Q4_K_M build, served by llama-server for this run only
    python -m jev_factory.measure.run --domain ... --run-dir RUN --split RUN/splits/test.json \\
        --final --scorer served --serve RUN/quant/model.Q4_K_M.gguf --llama-server BIN \\
        --tokenizer RUN/merged --model model.Q4_K_M --revision <sha> --label final-q4

Each entry is scored once through the causal-LM adapter
(:func:`jev_factory.backbones.causal_lm.scorer.score`): the domain's lettered
prompt, one next-token read of :data:`~jev_factory.backbones.causal_lm.readout.READOUT_TOP`
log-probabilities, arguments grounded outside the model. Every line is written
in the shared predictions schema (:mod:`jev_factory.measure.predictions`) and
scored by :func:`jev_factory.core.metrics.compute`, so every model is compared
on one set of figures. The request text is the entry's ``text``, exactly as
:mod:`jev_factory.data.assemble` trains on it.

Scorers:

* ``--scorer in-process``: the model loaded in this process (transformers,
  imported only then); every letter of the alphabet is read.
* ``--scorer served --base-url URL --max-logprobs N``: attach to a server the
  operator started; ``N`` (the engine's ``--max-logprobs``) must cover
  ``READOUT_TOP``.
* ``--scorer served --serve MODEL``: start the server for this run
  (:mod:`jev_factory.measure.serve`: vLLM by digest for a directory,
  llama-server for a ``.gguf``), wait for it, measure, stop it.

A served run is preflighted before the first entry (the server must list the
model at the labelled context, :mod:`jev_factory.measure.preflight`). A scorer
call that fails (a server that died) is a ``tier_error`` line, counted apart
from a model that put no mass on any letter; a run with more tier errors than
``--allow-tier-errors`` keeps its predictions for debugging but writes no
results page. An incomplete readout (a letter missing from the top
log-probabilities) is counted and carries no distribution, never one
renormalised over the letters that came back.

Guards:

* the split's side comes from BOTH its file name and its header: the sealed
  held-out set needs ``--acceptance``, the test side ``--final`` (each flag
  refused on any other file), and a file neither places is refused unless it
  is the domain's seed corpus;
* **each sealed side is measured once** (:mod:`jev_factory.measure.once`): a
  second ``--final`` or ``--acceptance`` measurement in the same run
  directory without ``--deviation <id>`` exits 1 before any model starts;
* a run in which every model's start-up failed measured nothing: its page is
  marked :data:`NOT_MEASURED_MARKER`, no once-ledger record is written, and a
  re-run with the same label replaces the page without ``--force`` (nvsh
  P71). A page from a run that measured anything is never overwritten
  without ``--force``;
* ``--details``-style per-entry dumps do not exist here: a predictions line
  carries ids, expected blocks and outcomes, never request text;
* a stage that starts a model (in-process or ``--serve``) first refuses a GPU
  held by a compute process this run did not start
  (:func:`jev_factory.factory.gpu.assert_gpu_free`) unless
  ``--allow-foreign-gpu``;
* ``--calibration PARAMS`` rescales every line's ``candidates`` (temperature,
  then vector) before metrics, keeping ``raw_probabilities``; params whose
  recorded ``predictions_source`` names the test or held-out split are
  refused. Decisions are never rescaled;
* grounding is against one fixed snapshot (``--ground-snapshot``, the
  domain's world schema, sha256 recorded), the split's own world, a
  ``--world`` corpus, the seed corpus's world, or ``--live`` lookups.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import shlex
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory import __version__
from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.core import metrics
from jev_factory.core.calibration import apply_scaling, split_markers
from jev_factory.core.predictions import Prediction, PredictionError, write_predictions
from jev_factory.domain.model import CONTROLS, Domain
from jev_factory.measure import once, serve
from jev_factory.measure.corpus import CorpusEntry, load_raw
from jev_factory.measure.predictions import TIER_ERROR_REASON, to_prediction
from jev_factory.measure.preflight import PreflightError, preflight_models
from jev_factory.measure.slices import missing_candidate_slice
from jev_factory.measure.snapshot import SnapshotError, load_snapshot

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/measure.py",
    "commit": "9debdc6",
    "adaptations": [
        "Track B (candidate scorer) path only: score_one, scorer_predictions, scorer_request,"
        " _WatchedScorer, readout_counts, --calibration, --reasons, --slice, the split-side"
        " guard, P71 and the tier-error gate (lines 1419-1781, 2022-2063, 2307-2617,"
        " 2649-3150); Track A (generative LfmTier/bench runs, RecordingChat, think blocks,"
        " generative_candidates) and the nvsh tier runtime are not imported",
        "nvsh imports (lines 171-191: nvsh.config, ops ground/table, platform, redact, tiers"
        " bench/lfm/toolchat/base/router/runtime/runtime_docker) are replaced by the Domain,"
        " the causal-LM adapter's score()/served_top_k/InProcessTopK, core.metrics,"
        " core.calibration, release.scan.redact and measure.serve",
        "_sibling/importlib path loading of calibration_fit/metrics/scorer/eval_slices/"
        "measure_skills replaced by package imports",
        "nvsh's managed launcher could not pass --max-logprobs, so a served scorer had to"
        " attach; here --serve starts vLLM (by digest, --max-logprobs >= READOUT_TOP) or"
        " llama-server itself, and attach needs --base-url with --max-logprobs",
        "revision verification against the nvsh HF cache (lines 786-857) is dropped: a"
        " served revision is operator-supplied and a served GGUF's sha256 is recorded",
        "a second final/held-out measurement without --deviation is refused through the"
        " once-ledger (nvsh only counted earlier final pages, lines 2767-2782)",
        "results go under --run-dir/measure/ instead of docs/benchmarks/; --details and"
        " the docker container guard are dropped; the GPU residency guard is added",
    ],
    "licence": "Apache-2.0",
}

EXIT_OK = 0
EXIT_USER = 1
EXIT_ENV = 2

FINAL_MARKER = "- Final run: yes"
#: A results page from a run in which every model's start-up failed (nvsh P71).
NOT_MEASURED_MARKER = (
    "- Run status: nothing measured (every start-up failed); a re-run with the same"
    " label replaces this page"
)

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SEED_RE = re.compile(r"seed=(\d+)")
_RUN_TIMEOUT = 30.0

SCORER_SERVED = "served"
SCORER_IN_PROCESS = "in-process"
READOUT_COMPLETE = "readout_complete"
READOUT_INCOMPLETE = "readout_incomplete"
READOUT_NO_MASS = "readout_no_label_mass"
READOUT_CALL_ERROR = "readout_call_error"
NO_DISTRIBUTION = "no distribution: "
REVISION_IN_PROCESS = "passed to from_pretrained in-process"
REVISION_UNVERIFIED = "operator-supplied, not verified"

SLICE_FULL = "full"
SLICE_MISSING = "missing-candidate"

HELD_OUT = once.HELD_OUT
TEST = once.TEST
DEV = "dev"
_NAMED_SIDES = ("train", "val", TEST)
_SPLIT_NOTE_RE = re.compile(r"Split '([A-Za-z0-9_-]+)' of (\S+?) \(seed=")
_HELD_OUT_HEADER_START = "held-out split"
_UNREAD = object()

RunFn = serve.RunFn
Render = Callable[[list[dict]], str]


class MeasureError(Exception):
    """A refusal: printed as one line plus a hint, never a traceback."""

    def __init__(self, code: int, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# ---------------------------------------------------------------------------
# Split file: which side it is
# ---------------------------------------------------------------------------


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sides_from_name(path: Path) -> set[str]:
    """What the file name says: ``test``/``val``/``train`` as a word, or held-out."""
    stem = path.stem.lower()
    words = {word for word in re.split(r"[^a-z0-9]+", stem) if word}
    sides = {side for side in _NAMED_SIDES if side in words}
    if _compact(HELD_OUT) in _compact(stem):
        sides.add(HELD_OUT)
    return sides


def _header_text(header: object) -> str:
    if header is None:
        return ""
    if isinstance(header, str):
        return header
    return json.dumps(header, sort_keys=True)


def sides_from_header(header: object) -> set[str]:
    """What the header says: every ``split.py`` note, and a held-out file's own header."""
    text = _header_text(header)
    sides: set[str] = set()
    for side, corpus in _SPLIT_NOTE_RE.findall(text):
        sides.add(side.lower())
        if _compact(HELD_OUT) in _compact(corpus):
            sides.add(HELD_OUT)
    if text.lstrip().lower().startswith(_HELD_OUT_HEADER_START):
        sides.add(HELD_OUT)
    return sides


def _read_header(path: Path) -> object:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw.get("header") if isinstance(raw, dict) else None


def _same_file(path: Path, other: Path | None) -> bool:
    if other is None:
        return False
    try:
        return path.resolve() == Path(other).resolve()
    except OSError:
        return False


def check_split_allowed(
    path: Path,
    *,
    acceptance: bool,
    final: bool,
    header: object = _UNREAD,
    dev_corpus: Path | None = None,
) -> str | None:
    """Refuse held-out without ``--acceptance`` and test without ``--final``; returns the
    sealed side measured (``"test"``/``"held-out"``) or ``None``.

    The side comes from BOTH the file name and its header, so renaming a file
    does not change what it is. A file neither places is refused unless it is
    *dev_corpus* (the domain's seed corpus).
    """
    if header is _UNREAD:
        header = _read_header(path)
    name = path.name
    by_name, by_header = sides_from_name(path), sides_from_header(header)
    sides = by_name | by_header
    if _same_file(path, dev_corpus):
        sides.add(DEV)

    def origin(side: str) -> str:
        return " and ".join(
            where for where, found in (("name", by_name), ("header", by_header)) if side in found
        )

    if not sides:
        raise MeasureError(
            EXIT_USER,
            f"cannot tell which side {name} is: neither its name nor its header names train,"
            " val, test, held-out or the domain's seed corpus",
            "measure a file written by the split stage",
        )
    if HELD_OUT in sides and not acceptance:
        raise MeasureError(
            EXIT_USER,
            f"{name} is the sealed held-out split (by its {origin(HELD_OUT)}); refusing to"
            " measure it without --acceptance",
            "pass --acceptance only for the one final held-out measurement",
        )
    if TEST in sides and not final:
        raise MeasureError(
            EXIT_USER,
            f"{name} is the test side (by its {origin(TEST)}); it is only measured on the"
            " final run (--final)",
            "iterate on val.json; pass --final for the final run",
        )
    if acceptance and HELD_OUT not in sides:
        raise MeasureError(
            EXIT_USER, f"--acceptance applies to the held-out split only, not {name}"
        )
    if final and TEST not in sides:
        raise MeasureError(EXIT_USER, f"--final applies to the test side only, not {name}")
    if HELD_OUT in sides:
        return HELD_OUT
    return TEST if TEST in sides else None


def read_split(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MeasureError(EXIT_USER, f"cannot read split file {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise MeasureError(EXIT_USER, f"{path} is not a split file ({{header, entries}})")
    return raw


def seed_from_header(header: object) -> int | None:
    """The seed the split stage wrote into the header (``... (seed=39).``), if any."""
    if isinstance(header, dict):
        seed = header.get("seed")
        return seed if isinstance(seed, int) else None
    if isinstance(header, str):
        match = _SEED_RE.search(header)
        return int(match.group(1)) if match else None
    return None


def home_relative(text: str) -> str:
    """*text* with the user's home directory written as ``$HOME`` (reports are shared)."""
    home = str(Path.home())
    return text.replace(home + "/", "$HOME/") if home not in ("", "/") else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# --calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """A validated calibration params file and where it came from."""

    path: str
    sha256: str
    temperature: float
    vector: Mapping[str, float]
    source: str
    fit_examples: object = None


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_calibration(path: Path) -> Calibration:
    """Read and check *path*; refuse (exit 1) a malformed one or one fitted on test/held-out."""
    hint = "write one with: python -m jev_factory.core.calibration fit ..."
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise MeasureError(EXIT_USER, f"--calibration {path}: {exc}", hint) from exc
    if not isinstance(raw, dict):
        raise MeasureError(EXIT_USER, f"--calibration {path}: not a JSON object", hint)
    temperature = raw.get("temperature", 1.0)
    if not _positive_number(temperature):
        raise MeasureError(
            EXIT_USER, f"--calibration {path}: temperature must be a positive number", hint
        )
    vector = raw.get("vector") or {}
    if not isinstance(vector, dict) or not all(
        isinstance(label, str) and _positive_number(scale) for label, scale in vector.items()
    ):
        raise MeasureError(
            EXIT_USER,
            f"--calibration {path}: vector must map each label to a positive number",
            hint,
        )
    source = raw.get("predictions_source")
    if not isinstance(source, str) or not source:
        raise MeasureError(
            EXIT_USER,
            f"--calibration {path}: no predictions_source, so where it was fitted is unknown",
            hint,
        )
    markers = split_markers(Path(source))
    if markers:
        raise MeasureError(
            EXIT_USER,
            f"--calibration {path} was fitted on {source}, which looks like the"
            f" {' and '.join(sorted(markers))} split; calibration is fitted on a validation"
            " fold only",
            "fit the params on a validation predictions file",
        )
    return Calibration(
        path=home_relative(str(path)),
        sha256=hashlib.sha256(data).hexdigest(),
        temperature=float(temperature),
        vector={label: float(scale) for label, scale in vector.items()},
        source=home_relative(source),
        fit_examples=raw.get("fit_examples"),
    )


def calibrate_lines(lines: Sequence[Prediction], calibration: Calibration) -> list[Prediction]:
    """*lines* with each ``candidates`` rescaled; the decision and the raw distribution stay."""
    return [
        (
            replace(
                line,
                candidates=apply_scaling(
                    line.candidates, calibration.temperature, dict(calibration.vector)
                ),
            )
            if line.candidates is not None
            else line
        )
        for line in lines
    ]


def calibration_note(calibration: Calibration | None) -> str:
    if calibration is None:
        return "none (the model's own candidate distributions)"
    vector = (
        f"vector over {len(calibration.vector)} label(s)" if calibration.vector else "no vector"
    )
    examples = (
        f", {calibration.fit_examples} fit example(s)"
        if calibration.fit_examples is not None
        else ""
    )
    return (
        f"`{calibration.path}` sha256 `{calibration.sha256}` (temperature"
        f" {calibration.temperature:.4f}, {vector}; fitted on `{calibration.source}`{examples});"
        " applied to every line's candidates before the predictions file and metrics"
    )


def readout_counts(notes: Mapping[str, int]) -> dict[str, int]:
    """A scorer run's readouts: complete, incomplete, no label mass, call error."""
    return {
        "complete": int(notes.get(READOUT_COMPLETE, 0)),
        "incomplete": int(notes.get(READOUT_INCOMPLETE, 0)),
        "no_label_mass": int(notes.get(READOUT_NO_MASS, 0)),
        "call_error": int(notes.get(READOUT_CALL_ERROR, 0)),
    }


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorerSpec:
    """What one model's scorer is built from."""

    kind: str
    model: str
    revision: str
    #: Where the tokenizer (and an in-process model) load from; ``None`` is *model*.
    tokenizer: str | None = None
    #: An attached server's base URL (``http://127.0.0.1:PORT/v1``).
    base_url: str | None = None
    #: A model directory or ``.gguf`` this run serves itself.
    serve_path: str | None = None
    port: int | None = None
    settings: serve.ServeSettings | None = None


@dataclass(frozen=True)
class ScorerHandle:
    """A top-k scoring function, the prompt renderer for its model, and its release."""

    top_k: sc.TopK
    render: Render
    close: Callable[[], None]
    #: The served endpoint to preflight; ``None`` for the in-process scorer.
    base_url: str | None = None
    #: What serves it (backend, argv, version), for the report.
    serving: Mapping[str, object] = field(default_factory=dict)


def tokenizer_renderer(spec: ScorerSpec) -> Render:  # loads a tokenizer
    """Render with the model's own chat template (thinking off), transformers loaded lazily."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer or spec.model, revision=spec.revision)

    def render(messages: list[dict]) -> str:
        return sc.render_prompt(tokenizer, messages)

    return render


def in_process_top_k(spec: ScorerSpec) -> sc.TopK:  # loads a model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = spec.tokenizer or spec.model
    tokenizer = AutoTokenizer.from_pretrained(source, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model, revision=spec.revision, torch_dtype=torch.bfloat16
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return sc.InProcessTopK(tokenizer, sc.transformers_logprobs(model, tokenizer))


def build_scorer(
    spec: ScorerSpec,
    *,
    renderer: Callable[[ScorerSpec], Render] = tokenizer_renderer,
    in_process: Callable[[ScorerSpec], sc.TopK] = in_process_top_k,
) -> ScorerHandle:
    """The scorer for *spec*: in-process, attached, or a server this run starts and stops."""
    render = renderer(spec)
    if spec.kind == SCORER_IN_PROCESS:
        return ScorerHandle(top_k=in_process(spec), render=render, close=lambda: None)
    if spec.serve_path is None:
        base_url = str(spec.base_url)
        return ScorerHandle(
            top_k=sc.served_top_k(base_url, spec.model),
            render=render,
            close=lambda: None,
            base_url=base_url,
            serving={"backend": "attached endpoint"},
        )
    settings = serve.with_name(spec.settings or serve.ServeSettings(), spec.model)
    port = int(spec.port or 0)
    record = Path(settings.run_dir) / f"{serve.NAME_PREFIX}{port}.record.json"
    serve.start(spec.serve_path, port, settings, record)
    try:
        serve.wait(port, settings)
    except BaseException:
        serve.stop(port, settings)
        raise
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        serving = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        serving = {}
    return ScorerHandle(
        top_k=sc.served_top_k(base_url, spec.model),
        render=render,
        close=lambda: serve.stop(port, settings),
        base_url=base_url,
        serving=serving,
    )


class _WatchedTopK:
    """Wraps a top-k function, counting (never swallowing) any exception.

    :func:`scorer.score` treats a failed call exactly like a model that put
    no mass on any letter -- right for an uncertain model, wrong for a server
    that died. This wrapper re-raises unchanged and only counts, so a real
    call error becomes a ``tier_error`` line instead of ``no_label_mass``.
    """

    def __init__(self, inner: sc.TopK) -> None:
        self._inner = inner
        self.errors = 0
        self.tops: list[int] = []

    def __call__(self, prompt: str, top: int) -> Mapping[str, float]:
        self.tops.append(top)
        try:
            return self._inner(prompt, top)
        except Exception:
            self.errors += 1
            raise


def scorer_request(
    domain: Domain, request_text: str, offered: Sequence[str] | None, *, reasons: bool
) -> tuple[list[dict], dict]:
    """``(prompt messages, scorer.score keyword arguments)`` for one request.

    *offered* is a missing-candidate slice entry's operations (``None``:
    every candidate). Without *reasons*: the default pool (bare
    ``escalate``), positional letters, domain descriptions. With *reasons*:
    the reasons pool (operations, explain, one ``escalate:<reason>`` per
    reason), lettered by position in it.
    """
    if not reasons:
        candidates = None if offered is None else tuple(offered) + CONTROLS
        return sc.prompt_messages(domain, request_text, candidates), {"offered": candidates}
    full = sc.candidate_pool(domain, True)
    keep = None if offered is None else set(offered)
    order = tuple(name for name in full if keep is None or domain.get(name) is None or name in keep)
    labels = sc.positional_labels(order, full)
    messages = sc.prompt_messages(domain, request_text, labels=labels, order=order, reasons=True)
    return messages, {"labels": labels, "order": order}


# ---------------------------------------------------------------------------
# One run per model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPlan:
    domain: Domain
    entries: Sequence[CorpusEntry]
    problems: Sequence[str]
    split: str
    world: Mapping[str, Any] | None
    grounding: str
    scorer_kind: str
    tokenizer: str | None = None
    reasons: bool = False
    base_url: str | None = None
    serve_path: str | None = None
    port: int | None = None
    settings: serve.ServeSettings | None = None
    ctx: int = 2048
    #: A :class:`~jev_factory.factory.detach.Progress` advanced once per scored entry.
    progress: Any = None


@dataclass
class RunRecord:
    model: str
    revision: str
    revision_status: str = ""
    nvidia_smi: str = ""
    startup_s: float | None = None
    failure: str = ""
    predictions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: Counter = field(default_factory=Counter)
    raw_calibration: dict | None = None
    serving: Mapping[str, object] = field(default_factory=dict)
    top_asked: tuple[int, ...] = ()


def scorer_predictions(
    plan: RunPlan, handle: ScorerHandle, clock: Callable[[], float]
) -> tuple[list[Prediction], Counter, tuple[int, ...]]:
    """Score every entry once; returns the lines, the readout notes and each top asked."""
    watched = _WatchedTopK(handle.top_k)
    lines: list[Prediction] = []
    notes: Counter = Counter()
    for entry in plan.entries:
        messages, offer = scorer_request(
            plan.domain, entry.text, entry.candidates, reasons=plan.reasons
        )
        prompt = handle.render(messages)
        started = clock()
        before = watched.errors
        scored = sc.score(
            plan.domain,
            watched,
            prompt,
            entry.text,
            reasons=plan.reasons,
            world=plan.world,
            entry_id=entry.id,
            **offer,
        )
        elapsed_ms = (clock() - started) * 1000.0
        call_error = watched.errors > before
        if call_error:
            notes[NO_DISTRIBUTION + "the scorer's call failed"] += 1
            notes[READOUT_CALL_ERROR] += 1
        elif scored.incomplete is not None:
            notes[NO_DISTRIBUTION + "labels missing from the top log-probabilities"] += 1
            notes[READOUT_INCOMPLETE] += 1
        elif scored.candidates is None:
            notes[NO_DISTRIBUTION + "no label mass"] += 1
            notes[READOUT_NO_MASS] += 1
        else:
            notes[READOUT_COMPLETE] += 1
        lines.append(
            to_prediction(
                scored,
                entry_id=entry.id,
                expected=entry.expect,
                elapsed_ms=elapsed_ms,
                call_error=call_error,
            )
        )
        if plan.progress is not None:
            plan.progress.advance()
    return lines, notes, tuple(watched.tops)


def _capture(run: RunFn, argv: list[str]) -> str:
    from jev_factory.release.scan import redact

    code, output = run(argv, _RUN_TIMEOUT)
    text = redact(output.encode("utf-8", "replace")).decode("utf-8", "replace").rstrip()
    return text if code == 0 else f"(exit {code}) {text}".rstrip()


def revision_status(plan: RunPlan) -> str:
    if plan.scorer_kind == SCORER_IN_PROCESS:
        return REVISION_IN_PROCESS
    if plan.serve_path is None:
        return f"{REVISION_UNVERIFIED}: attached endpoint"
    path = Path(plan.serve_path)
    if path.is_file():
        return f"{REVISION_UNVERIFIED}; served file sha256 `{sha256_file(path)}`"
    return f"{REVISION_UNVERIFIED}; served directory"


_STARTUP_ERRORS = (
    serve.ServeError,
    sc.ServedError,
    OSError,
    ValueError,
    ImportError,
    RuntimeError,
)


def score_one(plan: RunPlan, model: str, revision: str, seams: Seams) -> RunRecord:
    """The candidate scorer over every entry of *plan*, for one model."""
    record = RunRecord(model=model, revision=revision)
    record.revision_status = revision_status(plan)
    record.nvidia_smi = _capture(seams.run, ["nvidia-smi"])
    spec = ScorerSpec(
        kind=plan.scorer_kind,
        model=model,
        revision=revision,
        tokenizer=plan.tokenizer,
        base_url=plan.base_url,
        serve_path=plan.serve_path,
        port=plan.port,
        settings=plan.settings,
    )
    started = seams.clock()
    try:
        handle = seams.build_scorer(spec)
    except _STARTUP_ERRORS as exc:
        record.failure = f"scorer start-up failed: {exc}"
        return record
    record.startup_s = seams.clock() - started
    record.serving = dict(handle.serving)
    try:
        if handle.base_url is not None:
            try:
                seams.preflight(handle.base_url, model, ctx=plan.ctx)
            except PreflightError as exc:
                raise MeasureError(EXIT_ENV, str(exc)) from exc
        record.predictions, record.notes, record.top_asked = scorer_predictions(
            plan, handle, seams.clock
        )
    finally:
        handle.close()
    return record


def score_predictions(lines: Sequence[Prediction], path: Path, domain: Domain) -> dict:
    """Write *lines* to *path* as JSONL, then score that file with core.metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(path, lines)
    try:
        return metrics.compute(metrics.read_predictions(path), domain)
    except PredictionError as exc:
        raise MeasureError(EXIT_ENV, f"metrics refused the predictions file: {exc}") from exc


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _of(count: int, total: int) -> str:
    return f"{count} of {total}"


def _pct(value: object) -> str:
    return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _num(value: object, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _ms(value: object) -> str:
    return f"{value:.0f} ms" if isinstance(value, (int, float)) else "n/a"


def _timing(block: Mapping[str, object]) -> str:
    return (
        f"{_ms(block.get('cold_ms'))} / {_ms(block.get('warm_median_ms'))} / "
        f"{_ms(block.get('warm_p95_ms'))}"
    )


def _ci(block: object, *, percent: bool) -> str:
    """core.metrics' ``{n, value, ci_low, ci_high}`` as ``[low, high] (n=N)``."""
    if not isinstance(block, Mapping):
        return "n/a"
    low, high = block.get("ci_low"), block.get("ci_high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return f"n/a (n={block.get('n', 0)})"
    shown = (_pct(low), _pct(high)) if percent else (_num(low), _num(high))
    return f"[{shown[0]}, {shown[1]}] (n={block.get('n', 0)})"


def _path(data: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)  # type: ignore[assignment]
    return data


CI_SUFFIX = ", 95% bootstrap CI"


def _readout_cell(record: RunRecord) -> str:
    counts = readout_counts(record.notes)
    if not any(counts.values()):
        return "n/a"
    cell = f"{counts['complete']} / {counts['incomplete']}"
    other = [
        f"{why.replace('_', ' ')}: {counts[why]}"
        for why in ("no_label_mass", "call_error")
        if counts[why]
    ]
    return cell + (f" ({', '.join(other)})" if other else "")


def _before_calibration(record: RunRecord) -> str:
    raw = record.raw_calibration
    if raw is None:
        return "n/a"
    return f"{_num(raw.get('ece'))} / {_num(raw.get('brier'))}"


def metric_rows(records: Sequence[RunRecord]) -> list[tuple[str, list[str]]]:
    """``(metric, [one cell per model])`` for core.metrics' figures over each predictions file."""

    def cells(fn: Callable[[dict, RunRecord], str]) -> list[str]:
        return [
            "not measured" if record.failure or not record.metrics else fn(record.metrics, record)
            for record in records
        ]

    def invalid(m: dict, _r: RunRecord) -> str:
        reasons = ", ".join(f"{k}: {v}" for k, v in m["invalid"]["by_reason"].items())
        return _of(m["invalid"]["n"], m["invalid"]["N"]) + (f" ({reasons})" if reasons else "")

    return [
        ("Model revision", [f"`{r.revision}` ({r.revision_status})" for r in records]),
        (
            "Right proposals",
            cells(lambda m, _r: _of(m["right_proposals"]["n"], m["right_proposals"]["N"])),
        ),
        (
            "Right proposals" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "right_proposals", "ci"), percent=True)),
        ),
        (
            "Abstention recall (escalate entries escalated)",
            cells(
                lambda m, _r: f"{_pct(m['abstention']['recall'])} "
                f"({_of(m['escalation']['tp'], m['escalation']['tp'] + m['escalation']['fn'])})"
            ),
        ),
        (
            "Abstention recall" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "escalation", "recall_ci"), percent=True)),
        ),
        ("Abstention precision, strict", cells(lambda m, _r: _pct(m["abstention"]["precision"]))),
        (
            "Abstention precision, strict" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "escalation", "precision_strict_ci"), percent=True)),
        ),
        (
            "False-positive tool calls (proposals on explain/escalate entries)",
            cells(
                lambda m, _r: _of(
                    m["false_positive_tool_calls"]["n"], m["false_positive_tool_calls"]["N"]
                )
            ),
        ),
        (
            "False-positive tool calls" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "false_positive_tool_calls", "ci"), percent=True)),
        ),
        (
            "Wrong mutating, total (wrong operation + wrong arguments)",
            cells(
                lambda m, _r: f"{m['wrong_mutating']['total']} "
                f"({m['wrong_mutating']['wrong_operation']} + "
                f"{m['wrong_mutating']['wrong_arguments']})"
            ),
        ),
        ("Invalid outputs", cells(invalid)),
        (
            "Invalid outputs" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "invalid", "ci"), percent=True)),
        ),
        (
            "Lines with a candidate distribution",
            cells(
                lambda m, _r: _of(
                    m["calibration"]["n"],
                    m["calibration"]["n"] + m["calibration"]["without_distribution"],
                )
            ),
        ),
        (
            "Scorer readouts, complete / incomplete (never renormalised)",
            cells(lambda _m, r: _readout_cell(r)),
        ),
        ("ECE (10 equal-width bins)", cells(lambda m, _r: _num(m["calibration"]["ece"]))),
        (
            "ECE" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "calibration", "ece_ci"), percent=False)),
        ),
        ("Brier (multi-class)", cells(lambda m, _r: _num(m["calibration"]["brier"]))),
        (
            "Brier" + CI_SUFFIX,
            cells(lambda m, _r: _ci(_path(m, "calibration", "brier_ci"), percent=False)),
        ),
        ("ECE / Brier before --calibration", cells(lambda _m, r: _before_calibration(r))),
        (
            "Decision latency, cold / warm median / warm p95",
            cells(lambda m, _r: _timing(m["latency"])),
        ),
        (
            "Start-up",
            [r.failure or _num(r.startup_s, 1) + " s" for r in records],
        ),
    ]


def render_metrics(records: Sequence[RunRecord]) -> str:
    header = "| Metric | " + " | ".join(f"`{record.model}`" for record in records) + " |"
    lines = [header, "|---|" + "---|" * len(records)]
    for metric, row in metric_rows(records):
        lines.append(f"| {metric} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _with_ci(shown: str, block: object, *, percent: bool = True) -> str:
    if isinstance(block, Mapping) and not block.get("n"):
        return "n/a (n=0)"
    return f"{shown} {_ci(block, percent=percent)}"


def _slice_rows(slices: Mapping[str, object]) -> list[str]:
    lines = [
        "| Slice | Lines | With a distribution | ECE [95% CI] | Brier [95% CI] |"
        " Missing-candidate rate [95% CI] | Offered candidates, mean / median |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in metrics.SLICE_NAMES:
        data = slices.get(name)
        if not isinstance(data, Mapping):
            continue
        cal = data.get("calibration") or {}
        missing = data.get("missing_candidate") or {}
        count = data.get("candidate_count") or {}
        lines.append(
            f"| {name} | {data.get('n', 0)} | {cal.get('n', 0)} |"
            f" {_with_ci(_num(cal.get('ece')), cal.get('ece_ci'), percent=False)} |"
            f" {_with_ci(_num(cal.get('brier')), cal.get('brier_ci'), percent=False)} |"
            f" {_with_ci(_of(missing.get('n', 0), missing.get('N', 0)), missing.get('rate'))} |"
            f" {_num(count.get('mean'), 1)} / {_num(count.get('median'), 1)} |"
        )
    return lines


def _slices_section(records: Sequence[RunRecord]) -> list[str]:
    lines = [
        "## Per-slice calibration",
        "",
        "Slices by the gold label: read-only and mutating operations (the domain's",
        "`read_only` flag) and escalate-or-explain entries. Confidence intervals are",
        "seeded percentile bootstraps over the slice's lines.",
        "",
    ]
    for record in records:
        lines += [f"### `{record.model}`", ""]
        slices = record.metrics.get("slices") if not record.failure else None
        if not isinstance(slices, Mapping):
            lines += ["Not measured.", ""]
            continue
        reliability = metrics.reliability_markdown(slices)
        lines += [*_slice_rows(slices), "", re.sub(r"(?m)^### ", "#### ", reliability), ""]
    return lines


def _notes_lines(records: Sequence[RunRecord]) -> list[str]:
    lines = []
    for record in records:
        missing = sorted(
            (key[len(NO_DISTRIBUTION) :], count)
            for key, count in record.notes.items()
            if key.startswith(NO_DISTRIBUTION)
        )
        if missing:
            detail = "; ".join(f"{why}: {count}" for why, count in missing)
            lines.append(f"- `{record.model}`: lines without a candidate distribution: {detail}")
    return lines


@dataclass(frozen=True)
class Provenance:
    date: str
    label: str
    command: str
    domain: str
    surface_sha256: str
    split_path: str
    split_sha256: str
    split_count: int
    source_count: int
    problems: Sequence[str]
    seed: int | None
    seed_origin: str
    grounding: str
    final: bool
    acceptance: bool
    sealed_before: int
    deviation: str | None
    decision_mode: str
    slice_note: str
    snapshot_note: str = ""
    tier_errors: int = 0
    tier_errors_allowed: int = 0
    calibration_note: str = ""
    ctx: int = 2048
    slice: str = SLICE_FULL


def _serving_line(record: RunRecord) -> str:
    serving = record.serving or {}
    parts = [f"backend={serving.get('backend', 'in-process')}"]
    for key in ("version", "image", "served_model_name"):
        if serving.get(key):
            parts.append(f"{key}=`{str(serving[key]).splitlines()[0]}`")
    return f"- `{record.model}` serving: " + ", ".join(parts)


def render_markdown(prov: Provenance, records: Sequence[RunRecord]) -> str:
    seed = f"{prov.seed} ({prov.seed_origin})" if prov.seed is not None else "not recorded"
    nothing = bool(records) and all(record.failure for record in records)
    lines = [f"# Scorer measurement, {prov.date}: {prov.label}", ""]
    if nothing:
        lines += [
            "**Nothing was measured: every model's start-up failed.** A re-run with the",
            "same label replaces this page.",
            "",
        ]
    if prov.tier_errors:
        lines += [
            f"**{prov.tier_errors} tier-error prediction(s) permitted by "
            f"`--allow-tier-errors {prov.tier_errors_allowed}`** -- the server was "
            "unreachable for at least one entry; this run is not a clean measurement.",
            "",
        ]
    lines += [
        f"- Command: `{prov.command}`",
        f"- Domain: `{prov.domain}`, candidate surface sha256 `{prov.surface_sha256}`",
        f"- Split: `{prov.split_path}` sha256 `{prov.split_sha256}` ({prov.split_count}"
        f" entries, {prov.source_count} sources)",
        f"- Seed: {seed}",
        f"- jev-factory: {__version__}",
        "- Models (id @ revision): "
        + "; ".join(
            f"`{record.model}` @ `{record.revision}` ({record.revision_status})"
            for record in records
        ),
        f"- Context: {prov.ctx}",
        *(_serving_line(record) for record in records),
        f"- Grounding: {prov.grounding}",
        f"- Decision mode: {prov.decision_mode}",
        f"- Slice: {prov.slice_note}",
    ]
    if prov.snapshot_note:
        lines.append(f"- Ground snapshot: {prov.snapshot_note}")
    lines.append(f"- Calibration: {prov.calibration_note or calibration_note(None)}")
    if nothing:
        lines.append(NOT_MEASURED_MARKER)
    lines += [
        f"- Acceptance run: {'yes' if prov.acceptance else 'no'}",
        FINAL_MARKER if prov.final else "- Final run: no",
    ]
    if prov.final or prov.acceptance:
        side = TEST if prov.final else HELD_OUT
        slice_note = "" if prov.slice == SLICE_FULL else f" ({prov.slice} slice)"
        lines.append(
            f"- Measurements of the {side} side{slice_note} in this run, including this one:"
            f" {prov.sealed_before + (0 if nothing else 1)}"
        )
    if prov.deviation:
        lines.append(f"- Deviation: `{prov.deviation}`")
    if prov.problems:
        lines.append(f"- Corpus problems (entries skipped): {len(prov.problems)}")
    notes = _notes_lines(records)
    lines += [
        "",
        "## Metrics",
        "",
        "core.metrics over each model's predictions file (one line per entry).",
        "",
        render_metrics(records),
        "",
        *([*notes, ""] if notes else []),
        metrics.issue46_mapping()["markdown"],
        "",
        metrics.ISSUE46_NOTE,
        "",
        *_slices_section(records),
        "## Background before each run",
        "",
    ]
    for record in records:
        lines += [f"### `{record.model}`", "", "`nvidia-smi`:", "", "```text"]
        lines += [record.nvidia_smi or "(not captured)", "```", ""]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Seams: everything that touches the machine, injectable for tests
# ---------------------------------------------------------------------------


def _today() -> str:  # wall clock
    return datetime.date.today().isoformat()


def _gpu_guard(stage: str, *, allow_foreign: bool = False) -> object:
    from jev_factory.factory.gpu import assert_gpu_free

    return assert_gpu_free(stage, allow_foreign=allow_foreign)


@dataclass
class Seams:
    run: RunFn = serve.default_run
    today: Callable[[], str] = _today
    clock: Callable[[], float] = time.monotonic
    build_scorer: Callable[[ScorerSpec], ScorerHandle] = build_scorer
    #: ``(base_url, model, ctx=...)``; raises PreflightError when the server is not right.
    preflight: Callable[..., None] = preflight_models
    #: ``(stage, allow_foreign=...)``; raises GpuBusyError next to a foreign GPU process.
    gpu_guard: Callable[..., object] = _gpu_guard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.measure.run", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--domain", required=True, help="domain module or JSON domain file")
    parser.add_argument("--run-dir", required=True, help="the run's work directory")
    parser.add_argument("--split", required=True, help="split file from the split stage")
    parser.add_argument("--model", action="append", required=True, help="repeat to compare")
    parser.add_argument("--revision", action="append", default=[], help="one per --model")
    parser.add_argument("--label", default="measure", help="results file name part")
    parser.add_argument(
        "--progress-dir",
        default=None,
        help="write <dir>/measure-<label>.progress.json after every entry (jev status reads it)",
    )
    parser.add_argument(
        "--scorer", required=True, choices=(SCORER_SERVED, SCORER_IN_PROCESS), help="scorer"
    )
    parser.add_argument("--base-url", default=None, help="served: attach to this localhost URL")
    parser.add_argument(
        "--max-logprobs",
        type=int,
        default=None,
        help="served + --base-url: the --max-logprobs the attached engine was started with",
    )
    parser.add_argument("--serve", default=None, help="served: model dir or .gguf to serve")
    parser.add_argument("--port", type=int, default=18060, help="--serve: localhost port")
    parser.add_argument("--llama-server", default=None, help="--serve .gguf: llama-server binary")
    parser.add_argument("--image", default=None, help="--serve dir: vLLM image by @sha256")
    parser.add_argument("--ctx", type=int, default=None, help="served context (default 2048)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer (and in-process model) path")
    parser.add_argument("--seed", type=int, default=None, help="default: the split header's")
    parser.add_argument("--world", default=None, help="corpus file whose world to ground on")
    parser.add_argument("--live", action="store_true", help="ground with the live lookups")
    parser.add_argument("--ground-snapshot", default=None, help="fixed grounding snapshot")
    parser.add_argument("--acceptance", action="store_true", help="the sealed held-out split")
    parser.add_argument("--final", action="store_true", help="the test side (the final run)")
    parser.add_argument(
        "--deviation",
        default=None,
        help="recorded deviation id permitting a second final/held-out measurement",
    )
    parser.add_argument("--out", default=None, help="results page (default: RUN/measure/)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing results page")
    parser.add_argument(
        "--predictions", default=None, help="predictions + metrics dir (default: RUN/measure/)"
    )
    parser.add_argument("--allow-tier-errors", type=int, default=0, metavar="N")
    parser.add_argument("--allow-foreign-gpu", action="store_true")
    parser.add_argument(
        "--slice", choices=(SLICE_FULL, SLICE_MISSING), default=SLICE_FULL, help="what to score"
    )
    parser.add_argument("--calibration", default=None, metavar="PARAMS")
    parser.add_argument("--reasons", action="store_true", help="offer the escalate:<r> pool")
    return parser


def _check_flags(args: argparse.Namespace) -> None:
    """Refuse flag combinations before anything is loaded or started."""
    if not _LABEL_RE.match(args.label):
        raise MeasureError(
            EXIT_USER,
            f"--label {args.label!r} must be lower-case letters, digits, '.', '_' or '-'",
        )
    if len(args.revision) != len(args.model):
        raise MeasureError(
            EXIT_USER,
            f"{len(args.model)} --model but {len(args.revision)} --revision",
            "give one --revision (the pinned commit or build hash) after each --model",
        )
    if args.allow_tier_errors < 0:
        raise MeasureError(EXIT_USER, "--allow-tier-errors must be 0 or more")
    if args.live and args.ground_snapshot:
        raise MeasureError(EXIT_USER, "--live and --ground-snapshot are two different groundings")
    if args.ctx is not None and args.ctx <= 0:
        raise MeasureError(EXIT_USER, "--ctx must be a positive whole number")
    try:
        once.check_deviation_id(args.deviation)
    except ValueError as exc:
        raise MeasureError(EXIT_USER, str(exc)) from exc
    if args.scorer == SCORER_IN_PROCESS:
        if args.base_url or args.serve:
            raise MeasureError(EXIT_USER, "--base-url/--serve apply to --scorer served only")
        return
    if bool(args.base_url) == bool(args.serve):
        raise MeasureError(
            EXIT_USER,
            "a served scorer needs exactly one of --base-url (attach) or --serve (start it)",
        )
    if args.serve:
        if len(args.model) != 1:
            raise MeasureError(EXIT_USER, "--serve serves one --model per run")
        if args.tokenizer is None:
            raise MeasureError(
                EXIT_USER,
                "--serve needs --tokenizer: the served name is not a loadable tokenizer",
                "pass the source model directory the build was made from",
            )
        return
    needed = ro.READOUT_TOP
    if args.max_logprobs is None or args.max_logprobs < needed:
        raise MeasureError(
            EXIT_USER,
            f"a served scorer asks for {needed} log-probabilities (READOUT_TOP) per request;"
            " --max-logprobs must say the attached engine allows that",
            f"start the server with --max-logprobs {needed} or more and pass the same value,"
            " or let this stage start it (--serve)",
        )


def _world(
    args: argparse.Namespace, raw: dict, domain: Domain
) -> tuple[Mapping[str, Any] | None, str, str]:
    """``(world or None for live lookups, grounding note, snapshot note)``."""
    if args.live:
        return None, "live (the domain's lookups on this machine)", ""
    if args.ground_snapshot:
        path = Path(args.ground_snapshot)
        try:
            snapshot, digest = load_snapshot(path, domain)
        except SnapshotError as exc:
            raise MeasureError(EXIT_USER, str(exc)) from exc
        shown = home_relative(str(path))
        sizes = ", ".join(
            f"{len(snapshot[kind.world_field])} {kind.world_field}" for kind in domain.ground_kinds
        )
        note = f"`{shown}` sha256 `{digest}` ({sizes}; created {snapshot['created']})"
        return snapshot, f"fixed snapshot `{shown}`", note
    if args.world:
        try:
            other = json.loads(Path(args.world).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MeasureError(EXIT_USER, f"cannot read --world {args.world}: {exc}") from exc
        world = other.get("world") if isinstance(other, dict) else None
        if isinstance(world, dict):
            return world, f"fixture world from `{home_relative(args.world)}`", ""
        raise MeasureError(EXIT_USER, f"--world {args.world} holds no world object")
    if isinstance(raw.get("world"), dict):
        return raw["world"], "fixture world from the split file", ""
    if domain.seed_corpus is not None:
        try:
            return dict(domain.load_seed_corpus().world), "the seed corpus's world", ""
        except (OSError, ValueError) as exc:
            raise MeasureError(EXIT_USER, f"cannot read the seed corpus world: {exc}") from exc
    raise MeasureError(
        EXIT_USER,
        "no world to ground against",
        "pass --ground-snapshot, --world or --live",
    )


def measured_nothing(path: Path) -> bool:
    """True when *path* is a results page whose every start-up failed (nvsh P71)."""
    try:
        return NOT_MEASURED_MARKER in path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False


def _mode_note(args: argparse.Namespace) -> str:
    reasons = (
        "; reasons mode: operations, explain and one escalate:<reason> per reason, no bare"
        " escalate"
        if args.reasons
        else ""
    )
    if args.scorer == SCORER_IN_PROCESS:
        how = "in-process (every letter of the alphabet read)"
    elif args.serve:
        how = f"served by this run ({'llama-server' if args.serve.endswith('.gguf') else 'vLLM'})"
    else:
        how = f"attached endpoint, max-logprobs {args.max_logprobs} (operator-supplied)"
    return f"candidate scorer {how}; asks for {ro.READOUT_TOP} log-probabilities{reasons}"


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model.lower()).strip("-") or "model"


def run(argv: Sequence[str], seams: Seams) -> int:
    from jev_factory.domain.validate import DomainError, load_domain

    args = _parser().parse_args(list(argv))
    _check_flags(args)
    try:
        domain = load_domain(args.domain)
    except DomainError as exc:
        raise MeasureError(EXIT_USER, f"--domain: {exc}") from exc
    run_dir = Path(args.run_dir)
    try:
        from jev_factory.factory.workroot import resolve_work_root

        run_dir = resolve_work_root(run_dir)
    except Exception as exc:  # noqa: BLE001 -- CliError from the work-root guard
        raise MeasureError(getattr(exc, "code", EXIT_USER), str(exc)) from exc
    split_path = Path(args.split)
    raw = read_split(split_path)
    side = check_split_allowed(
        split_path,
        acceptance=args.acceptance,
        final=args.final,
        header=raw.get("header"),
        dev_corpus=domain.seed_corpus,
    )
    if args.deviation and side is None:
        raise MeasureError(EXIT_USER, "--deviation applies to a final or held-out run only")
    if side is None:
        return _run(args, argv, raw, split_path, domain, run_dir, seams, None)
    ledger = once.OnceLedger(run_dir)
    try:
        with ledger:
            ledger.check(side, args.deviation, args.slice)
            return _run(args, argv, raw, split_path, domain, run_dir, seams, (ledger, side))
    except once.OnceError as exc:
        raise MeasureError(
            EXIT_USER,
            str(exc),
            "a second measurement of a sealed side is a recorded deviation: pass --deviation"
            " <id> once the deviation is recorded",
        ) from exc


def _run(
    args: argparse.Namespace,
    argv: Sequence[str],
    raw: dict,
    split_path: Path,
    domain: Domain,
    run_dir: Path,
    seams: Seams,
    sealed: tuple[once.OnceLedger, str] | None,
) -> int:
    date = seams.today()
    measure_dir = run_dir / "measure"
    out = Path(args.out) if args.out else measure_dir / f"{date}-{args.label}.md"
    if out.exists() and not args.force:
        if not measured_nothing(out):
            raise MeasureError(
                EXIT_USER, f"{out} already exists", "pick another --label or --force"
            )
        print(
            f"note: replacing {out}: its run measured nothing (every start-up failed)",
            file=sys.stderr,
        )
    calibration = load_calibration(Path(args.calibration)) if args.calibration else None
    world, grounding, snapshot_note = _world(args, raw, domain)

    corpus_raw = raw
    if args.slice == SLICE_MISSING:
        header = raw.get("header")
        header_text = header if isinstance(header, str) else json.dumps(header, sort_keys=True)
        corpus_raw = missing_candidate_slice(
            {"header": header_text, "entries": raw["entries"]}, domain.names()
        )
    loaded = load_raw(corpus_raw, domain)
    if not loaded.entries:
        raise MeasureError(EXIT_USER, f"{split_path.name} has no valid entries")
    seed, seed_origin = args.seed, "--seed"
    if seed is None:
        seed, seed_origin = seed_from_header(raw.get("header")), "from the split header"

    settings = None
    if args.serve:
        settings = serve.ServeSettings.from_env()
        overrides = {
            "run_dir": measure_dir / "serve",
            "ctx": args.ctx,
            "llama_server": args.llama_server,
            "image": args.image,
        }
        settings = replace(settings, **{k: v for k, v in overrides.items() if v is not None})
    ctx = args.ctx if args.ctx is not None else (settings.ctx if settings else 2048)
    plan = RunPlan(
        domain=domain,
        entries=loaded.entries,
        problems=loaded.problems,
        split=split_path.stem,
        world=world,
        grounding=grounding,
        scorer_kind=args.scorer,
        tokenizer=args.tokenizer,
        reasons=args.reasons,
        base_url=args.base_url,
        serve_path=args.serve,
        port=args.port,
        settings=settings,
        ctx=ctx,
    )

    cpu_served = bool(
        args.serve
        and settings is not None
        and settings.gpu_layers == 0
        and str(args.serve).endswith(".gguf")
    )
    if args.scorer == SCORER_IN_PROCESS or (args.serve and not cpu_served):
        try:
            seams.gpu_guard("measure", allow_foreign=args.allow_foreign_gpu)
        except Exception as exc:  # noqa: BLE001 -- GpuBusyError / GpuQueryError
            raise MeasureError(
                EXIT_ENV, str(exc), "stop the other GPU job, or pass --allow-foreign-gpu"
            ) from exc
    sealed_before = 0
    if sealed is not None:
        sealed_before = len(sealed[0].measured(sealed[1], args.slice))

    if args.progress_dir:
        from jev_factory.factory.detach import Progress

        plan = replace(
            plan,
            progress=Progress(
                Path(args.progress_dir),
                "measure-" + args.label.replace(".", "_"),
                len(loaded.entries) * len(args.model),
            ),
        )
    records: list[RunRecord] = []
    status = once.FAILED_MID_RUN
    try:
        for model, revision in zip(args.model, args.revision):
            records.append(score_one(plan, model, revision, seams))
        keep = Path(args.predictions) if args.predictions else measure_dir / args.label
        for index, record in enumerate(records, start=1):
            if record.failure:
                continue
            name = f"{args.label}-{index}-{_slug(record.model)}"
            if calibration is not None:
                record.raw_calibration = metrics.compute_calibration(record.predictions)
                record.predictions = calibrate_lines(record.predictions, calibration)
            record.metrics = score_predictions(
                record.predictions, keep / f"{name}.predictions.jsonl", domain
            )
            kept = dict(record.metrics)
            kept["readout"] = readout_counts(record.notes)
            kept["top_asked"] = sorted(set(record.top_asked))
            if calibration is not None:
                kept["calibration_applied"] = {
                    "params": calibration.path,
                    "sha256": calibration.sha256,
                    "before": record.raw_calibration,
                }
            (keep / f"{name}.metrics.json").write_text(
                json.dumps(kept, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        tier_error_total = sum(
            record.metrics.get("invalid", {}).get("by_reason", {}).get(TIER_ERROR_REASON, 0)
            for record in records
            if not record.failure
        )
        if tier_error_total > args.allow_tier_errors:
            print(
                f"error: {tier_error_total} tier_error prediction(s) (server unreachable"
                f" mid-run), above --allow-tier-errors {args.allow_tier_errors}; not writing"
                f" {out}",
                file=sys.stderr,
            )
            print(
                "hint: fix or restart the server and re-run, or pass --allow-tier-errors N",
                file=sys.stderr,
            )
            return EXIT_ENV

        prov = Provenance(
            date=date,
            label=args.label,
            command=home_relative(shlex.join(["python -m jev_factory.measure.run", *argv])),
            domain=domain.name,
            surface_sha256=domain.surface_sha256(),
            split_path=home_relative(str(split_path)),
            split_sha256=sha256_file(split_path),
            split_count=len(loaded.entries),
            source_count=len({entry.source_id for entry in loaded.entries}),
            problems=loaded.problems,
            seed=seed,
            seed_origin=seed_origin,
            grounding=grounding,
            final=args.final,
            acceptance=args.acceptance,
            sealed_before=sealed_before,
            slice=args.slice,
            deviation=args.deviation,
            decision_mode=_mode_note(args),
            slice_note=(
                "full split"
                if args.slice == SLICE_FULL
                else f"{SLICE_MISSING} (each operation entry with its gold operation left out"
                " of the offered candidates, expected to escalate)"
            ),
            snapshot_note=snapshot_note,
            tier_errors=tier_error_total,
            tier_errors_allowed=args.allow_tier_errors,
            calibration_note=calibration_note(calibration),
            ctx=ctx,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(prov, records), encoding="utf-8")
        print(render_metrics(records))
        print(f"wrote {out}")
        failed = [record for record in records if record.failure]
        for record in failed:
            print(f"error: {record.model}: {record.failure}", file=sys.stderr)
        if not failed:
            status = once.MEASURED
        return EXIT_ENV if failed else EXIT_OK
    finally:
        # A sealed side is touched once: any run that scored entries is recorded, cleanly
        # or not; a run whose every start-up failed (or was refused) never is (P71).
        if sealed is not None and any(not record.failure for record in records):
            ledger, side = sealed
            ledger.append(
                once.OnceRecord(
                    side=side,
                    split_sha256=sha256_file(split_path),
                    label=args.label,
                    date=date,
                    status=status,
                    deviation=args.deviation,
                    slice=args.slice,
                )
            )


def main(argv: Sequence[str] | None = None, *, seams: Seams | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv), seams or Seams())
    except MeasureError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
