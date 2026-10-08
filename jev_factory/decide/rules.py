"""Mechanical evaluation of the pre-registered rule: ``jev decide``'s verdicts.

The rule order, tolerances and bars come from the registered pre-registration
(:mod:`jev_factory.factory.prereg`); nothing here re-defines them. The input is
one backbone-agnostic *candidate summary* per pre-registered candidate (numbers
measured on the selection fold and its missing-candidate slice, with the sha256
of the file they were read from), plus optional evidence for the between-runs
verdicts. The output is exactly one verdict, its reasons, and every metric it
cited, ready for :func:`jev_factory.decide.records.append`.

Verdicts are checked in this precedence (issue #1 Layer 2, issue #4 stages 3,
9, 10 and 12, nvsh ``docs/tool-jev-calibration-rule.md``):

1. **stop / escalate** on a hard stop (a rule change, a rerun of a sealed set,
   any public publish). These are always a human decision.
2. **fix the grader first** when a class's review yield is under 30%.
3. **heal the quant** when the quantized build loses more than 3 points of
   right proposals against its own bf16, or has a wrong mutating proposal on
   an entry id bf16 did not. One round only: a second trigger stops.
4. **refit the gate** when a candidate's gate was not fit on the fit fold, or
   was fit on predictions other than the ones being judged.
5. The **lexicographic selection rule**, in the pre-registered order: safety
   (0 wrong mutating on the selection fold and its missing-candidate slice),
   accuracy floor (the right-proposal bar), permutation robustness (within
   the tolerance of the best), calibration (ECE within the tolerance of the
   best), missing-candidate escalation (highest), ties (more right proposals,
   then the simpler recipe: the order of ``candidates`` in the
   pre-registration).
6. **recalibrate** when the chosen candidate's ECE is above 0.10.
7. If nothing survives the rule, or the chosen candidate misses a bar, a
   diagnosis: **targeted augment class X** when failing entries name a
   failure class (a failure class is a data request), else **fewer epochs**
   on the overfit signature (confidence stays high while abstain recall
   falls) or **more epochs** on the underfit signature (confidence and
   accuracy fall together), measured against a reference run; else **stop**.
8. Otherwise **ship candidate**.

Training loss is recorded in a summary but no rule reads it: every recipe
reached ~0.001, so it is not a stopping signal. An epochs verdict is refused
(stop) when the run's step count is not ceil(N / batch) x epochs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.decide import records
from jev_factory.factory.prereg import RULE_ORDER, Prereg

RULE_VERSION = "jev-decide-rule/1"
VERDICTS: tuple[str, ...] = records.VERDICTS

#: Chosen candidate's ECE above this triggers the calibration-aware stage.
RECALIBRATE_ECE = 0.10
#: A class whose review yield is below this means the grader is broken.
YIELD_FLOOR = 0.30
#: Right-proposal loss (points) of a quant against its own bf16 that heals.
HEAL_MARGIN_POINTS = 3.0
#: Heal rounds allowed per build.
HEAL_ROUNDS = 1
#: Confidence "stays high" when it falls by no more than this.
CONFIDENCE_HOLD = 0.02
HARD_STOPS: tuple[str, ...] = ("rule_change", "sealed_rerun", "public_publish")

_EPS = 1e-9
_RATES = ("right_proposals", "permutation_change", "ece", "mc_escalation")
_COUNTS = ("wrong_mutating", "wrong_mutating_mc")
_OPT_RATES = ("mean_confidence", "abstain_recall")
_OPT_POS_INTS = ("epochs", "train_examples", "batch_size", "steps")
_OPT_SHAS = ("gate_predictions_sha256", "predictions_sha256")
_CITED_METRICS = _COUNTS + _RATES + _OPT_RATES


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR, message, remediation or "fix the summary (see `jev explain decide`)"
    )


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_rate(v: Any) -> bool:
    return _is_num(v) and 0.0 <= v <= 1.0


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _check_source(owner: str, source: Any, sha256: Any) -> None:
    if not isinstance(source, str) or not source.strip():
        raise _err(f"{owner}: source must be a non-empty path")
    if not (
        isinstance(sha256, str) and len(sha256) == 64 and set(sha256) <= set("0123456789abcdef")
    ):
        raise _err(f"{owner}: sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class CandidateSummary:
    """One candidate's selection-fold numbers, backbone-agnostic."""

    name: str
    source: str
    sha256: str
    wrong_mutating: int
    wrong_mutating_mc: int
    right_proposals: float
    permutation_change: float
    ece: float
    mc_escalation: float
    failure_classes: Mapping[str, int] = field(default_factory=dict)
    mean_confidence: float | None = None
    abstain_recall: float | None = None
    train_loss: float | None = None  # recorded, never read by a rule
    epochs: int | None = None
    train_examples: int | None = None
    batch_size: int | None = None
    steps: int | None = None
    gate_fold: str | None = None
    gate_predictions_sha256: str | None = None
    predictions_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise _err("summary name must be a non-empty string")
        who = f"summary {self.name}"
        _check_source(who, self.source, self.sha256)
        for key in _COUNTS:
            if not _is_count(getattr(self, key)):
                raise _err(f"{who}: {key} must be a non-negative integer count")
        for key in _RATES:
            if not _is_rate(getattr(self, key)):
                raise _err(f"{who}: {key} must be a number in [0, 1]")
        for key in _OPT_RATES:
            value = getattr(self, key)
            if value is not None and not _is_rate(value):
                raise _err(f"{who}: {key} must be a number in [0, 1] or null")
        for key in _OPT_POS_INTS:
            value = getattr(self, key)
            if value is not None and not (_is_count(value) and value > 0):
                raise _err(f"{who}: {key} must be a positive integer or null")
        if self.train_loss is not None and not _is_num(self.train_loss):
            raise _err(f"{who}: train_loss must be a number or null")
        fc = self.failure_classes
        if not isinstance(fc, Mapping) or not all(
            isinstance(k, str) and k.strip() and _is_count(v) for k, v in fc.items()
        ):
            raise _err(f"{who}: failure_classes must map class names to non-negative counts")
        if self.gate_fold not in (None, "fit", "selection"):
            raise _err(f"{who}: gate_fold must be 'fit' or 'selection'")
        for key in _OPT_SHAS:
            value = getattr(self, key)
            if value is not None:
                _check_source(f"{who} {key}", self.source, value)

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any], *, source: str, sha256: str) -> CandidateSummary:
        if not isinstance(doc, Mapping):
            raise _err(f"{source}: a candidate summary must be a JSON object")
        allowed = {f.name for f in fields(cls)} - {"source", "sha256"}
        unknown = sorted(set(doc) - allowed)
        if unknown:
            raise _err(f"{source}: unknown summary fields: {', '.join(unknown)}")
        required = {"name", *_COUNTS, *_RATES}
        missing = sorted(required - set(doc))
        if missing:
            raise _err(f"{source}: summary missing: {', '.join(missing)}")
        return cls(**dict(doc), source=source, sha256=sha256)

    def gate_stale(self) -> str | None:
        """Why this candidate's gate must be refit, or None."""
        if self.gate_fold == "selection":
            return "its gate was fit on the selection fold (it must never peek, nvsh D47)"
        if (
            self.gate_predictions_sha256
            and self.predictions_sha256
            and self.gate_predictions_sha256 != self.predictions_sha256
        ):
            return "its gate was fit on other predictions than the ones being judged"
        return None

    def citations(self) -> list[dict[str, Any]]:
        out = []
        for key in _CITED_METRICS:
            value = getattr(self, key)
            if value is not None:
                out.append(records.cite(f"{self.name}.{key}", value, self.source, self.sha256))
        return out


@dataclass(frozen=True)
class ClassYield:
    """Review yield of one class (accepted / reviewed) from an augment or draft run."""

    name: str
    accepted: int
    reviewed: int
    source: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise _err("yield class name must be a non-empty string")
        _check_source(f"yield {self.name}", self.source, self.sha256)
        if not (_is_count(self.accepted) and _is_count(self.reviewed)):
            raise _err(f"yield {self.name}: accepted and reviewed must be non-negative integers")
        if self.accepted > self.reviewed:
            raise _err(f"yield {self.name}: accepted exceeds reviewed")

    @property
    def rate(self) -> float:
        return self.accepted / self.reviewed if self.reviewed else 0.0


@dataclass(frozen=True)
class QuantCheck:
    """A quantized build measured against its own bf16 checkpoint."""

    candidate: str
    bf16_right: float
    quant_right: float
    bf16_wrong_mutating_ids: frozenset[str]
    quant_wrong_mutating_ids: frozenset[str]
    heal_rounds: int
    source: str
    sha256: str

    def __post_init__(self) -> None:
        _check_source(f"quant {self.candidate}", self.source, self.sha256)
        if not (_is_rate(self.bf16_right) and _is_rate(self.quant_right)):
            raise _err(f"quant {self.candidate}: right-proposal rates must be in [0, 1]")
        if not _is_count(self.heal_rounds):
            raise _err(f"quant {self.candidate}: heal_rounds must be a non-negative integer")

    @property
    def loss_points(self) -> float:
        return round((self.bf16_right - self.quant_right) * 100.0, 9)

    @property
    def new_wrong_mutating_ids(self) -> list[str]:
        return sorted(set(self.quant_wrong_mutating_ids) - set(self.bf16_wrong_mutating_ids))

    def heal_needed(self) -> bool:
        return self.loss_points > HEAL_MARGIN_POINTS or bool(self.new_wrong_mutating_ids)

    def citations(self) -> list[dict[str, Any]]:
        c = self.candidate
        return [
            records.cite(f"{c}.bf16.right_proposals", self.bf16_right, self.source, self.sha256),
            records.cite(f"{c}.quant.right_proposals", self.quant_right, self.source, self.sha256),
            records.cite(
                f"{c}.quant.new_wrong_mutating",
                len(self.new_wrong_mutating_ids),
                self.source,
                self.sha256,
            ),
            records.cite(f"{c}.heal_rounds", self.heal_rounds, self.source, self.sha256),
        ]


@dataclass(frozen=True)
class Decision:
    verdict: str
    params: dict[str, Any]
    reasons: list[str]
    cited: list[dict[str, Any]]
    trail: list[dict[str, Any]]
    prereg_sha256: str | None = None

    def record_fields(self) -> dict[str, Any]:
        """Keyword arguments for :func:`records.append` (decider ``rule``)."""
        return {
            "verdict": self.verdict,
            "params": self.params,
            "reasons": self.reasons,
            "cited": self.cited,
            "rule_version": RULE_VERSION,
            "prereg_sha256": self.prereg_sha256,
            "decider": "rule",
            "details": {"trail": self.trail},
        }


def load_summary(path: Path) -> CandidateSummary:
    """Read a candidate summary JSON file, hashing it for citation."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CliError(EXIT_ENV_ERROR, f"cannot read {path}: {exc}", "check the path") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _err(f"{path} is not valid JSON: {exc}") from exc
    return CandidateSummary.from_dict(doc, source=str(path), sha256=records.sha256_file(path))


# --- the lexicographic rule ---------------------------------------------------


def _step(name: str, survivors: Sequence[CandidateSummary], detail: str) -> dict[str, Any]:
    return {"step": name, "survivors": [c.name for c in survivors], "detail": detail}


def select(
    prereg: Prereg, candidates: Sequence[CandidateSummary]
) -> tuple[CandidateSummary | None, list[dict[str, Any]]]:
    """Apply the registered rule order; return the winner (or None) and the trail."""
    tol = prereg.tolerances
    floor = prereg.bars["right_proposals"].bar
    order = {name: i for i, name in enumerate(prereg.candidates)}
    alive = list(candidates)
    trail: list[dict[str, Any]] = []
    for step in prereg.rule_order:
        if step == "safety":
            alive = [c for c in alive if c.wrong_mutating == 0 and c.wrong_mutating_mc == 0]
            detail = "0 wrong mutating on the selection fold and its missing-candidate slice"
        elif step == "accuracy_floor":
            alive = [c for c in alive if c.right_proposals >= floor - _EPS]
            detail = f"right proposals >= the pre-registered bar {floor:.4f}"
        elif step == "permutation_robustness" and alive:
            best = min(c.permutation_change for c in alive)
            limit = best + tol["permutation_change"]
            alive = [c for c in alive if c.permutation_change <= limit + _EPS]
            detail = f"permutation change within {tol['permutation_change']} of best {best:.4f}"
        elif step == "calibration" and alive:
            best = min(c.ece for c in alive)
            alive = [c for c in alive if c.ece <= best + tol["ece"] + _EPS]
            detail = f"ECE within {tol['ece']} of best {best:.4f}"
        elif step == "mc_escalation" and alive:
            best = max(c.mc_escalation for c in alive)
            alive = [c for c in alive if c.mc_escalation >= best - _EPS]
            detail = f"highest missing-candidate escalation {best:.4f}"
        elif step == "ties" and alive:
            alive = sorted(alive, key=lambda c: (-c.right_proposals, order[c.name]))[:1]
            detail = "more right proposals, then the simpler recipe (pre-registered order)"
        else:
            detail = "no candidate left"
        trail.append(_step(step, alive, detail))
    return (alive[0] if alive else None), trail


# --- diagnosis ---------------------------------------------------------------


def _step_count_problem(c: CandidateSummary) -> str | None:
    known = (c.epochs, c.train_examples, c.batch_size, c.steps)
    if any(v is None for v in known):
        return None
    expected = math.ceil(c.train_examples / c.batch_size) * c.epochs
    if c.steps != expected:
        return (
            f"{c.name}'s run record has {c.steps} steps, not ceil({c.train_examples}/"
            f"{c.batch_size}) x {c.epochs} = {expected}; its epochs cannot be trusted"
        )
    return None


def signature(c: CandidateSummary, ref: CandidateSummary) -> str | None:
    """``overfit``, ``underfit`` or None, against a reference run."""
    if c.mean_confidence is None or ref.mean_confidence is None:
        return None
    conf_drop = ref.mean_confidence - c.mean_confidence
    recall_fell = (
        c.abstain_recall is not None
        and ref.abstain_recall is not None
        and c.abstain_recall < ref.abstain_recall - _EPS
    )
    if conf_drop <= CONFIDENCE_HOLD + _EPS:
        return "overfit" if recall_fell else None
    if c.right_proposals < ref.right_proposals - _EPS or recall_fell:
        return "underfit"
    return None


def _diagnose(
    failing: Sequence[CandidateSummary],
    reference: CandidateSummary | None,
    why: list[str],
) -> tuple[str, dict[str, Any], list[str]]:
    classes: Counter[str] = Counter()
    for c in failing:
        classes.update({k: v for k, v in c.failure_classes.items() if v > 0})
    if classes:
        top = min(classes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return (
            "targeted_augment",
            {"class": top},
            why
            + [
                f"a failure class is a data request: {top} has {classes[top]} failing "
                f"entries (all classes: {dict(sorted(classes.items()))})"
            ],
        )
    if reference is not None:
        for c in failing:
            problem = _step_count_problem(c)
            if problem:
                return "stop_escalate", {}, why + [problem]
        sigs = {c.name: signature(c, reference) for c in failing}
        over = [c for c in failing if sigs[c.name] == "overfit"]
        under = [c for c in failing if sigs[c.name] == "underfit"]
        if over and under:
            return (
                "stop_escalate",
                {},
                why + ["conflicting signatures: overfit and underfit candidates together"],
            )
        if over:
            c = over[0]
            return (
                "fewer_epochs",
                {"candidate": c.name, "epochs": c.epochs},
                why
                + [
                    f"overfit signature vs {reference.name}: confidence stays high "
                    f"({c.mean_confidence} vs {reference.mean_confidence}) while abstain "
                    f"recall falls ({c.abstain_recall} vs {reference.abstain_recall})"
                ],
            )
        if under:
            c = under[0]
            return (
                "more_epochs",
                {"candidate": c.name, "epochs": c.epochs},
                why
                + [
                    f"underfit signature vs {reference.name}: confidence "
                    f"({c.mean_confidence} vs {reference.mean_confidence}) and accuracy "
                    f"({c.right_proposals} vs {reference.right_proposals}) fall together"
                ],
            )
    return (
        "stop_escalate",
        {},
        why + ["no mechanical verdict applies; the human decides (follow the diagnostic)"],
    )


# --- decide --------------------------------------------------------------------


def _check_inputs(
    prereg: Prereg, candidates: Sequence[CandidateSummary], hard_stops: Sequence[str]
) -> None:
    if tuple(prereg.rule_order) != RULE_ORDER:
        raise _err(
            f"the registered rule order {list(prereg.rule_order)} is not {list(RULE_ORDER)}",
            "a rule change is a recorded deviation, never an edit",
        )
    unknown = [s for s in hard_stops if s not in HARD_STOPS]
    if unknown:
        raise _err(f"unknown hard stop: {', '.join(unknown)} (known: {list(HARD_STOPS)})")
    if not candidates:
        raise _err("decide needs at least one candidate summary")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise _err("candidate summaries must have unique names")
    stray = [n for n in names if n not in prereg.candidates]
    if stray:
        raise _err(
            f"candidate {', '.join(stray)} is not pre-registered",
            "an out-of-plan run is a recorded deviation",
        )


def decide(
    prereg: Prereg,
    candidates: Sequence[CandidateSummary],
    *,
    prereg_sha256: str | None = None,
    reference: CandidateSummary | None = None,
    yields: Iterable[ClassYield] = (),
    quant: QuantCheck | None = None,
    hard_stops: Sequence[str] = (),
) -> Decision:
    """Apply the pre-registered rule mechanically and return one verdict."""
    hard_stops = list(hard_stops)
    yields = list(yields)
    _check_inputs(prereg, candidates, hard_stops)
    cited = [m for c in candidates for m in c.citations()]

    def out(verdict, params, reasons, trail=()):
        return Decision(verdict, params, list(reasons), cited, list(trail), prereg_sha256)

    if hard_stops:
        return out(
            "stop_escalate",
            {"hard_stops": hard_stops},
            [f"hard stop: {s} is always a human decision" for s in hard_stops],
        )

    for y in yields:
        cited.append(records.cite(f"yield.{y.name}", y.rate, y.source, y.sha256))
    low = sorted(y.name for y in yields if y.rate < YIELD_FLOOR - _EPS)
    if low:
        return out(
            "fix_grader",
            {"classes": low},
            [
                f"class {y.name} yield {y.accepted}/{y.reviewed} is under "
                f"{YIELD_FLOOR:.0%}: fix the grader before drafting more"
                for y in yields
                if y.name in low
            ],
        )

    if quant is not None:
        cited.extend(quant.citations())
        if quant.heal_needed():
            why = [
                f"{quant.candidate} quant loses {quant.loss_points} pts of right proposals vs "
                f"its own bf16 (heal above {HEAL_MARGIN_POINTS}); new wrong-mutating ids: "
                f"{quant.new_wrong_mutating_ids or 'none'}"
            ]
            if quant.heal_rounds >= HEAL_ROUNDS:
                return out(
                    "stop_escalate",
                    {"candidate": quant.candidate},
                    why + ["heal is one round only and it has been spent; drop this build"],
                )
            return out("heal_quant", {"candidate": quant.candidate}, why)

    stale = [(c.name, c.gate_stale()) for c in candidates if c.gate_stale()]
    if stale:
        return out(
            "refit_gate",
            {"candidates": [n for n, _ in stale]},
            [f"{n}: {why}; refit it on the fit fold before judging" for n, why in stale],
        )

    winner, trail = select(prereg, candidates)
    if winner is None:
        verdict, params, reasons = _diagnose(
            candidates, reference, ["no candidate survives the safety and accuracy filters"]
        )
    elif winner.ece > RECALIBRATE_ECE + _EPS:
        verdict, params, reasons = (
            "recalibrate",
            {"candidate": winner.name},
            [
                f"rule picks {winner.name} but its ECE {winner.ece} is above "
                f"{RECALIBRATE_ECE}: run the calibration-aware stage"
            ],
        )
    else:
        # wrong_mutating and right_proposals are already the safety and floor steps.
        missed = [
            f"{name} {getattr(winner, name)} misses the bar {bar.bar}"
            for name, bar in prereg.bars.items()
            if name in ("ece", "permutation_change", "mc_escalation")
            and not bar.met(getattr(winner, name))
        ]
        if missed:
            verdict, params, reasons = _diagnose(
                [winner], reference, [f"rule picks {winner.name}, but {m}" for m in missed]
            )
        else:
            verdict, params, reasons = (
                "ship_candidate",
                {"candidate": winner.name},
                [f"{winner.name} wins the pre-registered rule and meets every bar"]
                + [f"{s['step']}: {s['survivors']} ({s['detail']})" for s in trail],
            )
    if reference is not None and verdict in ("more_epochs", "fewer_epochs"):
        cited.extend(reference.citations())
    return out(verdict, params, reasons, trail)
