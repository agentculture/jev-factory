"""Permutation probe: does the scorer's choice survive reordering, re-lettering, subsets
and paraphrased descriptions?

For every entry of a split, the scorer is asked once with the default fixed
order and positional letters (the *baseline*), then ``--per-entry`` (default
10) seeded perturbations of each of five kinds, each built through the
causal-LM adapter's own seam (:func:`jev_factory.backbones.causal_lm.scorer.permute`,
:func:`~jev_factory.backbones.causal_lm.scorer.prompt_messages`' ``labels``/
``order``/``descriptions``):

``order``
    Same name -> letter map as the baseline; the listing order is shuffled.
``letters``
    Same listing order; the name -> letter map is re-drawn from the whole
    alphabet (a candidate may land on a lower-case letter).
``subset``
    The baseline order and letters, restricted to a random subset that always
    keeps the gold candidate; how often it also drops the baseline's own
    choice is reported separately.
``paraphrase``
    The baseline order and letters; one random alternative description per
    candidate (the domain's paraphrases, or ``--paraphrases``' JSON) replaces
    its default description.
``all``
    Order, letters and subset in one draw (the OpenJev-style full shuffle).

An *answer change* is a trial whose choice (compared by candidate name, never
by letter) differs from the baseline's. Seeds derive deterministically from
``(--seed, entry id, kind, trial index)``, so a report is exactly
reproducible. Trials of one entry are correlated, so each kind's 95% CI is a
bootstrap over **entries** (:func:`jev_factory.core.metrics.bootstrap_ci`),
never the flat trial list. A trial (or baseline) the scorer could not fully
read -- a letter missing from its readout, or no mass on any letter -- is
tallied as ``incomplete`` and never counted as a change or a non-change.

The scorer never sees a canonicalised order or letter map. The probe refuses
a split that looks like the test or held-out side unless ``--final``.
``--reasons`` probes the reasons pool (one ``escalate:<reason>`` per domain
reason in place of the bare escalate); an escalation's gold is then its own
reason (:meth:`Domain.reason_for_class`).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.core import metrics
from jev_factory.core.calibration import split_markers
from jev_factory.domain.model import ESCALATE, EXPLAIN, Domain
from jev_factory.measure.corpus import CorpusEntry

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/permutation_probe.py",
    "commit": "9debdc6",
    "adaptations": [
        "_sibling path loading of scorer/metrics/calibration_fit and the nvsh.tiers"
        " bench/lfm imports (lines 78-96) replaced by package imports and the Domain",
        "every scorer call takes the Domain and an injectable top-k function"
        " (scorer.score(domain, top_k, ...)) instead of a NextTokenScorer object",
        "the request text is the entry's text (tier_lfm.request_message dropped); gold names"
        " use Domain.reason_for_class and the domain's EXPLAIN/ESCALATE names",
        "--paraphrases defaults to the Domain's own paraphrases (at least 1 alternative"
        " each); a --paraphrases file keeps nvsh's at-least-2 rule",
        "full_alphabet_labels is dropped: the adapter's InProcessTopK already reads every"
        " letter of the alphabet; _build_real_scorer goes through measure.run.build_scorer",
    ],
    "licence": "Apache-2.0",
}

#: The five perturbation kinds, in report order.
KINDS = ("order", "letters", "subset", "paraphrase", "all")

DEFAULT_PER_ENTRY = 10
DEFAULT_SEED = 0

Render = Callable[[list[dict]], str]


class ProbeError(ValueError):
    """A refused input: the wrong split, a malformed paraphrase file, and so on."""


# ---------------------------------------------------------------------------
# Gold / paraphrases
# ---------------------------------------------------------------------------


def gold_name(
    domain: Domain,
    expect: Mapping[str, object],
    *,
    reasons: bool = False,
    cls: str | None = None,
) -> str:
    """The candidate *name* (never a calibration label) the corpus entry expects."""
    kind = metrics.expect_kind(expect)
    if kind == "escalate":
        return domain.reason_for_class(cls) if reasons else ESCALATE
    if kind == "explain":
        return EXPLAIN
    return str(expect["operation"])


def load_paraphrases(path: Path) -> dict[str, list[str]]:
    """``{candidate: [alt description, ...]}`` from *path*; refuses a malformed file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeError(f"cannot read paraphrase file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProbeError(f"{path} is not a {{candidate: [alt description, ...]}} object")
    result: dict[str, list[str]] = {}
    for name, alts in raw.items():
        if not isinstance(alts, list) or not all(isinstance(alt, str) for alt in alts):
            raise ProbeError(f"{path}: {name!r}'s alternatives must be a list of strings")
        if len(alts) < 2:
            raise ProbeError(f"{path}: {name!r} has fewer than 2 alternative descriptions")
        result[name] = list(alts)
    return result


def domain_paraphrases(domain: Domain) -> dict[str, list[str]]:
    """The domain's own ``{operation: [alt description, ...]}``."""
    return {name: list(texts) for name, texts in domain.paraphrases if texts}


# ---------------------------------------------------------------------------
# Seeded draws per kind
# ---------------------------------------------------------------------------


def derive_seed(seed: object, entry_id: str, kind: str, index: int) -> str:
    """One deterministic, reproducible seed for (*seed*, *entry_id*, *kind*, *index*)."""
    return f"{seed}:{entry_id}:{kind}:{index}"


def _random_subset_order(
    rng: random.Random, order: Sequence[str], keep: Sequence[str]
) -> list[str]:
    """A random subset of *order*, in *order*'s own order, always keeping *keep*."""
    keep_set = dict.fromkeys(keep)
    rest = [name for name in order if name not in keep_set]
    size = rng.randint(len(keep_set), len(order))
    extra_count = max(0, min(size - len(keep_set), len(rest)))
    extra = set(rng.sample(rest, extra_count)) if extra_count else set()
    chosen = set(keep_set) | extra
    return [name for name in order if name in chosen]


@dataclass(frozen=True)
class Trial:
    """One perturbation's rendered order/labels/descriptions, ready to score."""

    order: tuple[str, ...]
    labels: dict[str, str]
    descriptions: dict[str, str] | None = None
    #: ``subset`` only: whether this draw dropped the baseline's own choice.
    baseline_choice_dropped: bool | None = None


def build_trial(
    kind: str,
    seed: object,
    pool: Sequence[str],
    baseline_order: Sequence[str],
    baseline_labels: Mapping[str, str],
    gold: str,
    baseline_choice: str | None,
    paraphrases: Mapping[str, Sequence[str]] | None,
) -> Trial:
    """The *kind* perturbation drawn from *seed*, isolating exactly what that kind varies."""
    if kind == "order":
        return Trial(order=sc.permute(seed, pool).order, labels=dict(baseline_labels))
    if kind == "letters":
        return Trial(order=tuple(baseline_order), labels=sc.permute(seed, pool).labels)
    if kind == "subset":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        order = _random_subset_order(rng, baseline_order, keep=(gold,))
        labels = {name: baseline_labels[name] for name in order}
        dropped = baseline_choice is not None and baseline_choice not in order
        return Trial(order=tuple(order), labels=labels, baseline_choice_dropped=dropped)
    if kind == "paraphrase":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        descriptions: dict[str, str] = {}
        for name in baseline_order:
            alts = (paraphrases or {}).get(name)
            if alts:
                descriptions[name] = rng.choice(list(alts))
        return Trial(
            order=tuple(baseline_order), labels=dict(baseline_labels), descriptions=descriptions
        )
    if kind == "all":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        keep = (gold,)
        subset_size = rng.randint(len(keep), len(pool))
        perm = sc.permute(seed, pool, subset=subset_size, keep=keep)
        return Trial(order=perm.order, labels=perm.labels)
    raise ProbeError(f"unknown perturbation kind: {kind!r}")


# ---------------------------------------------------------------------------
# Scoring one entry
# ---------------------------------------------------------------------------


@dataclass
class EntryOutcome:
    """One entry's baseline plus every kind's trials."""

    entry_id: str
    gold: str
    baseline_choice: str | None
    #: kind -> list of (changed, has_lowercase, baseline_choice_dropped-or-None, incomplete)
    trials: dict[str, list[tuple[bool, bool, bool | None, bool]]] = field(default_factory=dict)


def _score(
    domain: Domain,
    top_k: sc.TopK,
    render: Render,
    request_text: str,
    order: Sequence[str],
    labels: Mapping[str, str],
    descriptions: Mapping[str, str] | None,
    reasons: bool,
) -> sc.ScorerPrediction:
    messages = sc.prompt_messages(
        domain,
        request_text,
        labels=labels,
        order=order,
        descriptions=descriptions,
        reasons=reasons,
    )
    return sc.score(
        domain, top_k, render(messages), request_text, labels=labels, order=order, reasons=reasons
    )


def _incomplete(scored: sc.ScorerPrediction) -> bool:
    """True when *scored* made no comparable choice: missing labels, or no label had mass."""
    return scored.choice is None or scored.incomplete is not None


def probe_entry(
    domain: Domain,
    top_k: sc.TopK,
    render: Render,
    entry: CorpusEntry,
    *,
    pool: Sequence[str],
    per_entry: int,
    seed: object,
    paraphrases: Mapping[str, Sequence[str]] | None,
    kinds: Sequence[str] = KINDS,
    reasons: bool = False,
) -> EntryOutcome:
    """Baseline plus *per_entry* trials of each of *kinds* for one corpus entry."""
    request_text = entry.text
    gold = gold_name(domain, entry.expect, reasons=reasons, cls=entry.phrasing or None)
    if reasons:
        baseline_labels = sc.positional_labels(pool, pool)
    else:
        baseline_labels = sc.labels_for(domain, pool)
    baseline_order = list(pool)
    baseline = _score(
        domain, top_k, render, request_text, baseline_order, baseline_labels, None, reasons
    )
    baseline_incomplete = _incomplete(baseline)
    outcome = EntryOutcome(entry_id=entry.id, gold=gold, baseline_choice=baseline.choice)
    for kind in kinds:
        trials: list[tuple[bool, bool, bool | None, bool]] = []
        for index in range(per_entry):
            if kind == "paraphrase" and not paraphrases:
                break
            trial = build_trial(
                kind,
                derive_seed(seed, entry.id, kind, index),
                pool,
                baseline_order,
                baseline_labels,
                gold,
                baseline.choice,
                paraphrases,
            )
            scored = _score(
                domain,
                top_k,
                render,
                request_text,
                trial.order,
                trial.labels,
                trial.descriptions,
                reasons,
            )
            # A baseline or trial the scorer could not fully read is not evidence of an
            # answer change either way -- tallied separately, never scored.
            incomplete = baseline_incomplete or _incomplete(scored)
            changed = False if incomplete else not sc.same_choice(scored, baseline)
            has_lowercase = any(letter.islower() for letter in trial.labels.values())
            trials.append((changed, has_lowercase, trial.baseline_choice_dropped, incomplete))
        if trials:
            outcome.trials[kind] = trials
    return outcome


# ---------------------------------------------------------------------------
# Aggregation: bootstrap over entries, not trials
# ---------------------------------------------------------------------------


def _rate_stat(pairs: Sequence[tuple[int, int]]) -> float | None:
    total_trials = sum(trials for trials, _ in pairs)
    if total_trials == 0:
        return None
    return sum(changes for _, changes in pairs) / total_trials


def kind_report(outcomes: Sequence[EntryOutcome], kind: str, *, bootstrap_seed: int) -> dict | None:
    """One kind's report: trials, changes, rate with a bootstrap CI over entries, label case."""
    per_entry_pairs: list[tuple[int, int]] = []
    lowercase_trials = lowercase_changes = 0
    uppercase_trials = uppercase_changes = 0
    baseline_dropped = subset_trials_seen = 0
    incomplete_count = raw_trial_count = 0
    for outcome in outcomes:
        trials = outcome.trials.get(kind)
        if not trials:
            continue
        raw_trial_count += len(trials)
        incomplete_count += sum(1 for *_, incomplete in trials if incomplete)
        scored_trials = [
            (changed, has_lowercase, dropped)
            for changed, has_lowercase, dropped, incomplete in trials
            if not incomplete
        ]
        if scored_trials:
            changes = sum(1 for changed, _, _ in scored_trials if changed)
            per_entry_pairs.append((len(scored_trials), changes))
        for changed, has_lowercase, dropped in scored_trials:
            if has_lowercase:
                lowercase_trials += 1
                lowercase_changes += int(changed)
            else:
                uppercase_trials += 1
                uppercase_changes += int(changed)
            if dropped is not None:
                subset_trials_seen += 1
                baseline_dropped += int(dropped)
    if raw_trial_count == 0:
        return None
    ci = metrics.bootstrap_ci(per_entry_pairs, _rate_stat, seed=bootstrap_seed)
    report = {
        "kind": kind,
        "entries": len(per_entry_pairs),
        "trials": sum(t for t, _ in per_entry_pairs),
        "changes": sum(c for _, c in per_entry_pairs),
        "rate": ci["value"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "bootstrap_note": "95% CI is a bootstrap over entries, not trials (trials of one"
        " entry are correlated)",
        "incomplete": {
            "trials": incomplete_count,
            "of": raw_trial_count,
            "rate": (incomplete_count / raw_trial_count) if raw_trial_count else None,
        },
        "label_case": {
            "lowercase_trials": lowercase_trials,
            "lowercase_changes": lowercase_changes,
            "lowercase_rate": (lowercase_changes / lowercase_trials) if lowercase_trials else None,
            "uppercase_trials": uppercase_trials,
            "uppercase_changes": uppercase_changes,
            "uppercase_rate": (uppercase_changes / uppercase_trials) if uppercase_trials else None,
        },
    }
    if kind == "subset":
        report["baseline_choice_removed"] = {
            "trials": subset_trials_seen,
            "removed": baseline_dropped,
            "rate": (baseline_dropped / subset_trials_seen) if subset_trials_seen else None,
        }
    return report


def run_probe(
    domain: Domain,
    top_k: sc.TopK,
    render: Render,
    entries: Sequence[CorpusEntry],
    *,
    pool: Sequence[str] | None = None,
    per_entry: int = DEFAULT_PER_ENTRY,
    seed: object = DEFAULT_SEED,
    paraphrases: Mapping[str, Sequence[str]] | None = None,
    bootstrap_seed: int = metrics.DEFAULT_BOOTSTRAP_SEED,
    reasons: bool = False,
    progress: Any = None,
) -> dict:
    """The full probe report: every kind's aggregate over *entries*. *progress* (a
    :class:`~jev_factory.factory.detach.Progress`) is advanced once per entry."""
    resolved_pool = tuple(sc.candidate_pool(domain, reasons) if pool is None else pool)
    if len(resolved_pool) > len(ro.LABEL_ALPHABET):
        raise ProbeError(f"{len(resolved_pool)} candidates but only 52 letters")
    outcomes = []
    for entry in entries:
        outcomes.append(
            probe_entry(
                domain,
                top_k,
                render,
                entry,
                pool=resolved_pool,
                per_entry=per_entry,
                seed=seed,
                paraphrases=paraphrases,
                reasons=reasons,
            )
        )
        if progress is not None:
            progress.advance()
    kinds_report = [
        report
        for kind in KINDS
        if (report := kind_report(outcomes, kind, bootstrap_seed=bootstrap_seed)) is not None
    ]
    return {
        "per_entry": per_entry,
        "seed": str(seed),
        "entries": len(entries),
        "kinds": kinds_report,
        "canonicalised_order": False,
        "reasons": reasons,
    }


def pooled_change_rate(report: Mapping[str, object]) -> float | None:
    """Answer changes over scored trials, pooled across every kind (the OpenJev bar's figure)."""
    kinds = report.get("kinds") or []
    trials = sum(int(k["trials"]) for k in kinds)  # type: ignore[index,union-attr]
    changes = sum(int(k["changes"]) for k in kinds)  # type: ignore[index,union-attr]
    return (changes / trials) if trials else None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "n/a"


def render_markdown(report: Mapping[str, object]) -> str:
    """A markdown table of the probe report, one row per kind."""
    lines = [
        f"# Permutation probe ({report['entries']} entries, {report['per_entry']} per kind, seed"
        f" {report['seed']})",
        "",
        "The scorer is never canonicalised: each kind's draw is rendered and scored exactly as"
        " drawn.",
        "",
        "| kind | trials | changes | rate | 95% CI | lowercase rate | uppercase rate |"
        " incomplete |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    kinds = report["kinds"]
    for kind_data in kinds:  # type: ignore[union-attr]
        case = kind_data["label_case"]
        incomplete = kind_data["incomplete"]
        ci = f"[{_fmt(kind_data['ci_low'])}, {_fmt(kind_data['ci_high'])}]"
        lines.append(
            f"| {kind_data['kind']} | {kind_data['trials']} | {kind_data['changes']} |"
            f" {_fmt(kind_data['rate'])} | {ci}"
            f" | {_fmt(case['lowercase_rate'])} | {_fmt(case['uppercase_rate'])}"
            f" | {incomplete['trials']}/{incomplete['of']} |"
        )
        if kind_data["kind"] == "subset":
            dropped = kind_data["baseline_choice_removed"]
            lines.append(
                f"\nsubset: baseline choice removed in {dropped['removed']}/{dropped['trials']}"
                f" trials ({_fmt(dropped['rate'])})."
            )
    lines.append("")
    lines.append(f"_{kinds[0]['bootstrap_note'] if kinds else ''}_")  # type: ignore[index]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_real_scorer(args: argparse.Namespace):  # a model or server
    from jev_factory.measure import run as measure

    spec = measure.ScorerSpec(
        kind=args.scorer,
        model=args.model,
        revision=args.revision,
        tokenizer=args.tokenizer,
        base_url=args.base_url,
    )
    return measure.build_scorer(spec)


def main(
    argv: Sequence[str] | None = None,
    *,
    build_scorer: Callable[[argparse.Namespace], object] = _build_real_scorer,
) -> int:
    from jev_factory.domain.validate import DomainError, load_domain
    from jev_factory.measure.corpus import load_corpus

    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.measure.probe",
        description="Answer-change rate under seeded order/letter/subset/paraphrase"
        " permutations.",
    )
    parser.add_argument("--domain", required=True, help="domain module or JSON domain file")
    parser.add_argument("--split", required=True, help="split or corpus file to probe")
    parser.add_argument("--paraphrases", default=None, help="{candidate: [alt, ...]} JSON")
    parser.add_argument("--per-entry", type=int, default=DEFAULT_PER_ENTRY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=metrics.DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--final", action="store_true", help="allow a test/held-out split")
    parser.add_argument("--out", default=None, help="JSON report path (default: stdout)")
    parser.add_argument("--markdown", default=None, help="markdown table path")
    parser.add_argument(
        "--progress-dir",
        default=None,
        help="write <dir>/probe-<out stem>.progress.json after every entry (jev status reads it)",
    )
    parser.add_argument("--model", default=None, help="model path, or the served model name")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--base-url", default=None, help="served: attached localhost endpoint")
    parser.add_argument("--scorer", default="in-process", choices=("served", "in-process"))
    parser.add_argument("--reasons", action="store_true", help="probe the escalate:<r> pool")
    args = parser.parse_args(argv)

    split_path = Path(args.split)
    try:
        domain = load_domain(args.domain)
        raw = json.loads(split_path.read_text(encoding="utf-8"))
    except (DomainError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    header = raw.get("header") if isinstance(raw, dict) else None
    markers = split_markers(split_path, header)
    if markers and not args.final:
        print(
            f"error: {split_path} looks like the {' and '.join(sorted(markers))} split; "
            "refusing to probe it without --final",
            file=sys.stderr,
        )
        return 1
    loaded = load_corpus(split_path, domain)
    if not loaded.entries:
        print(f"error: {split_path} has no valid entries", file=sys.stderr)
        return 1
    try:
        paraphrases = (
            load_paraphrases(Path(args.paraphrases))
            if args.paraphrases
            else domain_paraphrases(domain)
        )
    except ProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.model:
        print("error: --model is required (no fake scorer is wired to the CLI)", file=sys.stderr)
        return 1
    if args.scorer == "served" and not args.base_url:
        print("error: --scorer served needs --base-url", file=sys.stderr)
        return 1
    progress = None
    if args.progress_dir:
        from jev_factory.factory.detach import Progress

        where = Path(args.out) if args.out else Path(args.split)
        stem = f"{where.parent.name}-{where.stem}" if where.parent.name else where.stem
        progress = Progress(
            Path(args.progress_dir), "probe-" + stem.replace(".", "_"), len(loaded.entries)
        )
    handle = build_scorer(args)
    try:
        report = run_probe(
            domain,
            handle.top_k,  # type: ignore[attr-defined]
            handle.render,  # type: ignore[attr-defined]
            loaded.entries,
            per_entry=args.per_entry,
            seed=args.seed,
            paraphrases=paraphrases,
            bootstrap_seed=args.bootstrap_seed,
            reasons=args.reasons,
            progress=progress,
        )
    finally:
        handle.close()  # type: ignore[attr-defined]
    report["pooled_change_rate"] = pooled_change_rate(report)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
