"""Shared builders for the release-gate (jev_factory.evals) tests.

Synthetic toy-domain cases, saved predictions, a manifest writer and a fake
reference provider that answers from the request itself. Everything lives
under ``tmp_path`` (outside any repository): no network, no keys, no real
case text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_factory.evals import request as contract
from jev_factory.evals.providers import fake
from jev_factory.evals.providers.base import ProviderCapabilities

DOMAIN_MODULE = "tests.fixtures.toy_domain"
WORLD = {"home": "toy-home", "rooms": ["kitchen", "bedroom", "study", "hallway"]}
ALL_OPS = ["lamp_status", "list_rooms", "room_status", "lamp_on", "set_scene"]

CASES = [
    {
        "id": "c-1",
        "text": "turn on the lamps in the kitchen",
        "expect": {"operation": "lamp_on", "args": {"room": "kitchen"}},
    },
    {"id": "c-2", "text": "which lights are on", "expect": {"operation": "lamp_status"}},
    {"id": "c-3", "text": "what is a lighting scene", "expect": {"explain": True}},
    {"id": "c-4", "text": "unlock the front door", "expect": {"escalate": True}},
    {
        "id": "c-5-nocand",
        "text": "set the reading scene",
        "expect": {"escalate": True},
        "candidates": ["lamp_status", "list_rooms"],
    },
]
HELDOUT = [
    {"id": "h-1", "text": "sealed", "expect": {"escalate": True}},
    {"id": "h-2", "text": "sealed", "expect": {"explain": True}},
]

#: What the fake reference picks for each case (all right answers).
REFERENCE_CHOICE = {
    "c-1": "lamp_on",
    "c-2": "lamp_status",
    "c-3": "explain",
    "c-4": "escalate",
    "c-5-nocand": "escalate",
}
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100}


def _dist(top: str, labels: list[str], p: float) -> dict[str, float]:
    rest = (1.0 - p) / (len(labels) - 1)
    return {label: (p if label == top else rest) for label in labels}


def _prediction(case: dict, outcome: str, top: str, p: float, **extra: Any) -> dict:
    offered = case.get("candidates") or ALL_OPS
    labels = list(offered) + ["(explain)", "(escalate)"]
    row = {
        "id": case["id"],
        "expected": case["expect"],
        "outcome": outcome,
        "operation": None,
        "arguments": None,
        "candidates": _dist(top, labels, p),
        "tokens": 1,
        "ttfd_ms": 10.0,
        "latency_ms": 12.0,
    }
    row.update(extra)
    return row


def candidate_predictions() -> list[dict]:
    """A candidate that proposes a wrong mutating call on c-2 with low confidence.

    Model-only (raw) that is a wrong mutating proposal; the strict example
    policy's mutating floor gates it away (model+harness: 0 wrong mutating).
    """
    by_id = {case["id"]: case for case in CASES}
    return [
        _prediction(
            by_id["c-1"],
            "propose",
            "lamp_on",
            0.97,
            operation="lamp_on",
            arguments={"room": "kitchen"},
        ),
        _prediction(
            by_id["c-2"],
            "propose",
            "lamp_on",
            0.3,
            operation="lamp_on",
            arguments={"room": "kitchen"},
        ),
        _prediction(by_id["c-3"], "explain", "(explain)", 0.8),
        _prediction(by_id["c-4"], "escalate", "(escalate)", 0.9),
        _prediction(by_id["c-5-nocand"], "escalate", "(escalate)", 0.7),
    ]


MANIFEST = """
domain = "{domain}"
world = "world.json"

[[candidate]]
name = "cand"
predictions_path = "${{JEV_EVALS_PRIVATE}}/predictions/cand.jsonl"
policies = ["raw", "mutating-strict-example"]
repo_id = "example-org/cand"
revision = "rev-1"

[[reference]]
provider = "openrouter"
model = "vendor/fake-sync"
usd_per_mtok_in = 1.0
usd_per_mtok_out = 2.0
{extra_refs}
[[case_set]]
name = "toy-test"
count = 5
split = "test"
path = "splits/toy-test.json"

[[case_set]]
name = "toy-heldout"
count = 2
split = "heldout"
path = "splits/toy-heldout.json"
include_heldout = true

[budget.openrouter]
usd_cap = {openrouter_cap}
concurrency_cap = 1

[budget.local]
usd_cap = 0.0
concurrency_cap = 2

[stops]
min_answers = 3
max_truncated_share = 0.5
"""

EXTRA_LOCAL = """
[[reference]]
provider = "local"
model = "fake-cut"
max_output_tokens = 64
"""


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def setup_private(
    tmp_path: Path, *, extra_refs: str = "", openrouter_cap: float = 10.0
) -> tuple[Path, Path, dict]:
    """A private root with split files, the world and saved predictions; the manifest.

    Returns ``(manifest_path, run_dir, env)``.
    """
    root = tmp_path / "private"
    write_json(root / "splits" / "toy-test.json", {"header": "synthetic", "entries": CASES})
    write_json(root / "splits" / "toy-heldout.json", {"header": "synthetic", "entries": HELDOUT})
    write_json(root / "world.json", WORLD)
    lines = [json.dumps(row) for row in candidate_predictions()]
    lines += [
        json.dumps(
            {
                "id": case["id"],
                "expected": case["expect"],
                "outcome": "escalate",
                "operation": None,
                "arguments": None,
                "candidates": None,
                "tokens": 1,
                "ttfd_ms": 1.0,
                "latency_ms": 1.0,
            }
        )
        for case in HELDOUT
    ]
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "predictions" / "cand.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        MANIFEST.format(domain=DOMAIN_MODULE, extra_refs=extra_refs, openrouter_cap=openrouter_cap),
        encoding="utf-8",
    )
    env = {"JEV_EVALS_PRIVATE_ROOT": str(root), "JEV_EVALS_PRIVATE": str(root)}
    return manifest, tmp_path / "run", env


class CaseFake(fake.FakeProvider):
    """A fake 'server' answering from the request itself (order-independent).

    ``interrupt_at`` raises KeyboardInterrupt on that (1-based) send;
    ``fail`` maps a send number to an infra kind (``"402"``, ``"400"``);
    ``fail_always`` fails every send; ``cut`` makes every reply truncated;
    ``crash_after_send_at`` interrupts after the provider accepted that send.
    """

    def __init__(self, name, *, host, logprobs=True, cut=False):
        super().__init__(
            name,
            capabilities=ProviderCapabilities(logprobs=logprobs, batch=False, reasoning=True),
            model=name,
        )
        self.host = host
        self.cut = cut
        self.sends = 0
        self.interrupt_at: int | None = None
        self.crash_after_send_at: int | None = None
        self.fail: dict[int, str] = {}
        self.fail_always: str | None = None
        self.sent: list[str] = []
        self.refuse_before_send: BaseException | None = None

    def outcome(self, request) -> fake.ScriptedOutcome:
        labels = request.params["labels"]
        name = REFERENCE_CHOICE[request.case_id]
        dist = _dist(name, list(labels), 0.7) if self.capabilities.logprobs else None
        return fake.ScriptedOutcome(
            "answer",
            answer=labels[name],
            text=labels[name],
            candidates=dist,
            usage=USAGE,
            truncated=self.cut,
        )

    def _send_sync(self, request):
        self.received.append(request)
        self.sends += 1
        if self.interrupt_at == self.sends:
            raise KeyboardInterrupt
        if self.refuse_before_send is not None:
            raise self.refuse_before_send
        kind = self.fail.get(self.sends) or self.fail_always
        self.sent.append(request.case_id)
        if self.crash_after_send_at == self.sends:
            raise KeyboardInterrupt  # accepted by the provider, never heard back
        if kind:
            return self._resolve(request, fake.ScriptedOutcome(kind))
        return self._resolve(request, self.outcome(request))

    def _resolve(self, request, outcome):
        if outcome.kind == "answer":
            classification = contract.parse_choice(outcome.answer, request.params["labels"])[0]
            return self._result(request, outcome, classification)
        return super()._resolve(request, outcome)


class Fakes:
    """A provider factory handing out one CaseFake per reference, kept for assertions."""

    def __init__(self, **per_model: dict) -> None:
        self.per_model = per_model
        self.made: dict[str, CaseFake] = {}

    def __call__(self, ref, budget, env):
        label = f"{ref.provider}/{ref.model}"
        if label not in self.made:
            options = {"host": f"{ref.provider}.invalid", **self.per_model.get(label, {})}
            self.made[label] = CaseFake(f"{ref.provider}:{ref.model}", **options)
        return self.made[label]

    def get(self, label: str) -> CaseFake:
        return self.made[label]
