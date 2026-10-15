"""End-to-end toy run of every ``jev distill`` stage (t25), CPU-only, on fakes.

A tiny typed-question run is built from the real rows in ``tests/fixtures/typed_decisions``
(cloned under fresh ids, workflows and state) plus synthetic rows
(``tests/fixtures/distill_e2e``). Every one of the 14 stages is first dry-run through the CLI
(nothing is written), then run with ``--apply`` through the CLI, against stdlib fakes for the
teacher, the correction drafter, the token counter, the benchmark and final servers, the
trainer, the scorer, the format builders, the measurer and the hub. Torch, transformers and
strands_decider are made un-importable and sockets refuse to connect for the whole run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.cli import main
from jev_factory.cli._commands import distill as distill_cmd
from jev_factory.distill import dataset as ddataset
from jev_factory.distill import final as dfinal
from jev_factory.distill import pins as dpins
from jev_factory.distill import pipeline as P
from jev_factory.distill import prereg as dprereg
from jev_factory.distill import quant as dquant
from jev_factory.distill import records as R
from jev_factory.distill import report as dreport
from jev_factory.distill.corrections import read_corrections
from jev_factory.distill.systemone import CallResult, QuestionResult
from jev_factory.factory import gpu
from jev_factory.factory.stages import read_manifest
from jev_factory.measure import once
from jev_factory.review import corrections as review
from tests.release_support import TOKEN, FakeHub

FIX = Path(__file__).parent / "fixtures"
STAGES = list(P.STAGE_ORDER)
REV = "a" * 40
ST = (2).to_bytes(8, "little") + b"{}"  # a valid empty safetensors header
NEUTRAL = {
    "head_type": "pointer",
    "base_model": "Qwen/Qwen3.5-2B",
    "temperature": 1.0,
    "temperature_by_kind": {},
}
WEAK_SOURCE = f"{R.JEV_BENCH_SOURCE}:go_emotions"
EXCLUDED_SOURCE = f"{R.JEV_BENCH_SOURCE}:banking77"
MODEL_REPO = "example-org/distill-e2e-student"
DATASET_REPO = "example-org/distill-e2e-data"
KNOBS = {"audit": {"floor": 0.6}, "corrections-draft": {"author": "model:fake-drafter"}}
UPLOAD_KNOBS = {"model_repo": MODEL_REPO, "dataset_repo": DATASET_REPO}
NOW = "2026-10-14"


def _fx(name: str) -> dict:
    return json.loads((FIX / "typed_decisions" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ the records


def _typed_row(template: dict, i: int, split: str) -> dict:
    row = copy.deepcopy(template)
    state = json.loads(row["state"])
    state["uid"] = f"UID-typed-{i:04d}"
    state["notes"] = _words(i)
    row.update(
        id=f"typed_{i:04d}",
        workflow=f"wf_{i:04d}",  # one workflow per row: the grouped split can spread them
        split=split,
        state=json.dumps(state, sort_keys=True),
    )
    return row


_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def _words(i: int) -> str:
    """Ten distinct pseudo-words per row, so no two rows are near-duplicates of each other."""
    rng = random.Random(i)
    return " ".join(f"{rng.choice(_WORDS)}{rng.randrange(10**6)}" for _ in range(10))


def _fit_slots(row: dict, slots: int = 24) -> dict:
    """Trim a real row's question to the student's 24 slots (the real go_emotions has 28)."""
    q = row["question"]
    q = json.loads(q) if isinstance(q, str) else dict(q)
    crit = q["criteria"]
    if isinstance(crit, dict):
        names = list(crit)[:slots]
        q["criteria"] = {k: crit[k] for k in names}
    else:
        names = crit[:slots]
        q["criteria"] = names
    row["question"] = json.dumps(q)
    soft = row.get("soft_label")
    if soft:
        soft = json.loads(soft) if isinstance(soft, str) else dict(soft)
        row["soft_label"] = json.dumps({k: v for k, v in soft.items() if k in names})
    return row


def _bench_row(template: dict, name: str, i: int, split: str) -> dict:
    row = copy.deepcopy(template)
    row.update(
        id=f"{name}/{split}/{i}",
        split=split,
        state=json.dumps(f"UID-{name}-{i:04d} {_words(i)}"),
    )
    return _fit_slots(row) if name == "go_emotions" else row


def make_records() -> list[R.Record]:
    """~100 records: three sources that pass the licence filter or not, test side included."""
    real = _fx("real_typed_decisions_row.json")
    synthetic = json.loads((FIX / "distill_e2e" / "synthetic_typed_rows.json").read_text())["rows"]
    templates = [real, *synthetic]
    rows = [_typed_row(templates[i % 3], i, "train") for i in range(36)]
    rows += [_typed_row(templates[i % 3], 100 + i, "test") for i in range(6)]
    out = [R.from_typed_decisions_row(r) for r in rows]
    emotions = _fx("real_jev_bench_go_emotions_row.json")
    out += [
        R.from_jev_bench_row(_bench_row(emotions, "go_emotions", i, "train")) for i in range(36)
    ]
    out += [
        R.from_jev_bench_row(_bench_row(emotions, "go_emotions", 100 + i, "test")) for i in range(4)
    ]
    banking = _fx("real_jev_bench_banking77_row.json")  # licence is a compound string: excluded
    out += [R.from_jev_bench_row(_bench_row(banking, "banking77", i, "train")) for i in range(3)]
    assert {r.source for r in out} == {R.TYPED_DECISIONS_SOURCE, WEAK_SOURCE, EXCLUDED_SOURCE}
    return out


def uid_of(record: R.Record) -> str:
    return next(tok for tok in record.state.replace('"', " ").split() if tok.startswith("UID-"))


# ------------------------------------------------------------------------------ the fakes


def _right(name: str, rec_id: str, qid: str, accuracy: float) -> bool:
    digest = hashlib.sha256(f"{name}|{rec_id}|{qid}".encode()).digest()
    return digest[0] / 256.0 < accuracy


def _rank_right(k: int, accuracy: float) -> bool:
    """Exactly *accuracy* of a fold's records are right, whatever the fold size (k % 20)."""
    return (k % 20) < round(accuracy * 20)


def _answer(rec: R.Record, qid: str, right: bool, conf: float) -> tuple[str, dict | None, float]:
    """``(kind, distribution, p_true)``: gold-confident when *right*, else on a wrong option."""
    q, gold = rec.questions[qid], rec.gold[qid]
    if q.type == "noul":
        truth = gold.probability_true >= 0.5
        return "noul", None, conf if truth == right else 1.0 - conf
    names = q.names
    g = names.index(gold.label)
    target = g if right else (g + 1) % len(names)
    rest = (1.0 - conf) / (len(names) - 1)
    return q.type, {n: (conf if k == target else rest) for k, n in enumerate(names)}, 0.0


class GoldClient:
    """A SystemOne-shaped fake server: right on ``accuracy`` of the questions, never on *wrong*."""

    def __init__(self, name, records, accuracy, conf=0.88, wrong_sources=()):
        self.name, self.accuracy, self.conf = name, accuracy, conf
        self.by_id = {r.id: r for r in records}
        self.wrong_sources = set(wrong_sources)
        self.asked: list[str] = []

    def ask(self, wire):
        rec = self.by_id[wire["id"]]
        self.asked.append(rec.id)
        results = []
        for qid in wire["questions"]:
            right = rec.source not in self.wrong_sources and _right(
                self.name, rec.id, qid, self.accuracy
            )
            kind, dist, p_true = _answer(rec, qid, right, self.conf)
            if kind == "noul":
                results.append(QuestionResult(qid, "noul", p_true=p_true))
            else:
                results.append(QuestionResult(qid, kind, distribution=dist))
        return CallResult(rec.id, self.name, {"url": "fake"}, "sha", results)


class FakeScorer:
    """Train-stage scorer: the alpha 0.9 candidate is the better one."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, candidate, checkpoint, fold_records):
        self.calls.append(candidate.cid)
        accuracy = 0.9 if candidate.knobs["alpha"] >= 0.9 else 0.6
        out = {}
        for fold, recs in fold_records.items():
            out[fold] = [
                {
                    "fold": fold,
                    "record_id": rec.id,
                    "error": None,
                    "questions": {
                        qid: dict(
                            zip(
                                ("kind", "distribution", "p_true"),
                                _answer(rec, qid, _rank_right(k, accuracy), 0.9),
                            ),
                            error=None,
                        )
                        for qid in rec.questions
                    },
                }
                for k, rec in enumerate(recs)
            ]
        return out


#: format -> (selection accuracy, peak MiB, median s)
PROFILE = {"bf16": (0.9, 5000.0, 0.05), "torch_int8_wo": (0.9, 3000.0, 0.04)}


class FakeMeasurer:
    """Quant/gate measurer: serves a format's artifact as a fake that answers from the profile."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, fmt, artifact, fold_records):
        self.calls.append((fmt, Path(artifact).name))
        accuracy, peak, median = PROFILE[fmt]
        preds = {}
        for fold, recs in fold_records.items():
            preds[fold] = [
                {
                    "fold": fold,
                    "record_id": rec.id,
                    "error": None,
                    "questions": {
                        qid: dict(
                            zip(
                                ("kind", "distribution", "p_true"),
                                _answer(rec, qid, _rank_right(k, accuracy), 0.9),
                            ),
                            error=None,
                        )
                        for qid in rec.questions
                    },
                }
                for k, rec in enumerate(recs)
            ]
        return {
            "predictions": preds,
            "latency": {"n": 40, "median_s": median, "p95_s": median, "max_s": median},
            "memory": {"probe": "fake", "peak_mib": peak, "measured": True},
        }


def write_checkpoint(out: Path) -> None:
    """What the distill trainer leaves behind: a raw-readout hobson-hf-export source."""
    (out / "lora").mkdir(parents=True, exist_ok=True)
    (out / player.ADAPTER_CONFIG).write_text('{"r": 16}')
    (out / player.ADAPTER).write_bytes(ST + b"adapter")
    (out / player.HEAD).write_bytes(ST + b"head")
    (out / player.CONFIG).write_text(json.dumps(NEUTRAL))
    for name in player.TOKENIZER_FILES:
        (out / name).write_text("{}")


def fake_int8_builder(fmt, checkpoint, out_dir):
    """A fake torchao int8 weight-only builder: a manifest-verified artifact, no torch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / qz.TORSO_FILE).write_bytes(b"int8-torso")
    (out_dir / player.HEAD).write_bytes(ST + b"head")
    (out_dir / player.CONFIG).write_text(json.dumps(NEUTRAL))
    for name in player.TOKENIZER_FILES:
        (out_dir / name).write_text("{}")
    qz.write_manifest(
        out_dir,
        bits=8,
        torchao_api=qz.TORCHAO_APIS[8][0],
        torchao_version="0.18.0",
        torch_version="2.12.1",
        source={"checkpoint": str(checkpoint), "sha256": {player.HEAD: "0" * 64}},
        base_model="Qwen/Qwen3.5-2B",
        base_model_revision=REV,
    )
    return out_dir


# ------------------------------------------------------------------------------ the run


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def small_prereg() -> dict:
    doc = dprereg.example_doc()
    doc["knob_grid"].update(
        {
            "alpha": [0.5, 0.9],
            "temperature": {"choice": [1.0], "score": [1.0], "noul": [1.0]},
            "teacher_filter": ["all"],
            "teacher_weak_floor": [0.6],
            "data_mix": ["teacher_plus_corrections"],
            "epochs": [1],
            "lr": [1e-4],
            "init_mode": ["continue"],
        }
    )
    doc["heal"]["lr"] = 1e-4  # the heal knobs must be grid values; the toy grid has one lr
    return doc


class Env:
    """Everything the e2e run shares: the work dir, the fakes and the CLI driver."""

    def __init__(self, tmp_path: Path, capsys, monkeypatch):
        self.tmp = tmp_path
        self.capsys = capsys
        self.records = make_records()
        self.work = tmp_path / "run"
        (self.work / "distill").mkdir(parents=True)
        self.log: list[str] = []
        self.gpu_calls: list[list[str]] = []
        records_file = tmp_path / "records.jsonl"
        records_file.write_text(
            "".join(json.dumps(r.to_json(), sort_keys=True) + "\n" for r in self.records), "utf-8"
        )
        cache = self._pin_cache(tmp_path / "cache", monkeypatch)
        probe, side = tmp_path / "probe.jsonl", tmp_path / "side.txt"  # operator-held, t16
        probe.write_text(json.dumps({"id": "probe-only", "state": "PROBE"}) + "\n", "utf-8")
        side.write_text("zz-side-model\n", "utf-8")
        (self.work / "run.toml").write_text(
            "\n".join(
                [
                    f"work = {json.dumps(str(self.work))}",
                    'base = "example-org/base"',
                    f'base_rev = "{"0" * 40}"',
                    'hub_prefix = "example-org/distill-"',
                    'licence = "Apache-2.0"',
                    f"distill_records = {json.dumps(str(records_file))}",
                    f"distill_pin_cache = {json.dumps(str(cache))}",
                    f"distill_probe_set = {json.dumps(str(probe))}",
                    f"distill_side_model_list = {json.dumps(str(side))}",
                    f'distill_decider_revision = "{REV}"',
                    f"distill_decider_repo = {json.dumps(str(tmp_path / 'strands-decider'))}",
                    'distill_trainer_python = "python3"',
                    'train_memory_max = "100G"',
                    'distill_teacher_url = "http://127.0.0.1:9"',
                    'distill_teacher_model = "clef"',
                    'distill_teacher_repo = "example-org/clef-27b"',
                    'distill_teacher_revision = "abc123"',
                    'distill_teacher_build_id = "clef-b1"',
                    'distill_teacher_engine = "sglang"',
                    'distill_alt_teacher_url = "http://127.0.0.1:9"',
                    'distill_alt_teacher_model = "fake-drafter"',
                    'distill_stock_url = "http://127.0.0.1:9"',
                    'distill_stock_model = "v21"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (self.work / dprereg.PREREG_REL).write_text(json.dumps(small_prereg()), encoding="utf-8")

        self.teacher = GoldClient("clef", self.records, 1.0, 0.85, wrong_sources={WEAK_SOURCE})
        self.measurer = FakeMeasurer()
        self.scorer = FakeScorer()
        self.hub = FakeHub(tmp_path / "remote")
        self.final_clients = {
            "clef": GoldClient("clef", self.records, 0.95),
            "stock": GoldClient("stock", self.records, 0.7),
            "student": GoldClient("student", self.records, 0.85),
        }
        self.services = P.Services(
            token_counter=lambda text: len(text.split()),
            teacher_client=self.teacher,
            drafter=lambda rec: {
                qid: {"label": g.label} for qid, g in rec.gold.items() if g.label is not None
            },
            benchmark_targets={n: self._target(n) for n in ("clef", "stock")},
            benchmark_clients={
                "clef": GoldClient("clef", self.records, 0.95),
                "stock": GoldClient("stock", self.records, 0.7),
            },
            memory_probe=lambda: None,
            scorer=self.scorer,
            builders={"bf16": dquant.bf16_builder(dpins.DEFAULT_BASE_REVISION)}
            | {"torch_int8_wo": fake_int8_builder},
            measurer=self.measurer,
            final_models={
                n: dfinal.FinalModel(target=self._target(n), client=c)
                for n, c in self.final_clients.items()
            },
            hub=self.hub,
            environ={"HF_TOKEN": TOKEN},
            today=lambda: NOW,
        )
        orig = distill_cmd._context
        monkeypatch.setattr(
            distill_cmd, "_context", lambda args: replace(orig(args), services=self.services)
        )
        monkeypatch.setattr(gpu, "run_gpu_stage", self._fake_gpu)

    def _target(self, name):
        from jev_factory.distill import benchmark as bench

        return bench.Target(
            name,
            "http://127.0.0.1:9",
            name,
            start=lambda: self.log.append(f"start {name}"),
            stop=lambda: self.log.append(f"stop {name}"),
        )

    def _pin_cache(self, root: Path, monkeypatch) -> Path:
        base = root / dpins.cache_dirname(dpins.BASE_REPO)
        base.mkdir(parents=True)
        (base / dpins.REVISION_FILE).write_text(dpins.DEFAULT_BASE_REVISION + "\n")
        dec = root / dpins.cache_dirname(dpins.DECIDER_REPO)
        (dec / "lora").mkdir(parents=True)
        (dec / dpins.REVISION_FILE).write_text(REV + "\n")
        (dec / dpins.ADAPTER_FILE).write_bytes(b"adapter")
        (dec / dpins.HEAD_FILE).write_bytes(b"head")
        (dec / dpins.PROVENANCE_FILE).write_text(
            json.dumps({"adapter_sha256": _sha(b"adapter"), "head_sha256": _sha(b"head")})
        )
        # The real v21 digests are of files we cannot ship; the toy cache holds toy bytes, so the
        # pins under test are the toy digests. Everything else in verify_pins runs for real.
        monkeypatch.setattr(dpins, "DECIDER_ADAPTER_SHA256", _sha(b"adapter"))
        monkeypatch.setattr(dpins, "DECIDER_HEAD_SHA256", _sha(b"head"))
        return root

    def _fake_gpu(self, stage, run_dir, cmd, **kw):
        self.gpu_calls.append(list(cmd))
        write_checkpoint(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    # -- the CLI driver

    def cli(self, stage: str, *flags: str, knobs: dict | None = None):
        self.capsys.readouterr()
        argv = ["distill", stage, "--work", str(self.work), *flags]
        for key, value in {**KNOBS.get(stage, {}), **(knobs or {})}.items():
            argv += ["--knob", f"{key}={value}"]
        rc = main(argv)
        out = self.capsys.readouterr()
        return rc, out.out, out.err

    def dry_run(self, stage: str, knobs: dict | None = None) -> dict:
        before = tree(self.tmp)
        rc, out, err = self.cli(stage, "--json", knobs=knobs)
        assert rc == 0, (stage, err)
        plan = json.loads(out)
        assert plan["applied"] is False
        assert plan["stage"] == stage
        assert plan["would_run"] is True
        assert tree(self.tmp) == before, f"the {stage} dry run changed files"
        rc, out, err = self.cli(stage, knobs=knobs)
        assert rc == 0, (stage, err)
        assert out.startswith(f"dry run: distill stage {stage}")
        assert tree(self.tmp) == before
        return plan

    def apply(self, stage: str, knobs: dict | None = None) -> dict:
        rc, out, err = self.cli(stage, "--apply", "--json", knobs=knobs)
        assert rc == 0, (stage, err)
        return json.loads(out)

    @property
    def base(self) -> Path:
        return self.work / "distill"

    def read(self, rel: str):
        return json.loads((self.base / rel).read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path, capsys, monkeypatch):
    # CPU-only and offline: the heavy modules cannot be imported and no socket connects.
    for name in ("torch", "transformers", "strands_decider"):
        monkeypatch.setitem(sys.modules, name, None)

    def refuse(*args, **kwargs):
        raise AssertionError("the e2e run opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return Env(tmp_path, capsys, monkeypatch)


def approve_corrections(env: Env) -> list:
    """The operator's side: approve every drafted correction (``jev review --corrections``)."""
    path = env.base / P.CORRECTIONS_FILE
    proposed = read_corrections(path)
    for c in proposed:
        decision = {"action": "approve", "record_id": c.record_id, "question_id": c.question_id}
        decision["reviewer"] = "operator"
        record = review.make_record(proposed, decision, at="2026-10-14T00:00:00Z")
        review.append_record(review.review_path(path), record)
    return proposed


# ------------------------------------------------------------------------------ the test


def test_every_stage_dry_runs_then_applies_end_to_end_into_a_complete_bundle(env):
    # 1. a dry run of every stage, on the fresh run: plans only, nothing written
    for stage in STAGES:
        knobs = UPLOAD_KNOBS if stage == "upload" else None
        plan = env.dry_run(stage, knobs)
        assert plan["deps"] == list(P.REGISTRY.get(stage).deps)
    assert not (env.base / "manifests").exists()

    # 2. --apply: the first seven stages, then the operator's approval of the drafted corrections
    for stage in STAGES[: STAGES.index("corrections-draft") + 1]:
        assert env.apply(stage)["status"] == "complete"
    proposed = approve_corrections(env)
    assert proposed  # the file never self-approves
    assert {c.status for c in proposed} == {"proposed"}
    for stage in ("assemble", "benchmark", "train", "quantize", "gate", "final", "bundle"):
        assert env.apply(stage)["status"] == "complete", stage
    assert env.apply("upload", UPLOAD_KNOBS)["status"] == "complete"

    for stage in STAGES:
        manifest = read_manifest(env.base, stage)
        assert manifest["status"] == "complete"
        assert manifest["stage"] == stage

    check_early_stages(env)
    check_training_chain(env)
    check_final_and_bundle(env)
    check_upload(env)
    check_final_is_once(env)


# ------------------------------------------------------------------------------ the checks


def check_early_stages(env: Env) -> None:
    ds = env.base / "dataset"
    manifest = env.read("dataset/manifest.json")
    counts = manifest["counts"]
    assert counts["licence_excluded"] == 3  # banking77's compound licence
    assert min(counts["train"], counts["fit"], counts["selection"], counts["held_out"]) > 0
    assert counts["test"] == 10
    assert (env.base / "prereg.lock.json").is_file()
    assert env.read("contamination.json")
    train_ids = {r.id for r in ddataset.read_records(ds / "train.jsonl")}
    assert set(env.teacher.asked) == train_ids  # only the train side is ever labelled
    table = env.read("teacher_audit.json")
    assert table["teacher_weak_sources"] == [WEAK_SOURCE]  # the model-gold source is never flagged
    drafted = env.read("corrections/draft.json")
    assert drafted["proposed"] > 0
    assert drafted["author"] == "model:fake-drafter"
    frozen = env.read("assemble/assemble_manifest.json")
    assert frozen["status"] == "frozen"
    assert frozen["counts"]["trainable_corrections"] > 0  # the operator's approvals took effect
    rows = [json.loads(x) for x in (env.base / P.TRAIN_SET).read_text().splitlines() if x.strip()]
    assert {r["target_source"] for r in rows} == {"clef+gold", "correction"}
    bench = env.read("benchmark/benchmark.json")
    assert set(bench["models"]) == {"clef", "stock"}
    assert set(bench["folds"]) == {"fit", "selection"}


def check_training_chain(env: Env) -> None:
    summary = env.read("select/summary.json")
    assert len(summary["candidates"]) == 2
    assert summary["winner"] is not None
    assert env.scorer.calls == ["c000", "c001"]
    assert len(env.gpu_calls) == 2  # one guarded trainer run per candidate
    assert summary["fold"] == "selection"
    assert set(summary["fold_ids"]) == {"fit", "selection"}
    quant = env.read("quant/quant_manifest.json")
    decision = env.read("quant/format_decision.json")
    assert quant["chosen_format"] == decision["format"] == "torch_int8_wo"
    assert quant["heal_count"] == 0
    # both builders ran (bf16 is the reference every format is measured against)
    assert {fmt for fmt, _ in env.measurer.calls} >= {"bf16", "torch_int8_wo"}
    calibration = env.read("quant/calibration.json")
    assert set(calibration["temperature_by_kind"]) >= {"choice"}
    gate = env.read("gate/gate.json")
    assert gate["fitted"] is True
    assert set(gate["thresholds_by_kind"]) == {"choice", "noul", "score"}
    assert gate["calibration_sha256"] == _sha((env.base / P.QUANT_CALIBRATION).read_bytes())


def check_final_and_bundle(env: Env) -> None:
    ledger = once.OnceLedger(env.work).records()
    assert sorted((r.side, r.status) for r in ledger) == [
        (once.HELD_OUT, once.MEASURED),
        (once.TEST, once.MEASURED),
    ]
    for client in env.final_clients.values():  # every sealed record, scored once per model
        assert (
            len(client.asked)
            == len(set(client.asked))
            == 10 + env.read("dataset/manifest.json")["counts"]["held_out"]
        )
    report = env.read(P.FINAL_REPORT)
    assert report["c42"]["bars"]
    assert report["c43"]["bars"]
    assert report["result"] in (dreport.PASSED, dreport.MISSED)
    assert report["decision_gate"]["fitted"] is True

    model = env.base / P.BUNDLE_MODEL
    data = env.base / P.BUNDLE_DATASET
    names = {p.relative_to(model).as_posix() for p in model.rglob("*") if p.is_file()}
    assert {
        "calibration.json",
        "gate.json",
        "scorer-train.json",
        "report.json",
        "distill_provenance.json",
        "assemble_manifest.json",
        "bundle.json",
        qz.MANIFEST_FILE,
        qz.TORSO_FILE,
        player.CONFIG,
        player.HEAD,
    } <= names
    gate = json.loads((model / "gate.json").read_text())
    assert gate["fitted"] is True
    assert set(gate["thresholds_by_kind"]) == {"choice", "noul", "score"}
    assert (model / "calibration.json").read_bytes() == (
        env.base / P.QUANT_CALIBRATION
    ).read_bytes()
    served = json.loads((model / player.CONFIG).read_text())
    assert served["temperature"] == 1.0
    assert served["temperature_by_kind"]
    index = json.loads((model / "scorer-train.json").read_text())
    assert index["rows"] == len(index["entries"]) > 0
    assert set(index["target_sources"]) == {"clef+gold", "correction"}  # target sources, per row
    assert all(e["target_source"] and e["licence"] for e in index["entries"])
    assert "state" not in index["entries"][0]  # an index, never text
    shipped = json.loads((model / "report.json").read_text())
    assert shipped["c42"]["bars"]
    assert shipped["c43"]["bars"]
    assert "decision_gate" in shipped
    prov = json.loads((model / "distill_provenance.json").read_text())
    assert prov["held_out_published"] is False
    assert prov["test_published"] is False
    assert (
        prov["training_set_sha256"]
        == env.read("assemble/assemble_manifest.json")["output"]["sha256"]
    )
    assert prov["prereg_sha256"] == dprereg.sha256_file(env.work / dprereg.PREREG_REL)
    assert {"train.jsonl", "dataset.json", "gate.json", "calibration.json"} <= {
        p.name for p in data.iterdir()
    }
    kept = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    assert kept
    assert {r["licence"].lower() for r in kept} <= {"apache-2.0", "mit"}

    # no sealed text anywhere in either bundle, nor in what the hub received
    sealed = {
        r["id"]
        for side in ("held_out.jsonl", "test.jsonl")
        for r in map(json.loads, (env.base / "dataset" / side).read_text().splitlines())
    }
    by_id = {r.id: r for r in env.records}
    assert len(sealed) == 10 + env.read("dataset/manifest.json")["counts"]["held_out"]
    blobs = [p.read_bytes() for root in (model, data) for p in root.rglob("*") if p.is_file()]
    blobs += [p.read_bytes() for p in (env.tmp / "remote").rglob("*") if p.is_file()]
    for rid in sealed:
        for text in (uid_of(by_id[rid]), json.dumps(rid)):  # the quoted JSON id
            assert not any(text.encode() in blob for blob in blobs), f"sealed {text} leaked"
    trained = {uid_of(by_id[r["record_id"]]) for r in kept}
    assert trained
    assert all(any(u.encode() in blob for blob in blobs) for u in trained)


def check_upload(env: Env) -> None:
    result = env.read(P.UPLOAD_FILE)
    assert result["applied"] is True
    assert result["model"]["private"] is True
    assert result["dataset"]["private"] is True
    calls = [c[0] for c in env.hub.calls]
    assert calls.count("create_repo") == 2
    assert calls.count("snapshot_download") == 2
    creates = [c for c in env.hub.calls if c[0] == "create_repo"]
    assert all(c[2]["private"] is True for c in creates)  # private-first, never public
    assert [c[1] for c in creates] == [MODEL_REPO, DATASET_REPO]
    assert not any("public" in c.lower() for c in calls)
    for repo in (MODEL_REPO, DATASET_REPO):  # fetched back and byte-identical
        remote = env.tmp / "remote" / repo
        local = env.base / (P.BUNDLE_MODEL if repo == MODEL_REPO else P.BUNDLE_DATASET)
        for p in local.rglob("*"):
            if p.is_file():
                assert (remote / p.relative_to(local)).read_bytes() == p.read_bytes()


def check_final_is_once(env: Env) -> None:
    asked = {n: len(c.asked) for n, c in env.final_clients.items()}
    ledger = once.OnceLedger(env.work).records()
    rc, out, err = env.cli("final", "--apply", "--json")  # fresh: skipped, nothing is rerun
    assert rc == 0
    assert json.loads(out)["skipped"] is True
    rc, out, err = env.cli("final", "--apply", "--json", "--force")  # a second final: refused
    assert rc != 0
    assert "deviation" in err
    assert once.OnceLedger(env.work).records() == ledger
    assert {n: len(c.asked) for n, c in env.final_clients.items()} == asked  # nothing scored
    assert read_manifest(env.base, "final")["status"] == "failed"  # recorded, not smoothed over


def test_the_cli_lists_all_fourteen_stages_and_status_sees_the_run(env):
    env.capsys.readouterr()
    assert main(["distill", "--json"]) == 0
    listed = json.loads(env.capsys.readouterr().out)["stages"]
    assert [s["stage"] for s in listed] == STAGES
    assert len(STAGES) == 14
    env.apply("preregister")
    assert main(["status", str(env.work), "--json"]) == 0
    rows = {s["stage"]: s for s in json.loads(env.capsys.readouterr().out)["distill_stages"]}
    assert list(rows) == STAGES
    assert rows["preregister"]["status"] == "complete"
    assert rows["upload"]["status"] == "not-run"
