"""t10: teacher-label stage, train only, resumable, with provenance (fake server)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.distill.systemone import SystemOneClient
from jev_factory.distill.teacher import TeacherConfig, label_dataset, label_records


def _record(rid, split="train"):
    qs = {
        "a": Question("choice", (("fix", "f"), ("wait", "w"))),
        "b": Question("noul", (("false", ""), ("true", ""))),
    }
    gold = {"a": Gold(label="fix"), "b": Gold(label="true", probability_true=1.0)}
    return Record(rid, f"state {rid}", qs, gold, split, "src", "MIT", "human")


class _Server:
    def __init__(self, break_for=()):
        outer = self
        self.bodies = []
        self.break_for = set(break_for)

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                ans = {}
                for qid, spec in body["questions"].items():
                    if spec["type"] == "noul":
                        ans[qid] = {"noul": 0.3}
                    else:
                        ans[qid] = {"probabilities": {"fix": 0.7, "wait": 0.3}}
                if body["state"] in outer.break_for:
                    ans.pop("b", None)
                raw = json.dumps({"answers": ans}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def make():
    servers = []

    def _make(**kw):
        s = _Server(**kw)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        s.httpd.shutdown()
        s.httpd.server_close()


def _ids(recs):
    return {r.id for r in recs}


def _cfg(url, **kw):
    return TeacherConfig(
        base_url=url,
        model="clef",
        repo_id="org/clef-27b",
        revision="abc123",
        build_id="build-1",
        engine="sglang",
        **kw,
    )


def _rows(tmp_path):
    lines = (tmp_path / "teacher_labels.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_labels_carry_full_provenance(make, tmp_path):
    srv = make()
    man = label_records(
        [_record("r1"), _record("r2")],
        tmp_path,
        _cfg(srv.url),
        allowed_ids=_ids([_record("r1"), _record("r2")]),
    )
    rows = _rows(tmp_path)
    assert {(r["record_id"], r["question_id"]) for r in rows} == {
        (i, q) for i in ("r1", "r2") for q in "ab"
    }
    r = next(x for x in rows if x["question_id"] == "a")
    assert r["distribution"] == {"fix": 0.7, "wait": 0.3}
    assert next(x for x in rows if x["question_id"] == "b")["p_true"] == 0.3
    for r in rows:
        assert r["teacher_repo_id"] == "org/clef-27b" and r["teacher_revision"] == "abc123"
        assert r["teacher_build_id"] == "build-1" and r["engine"] == "sglang"
        assert len(r["response_sha256"]) == 64 and r["request_config"]["body_sha256"]
        assert r["mode"] == "joint"
    assert man["requests_sent"] == 2 and man["mode"] == "joint"
    assert all("gold" not in b for b in srv.bodies)


def test_non_train_refused_before_any_request(make, tmp_path):
    srv = make()
    with pytest.raises(CliError):
        label_records(
            [_record("r1"), _record("r2", "test")],
            tmp_path,
            _cfg(srv.url),
            allowed_ids=_ids([_record("r1"), _record("r2", "test")]),
        )
    assert srv.bodies == []


def test_incomplete_answer_is_error_not_label(make, tmp_path):
    srv = make(break_for={"state r1"})
    man = label_records([_record("r1")], tmp_path, _cfg(srv.url), allowed_ids=_ids([_record("r1")]))
    rows = {r["question_id"]: r for r in _rows(tmp_path)}
    assert rows["b"]["teacher_error"] and rows["b"]["p_true"] is None
    assert rows["a"]["teacher_error"] is None
    assert man["teacher_errors"] == 1
    # errors are not cached: a re-run asks again
    label_records([_record("r1")], tmp_path, _cfg(srv.url), allowed_ids=_ids([_record("r1")]))
    assert len(srv.bodies) == 2


def test_transport_failure_is_error(tmp_path):
    cfg = _cfg("http://127.0.0.1:1")
    man = label_records([_record("r1")], tmp_path, cfg, allowed_ids=_ids([_record("r1")]))
    assert man["labelled"] == 0 and man["teacher_errors"] == 2


def test_resume_after_kill_resends_only_unanswered(make, tmp_path):
    srv = make()
    cfg = _cfg(srv.url)
    real = SystemOneClient(cfg.base_url, model=cfg.model)

    class Dies:
        def __init__(self):
            self.n = 0

        def ask(self, rec):
            if self.n == 2:
                raise KeyboardInterrupt
            self.n += 1
            return real.ask(rec)

    recs = [_record(f"r{i}") for i in range(4)]
    with pytest.raises(KeyboardInterrupt):
        label_records(recs, tmp_path, cfg, client=Dies(), allowed_ids=_ids(recs))
    assert len(srv.bodies) == 2
    man = label_records(recs, tmp_path, cfg, allowed_ids=_ids(recs))
    assert len(srv.bodies) == 4  # only the two unanswered were sent
    assert man["cache_hits"] == 2 and man["requests_sent"] == 2
    assert sum(1 for r in _rows(tmp_path) if r["cached"]) == 4


def test_mode_recorded_and_per_question_sends_one_each(make, tmp_path):
    srv = make()
    man = label_records(
        [_record("r1")], tmp_path, _cfg(srv.url, joint=False), allowed_ids=_ids([_record("r1")])
    )
    assert len(srv.bodies) == 2 and all(len(b["questions"]) == 1 for b in srv.bodies)
    assert man["mode"] == "per-question" and man["joint"] is False
    assert {r["mode"] for r in _rows(tmp_path)} == {"per-question"}


def test_fit_selection_heldout_records_refused_with_no_request(make, tmp_path):
    srv = make()
    train_side = {"r1"}
    # r2 has source split 'train' but is not on the dataset's train side
    with pytest.raises(CliError, match="train side"):
        label_records(
            [_record("r1"), _record("r2")], tmp_path, _cfg(srv.url), allowed_ids=train_side
        )
    assert srv.bodies == []


def test_allowed_ids_is_required(make, tmp_path):
    srv = make()
    with pytest.raises(TypeError):
        label_records([_record("r1")], tmp_path, _cfg(srv.url))  # type: ignore[call-arg]
    assert srv.bodies == []


def test_label_dataset_reads_train_side_only(make, tmp_path):
    from jev_factory.distill import dataset

    srv = make()
    recs = [_record(f"r{i}") for i in range(6)]
    (tmp_path / "ds").mkdir()
    ds_dir = tmp_path / "ds"
    (ds_dir / "train.jsonl").write_text(
        "".join(json.dumps(r.to_json()) + "\n" for r in recs[:3]), encoding="utf-8"
    )
    for side in ("fit", "selection", "held_out", "test"):
        (ds_dir / f"{side}.jsonl").write_text(
            "".join(json.dumps(r.to_json()) + "\n" for r in recs[3:]), encoding="utf-8"
        )
    man = label_dataset(ds_dir, tmp_path / "out", _cfg(srv.url))
    assert man["records"] == 3 and man["requests_sent"] == 3
    assert len(srv.bodies) == 3
    assert dataset.train_ids(ds_dir) == frozenset({"r0", "r1", "r2"})
