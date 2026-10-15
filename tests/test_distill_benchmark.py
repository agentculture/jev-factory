"""t16: zero-shot benchmark of two models on fit/selection folds, using two fake servers."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import benchmark as bm
from jev_factory.distill import dataset
from jev_factory.distill.records import Gold, Question, Record
from tests import test_distill_train as tt
from tests.test_distill_train import cache as pin_cache  # noqa: F401 - guard-test fixture


def _rec(i: int, split: str = "train") -> Record:
    return Record(
        id=f"r{i}",
        state=f"state {i}",
        questions={
            "c": Question("choice", [("fix", "fix it"), ("wait", "wait")]),
            "s": Question("score", [("0", "low"), ("1", "mid"), ("2", "high")]),
            "n": Question("noul", [("false", ""), ("true", "")]),
        },
        gold={
            "c": Gold(label="fix"),
            "s": Gold(label="1"),
            "n": Gold(label="true", probability_true=1.0),
        },
        split=split,
        source="src",
        licence="mit",
        gold_provenance="human",
    )


class _FakeServer:
    """A fake /v1/systemone server with a fixed delay, logging start/stop order."""

    def __init__(self, name, delay, log, p_fix=0.7):
        outer = self
        self.name, self.log, self.requests = name, log, []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                time.sleep(delay)
                answers = {
                    "c": {"probabilities": {"fix": p_fix, "wait": 1 - p_fix}},
                    "s": {"probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
                    "n": {"noul": 0.9},
                }
                raw = json.dumps({"answers": answers}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def target(self):
        return bm.Target(
            self.name,
            self.url,
            self.name,
            start=lambda: self.log.append(f"start {self.name}"),
            stop=lambda: self.log.append(f"stop {self.name}"),
        )

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers():
    made, log = [], []

    def make(name, delay=0.0, **kw):
        srv = _FakeServer(name, delay, log, **kw)
        made.append(srv)
        return srv

    make.log = log
    yield make
    for srv in made:
        srv.close()


FOLDS = {"fit": [_rec(1), _rec(2), _rec(3)], "selection": [_rec(4), _rec(5)]}


def test_scores_both_models_into_prediction_records_with_latency_and_memory(servers):
    clef = servers("clef", 0.0)
    v21 = servers("v21", 0.0, p_fix=0.55)
    out = bm.run_benchmark([clef.target(), v21.target()], FOLDS, memory_probe=lambda: 300.0)
    assert out["folds"] == {"fit": 3, "selection": 2}
    assert set(out["models"]) == {"clef", "v21"}
    for name, srv in (("clef", clef), ("v21", v21)):
        m = out["models"][name]
        assert m["n_records"] == 5
        assert m["n_failed"] == 0
        assert len(srv.requests) == 5
        assert m["latency"]["n"] == 5
        assert m["latency"]["median_s"] is not None
        assert m["latency"]["p95_s"] is not None
        assert m["latency"]["p95_s"] >= m["latency"]["median_s"]
        assert m["memory"]["measured"]
        assert m["memory"]["peak_mib"] >= 300.0
    row = out["models"]["v21"]["predictions"][0]
    assert row["model"] == "v21"
    assert row["fold"] == "fit"
    assert row["record_id"] == "r1"
    assert row["questions"]["c"]["distribution"] == {"fix": 0.55, "wait": pytest.approx(0.45)}
    assert row["questions"]["n"]["p_true"] == 0.9
    assert row["latency_s"] >= 0.0
    assert json.dumps(out)  # plain JSON, nothing opaque


def test_gold_is_never_sent_to_the_server(servers):
    srv = servers("clef")
    bm.run_benchmark([srv.target()], {"fit": [_rec(1)]}, memory_probe=None)
    wire = json.dumps(srv.requests[0])
    assert "gold" not in wire
    assert "licence" not in wire
    assert "src" not in wire
    assert srv.requests[0]["questions"]["s"]["criteria"] == ["low", "mid", "high"]


def test_servers_run_one_at_a_time_and_stop_even_on_failure(servers):
    a, b = servers("a"), servers("b")
    bm.run_benchmark([a.target(), b.target()], FOLDS, memory_probe=None)
    assert servers.log == ["start a", "stop a", "start b", "stop b"]

    servers.log.clear()
    dead = bm.Target(
        "dead",
        "http://127.0.0.1:1",
        "dead",
        start=lambda: servers.log.append("start dead"),
        stop=lambda: servers.log.append("stop dead"),
    )
    out = bm.run_benchmark([dead, b.target()], {"fit": [_rec(1)]}, memory_probe=None)
    assert out["models"]["dead"]["n_failed"] == 1  # transport failure is recorded, not raised
    assert servers.log == ["start dead", "stop dead", "start b", "stop b"]


def test_latency_differs_per_model_and_percentiles(servers):
    slow, fast = servers("slow", 0.05), servers("fast", 0.0)
    out = bm.run_benchmark([slow.target(), fast.target()], FOLDS, memory_probe=None)
    assert (
        out["models"]["slow"]["latency"]["median_s"] > out["models"]["fast"]["latency"]["median_s"]
    )
    assert bm.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert bm.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert bm.latency_summary([])["median_s"] is None


def test_memory_unmeasured_is_reported_not_zero(servers):
    srv = servers("clef")
    out = bm.run_benchmark([srv.target()], {"fit": [_rec(1)]}, memory_probe=lambda: None)
    mem = out["models"]["clef"]["memory"]
    assert mem["measured"] is False
    assert mem["peak_mib"] is None


def test_broken_memory_probe_does_not_abort(servers):
    def boom():
        raise RuntimeError("nvidia-smi exploded")

    srv = servers("clef")
    out = bm.run_benchmark([srv.target()], {"fit": [_rec(1)]}, memory_probe=boom)
    assert out["models"]["clef"]["n_records"] == 1
    assert out["models"]["clef"]["memory"]["measured"] is False


# --- sealed sides are refused ------------------------------------------------


def test_held_out_and_test_files_are_refused(tmp_path, servers):
    srv = servers("clef")
    targets = [srv.target()]
    for name in ("held_out.jsonl", "test.jsonl", "train.jsonl"):
        path = tmp_path / name
        path.write_text(json.dumps(_rec(1).to_json()) + "\n", encoding="utf-8")
        with pytest.raises(CliError, match="sealed side refused"):
            bm.run_benchmark(targets, {"fit": path}, memory_probe=None)
    assert srv.requests == []
    assert servers.log == []  # refused before any server starts


def test_unknown_fold_names_and_sealed_sides_are_refused(tmp_path, servers):
    srv = servers("clef")
    targets = [srv.target()]
    rec = _rec(1)
    for name in ("train", "held_out", "test"):
        with pytest.raises(CliError, match="refused"):
            bm.run_benchmark(targets, {name: [rec]}, memory_probe=None)
    path = tmp_path / "ho.jsonl"
    path.write_text(json.dumps({"id": "x", "source": "s"}) + "\n", encoding="utf-8")
    side = dataset.load_sealed(path)
    with pytest.raises(CliError, match="sealed side"):
        bm.run_benchmark(targets, {"fit": side}, memory_probe=None)
    assert servers.log == []


def test_test_split_records_and_sealed_ids_are_refused(servers):
    srv = servers("clef")
    targets = [srv.target()]
    mixed = {"fit": [_rec(1), _rec(2, "test")]}
    with pytest.raises(CliError, match="split 'test'"):
        bm.run_benchmark(targets, mixed, memory_probe=None)
    side = dataset.SealedSide("p", "0" * 64, ("r1",), 1, ())
    folds = {"fit": [_rec(1)]}
    with pytest.raises(CliError, match="sealed side"):
        bm.run_benchmark(targets, folds, sealed=[side], memory_probe=None)
    assert servers.log == []


def test_fit_and_selection_files_load_from_write_dataset(tmp_path, servers):
    for name, recs in (("fit", FOLDS["fit"]), ("selection", FOLDS["selection"])):
        lines = [json.dumps(r.to_json()) for r in recs]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    srv = servers("clef")
    out = bm.run_benchmark(
        [srv.target()],
        {"fit": tmp_path / "fit.jsonl", "selection": tmp_path / "selection.jsonl"},
        memory_probe=None,
    )
    assert out["folds"] == {"fit": 3, "selection": 2}


def test_duplicate_target_names_are_refused(servers):
    a = servers("a")
    targets = [a.target(), a.target()]
    with pytest.raises(CliError, match="duplicate"):
        bm.run_benchmark(targets, FOLDS, memory_probe=None)


def test_pred_items_feed_metrics(servers):
    srv = servers("clef")
    out = bm.run_benchmark([srv.target()], {"fit": FOLDS["fit"]}, memory_probe=None)
    items = bm.pred_items(FOLDS["fit"], out["models"]["clef"]["predictions"])
    assert len(items) == 9  # 3 records x (choice, score, noul)
    assert {i.kind for i in items} == {"choice", "score", "noul"}


# --- /proc RSS fallback --------------------------------------------------------


def _fake_proc(root, procs):
    """procs: {pid: (ppid, rss_kb)}; writes /proc/<pid>/{stat,status} under *root*."""
    for pid, (ppid, rss) in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        # comm may contain spaces and parentheses; the parser must cope
        (d / "stat").write_text(f"{pid} (my (srv) x) S {ppid} 1 1 0 -1 0\n")
        (d / "status").write_text(f"Name:\tsrv\nVmRSS:\t{rss} kB\nThreads:\t1\n")
    (root / "self").mkdir()
    (root / "stat").write_text("cpu 1 2 3\n")


def test_proc_rss_sums_the_target_and_its_descendants_only(tmp_path):
    _fake_proc(tmp_path, {10: (1, 1024), 11: (10, 2048), 12: (11, 1024), 20: (1, 99999)})
    assert bm.proc_rss_mib(10, proc_root=tmp_path) == 4.0
    assert bm.proc_rss_mib(12, proc_root=tmp_path) == 1.0
    assert bm.proc_rss_mib(999, proc_root=tmp_path) is None


def test_gpu_probe_none_falls_back_to_proc_rss_and_records_the_probe(servers, tmp_path):
    _fake_proc(tmp_path, {10: (1, 2048), 11: (10, 2048)})
    srv = servers("clef")
    t = srv.target()
    t = bm.Target(t.name, t.base_url, t.model, start=t.start, stop=t.stop, pid=lambda: 10)
    out = bm.run_benchmark([t], {"fit": [_rec(1)]}, memory_probe=lambda: None, proc_root=tmp_path)
    mem = out["models"]["clef"]["memory"]
    assert mem["measured"] is True
    assert mem["peak_mib"] == 4.0
    assert mem["probe"] == "proc-rss"


def test_gpu_probe_figure_wins_and_is_labelled_nvidia_smi(servers, tmp_path):
    _fake_proc(tmp_path, {10: (1, 2048)})
    srv = servers("clef")
    t = srv.target()
    t = bm.Target(t.name, t.base_url, t.model, pid=10)
    out = bm.run_benchmark(
        [t], {"fit": [_rec(1)]}, memory_probe=bm.nvidia_smi_used_mib, proc_root=tmp_path
    )
    mem = out["models"]["clef"]["memory"]
    if mem["probe"] == "nvidia-smi":
        assert mem["measured"] is True
    else:  # no usable GPU probe on this box: the fallback produced the figure
        assert mem["probe"] == "proc-rss"
        assert mem["peak_mib"] == 2.0


def test_neither_probe_working_stays_unmeasured(servers, tmp_path):
    srv = servers("clef")
    t = srv.target()  # no pid
    out = bm.run_benchmark([t], {"fit": [_rec(1)]}, memory_probe=lambda: None, proc_root=tmp_path)
    mem = out["models"]["clef"]["memory"]
    assert mem["measured"] is False
    assert mem["probe"] is None
    assert mem["peak_mib"] is None
    # a pid whose process is gone is also unmeasured
    t2 = bm.Target(t.name, t.base_url, t.model, pid=4242)
    out = bm.run_benchmark([t2], {"fit": [_rec(1)]}, memory_probe=lambda: None, proc_root=tmp_path)
    assert out["models"]["clef"]["memory"]["measured"] is False


# ---------------------------------------------------------------------------
# t13: stock reproduction of the card's JevBench w4096 result (180/231 = 0.7792)
# ---------------------------------------------------------------------------


def _bench_task(i: int, gold: str) -> Record:
    return Record(
        id=f"jb{i}",
        state=f"task {i}",
        questions={"c": Question("choice", [("fix", "fix it"), ("wait", "wait")])},
        gold={"c": Gold(label=gold)},
        split="train",
        source="jevbench",
        licence="mit",
        gold_provenance="human",
    )


def _tasks(n_right: int, n: int = bm.CARD_N) -> list[Record]:
    """*n* tasks; the fake server always answers 'fix', so exactly *n_right* are right."""
    return [_bench_task(i, "fix" if i < n_right else "wait") for i in range(n)]


def test_wilson_interval_brackets_the_point_and_handles_edges():
    lo, hi = bm.wilson_interval(180, 231)
    assert lo < bm.CARD_ACCURACY < hi
    assert 0.72 < lo < 0.73
    assert 0.82 < hi < 0.83
    assert bm.wilson_interval(0, 0) == (0.0, 1.0)
    assert bm.wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_reproduction_that_matches_the_card_is_recorded_without_mismatch(servers):
    """AC1: n, accuracy and CI are recorded; a CI that holds 0.7792 is not a base mismatch."""
    srv = servers("stock")
    out = bm.reproduce_stock(srv.target(), _tasks(180), base_revision="rev1", memory_probe=None)
    assert out["n"] == 231
    assert out["n_correct"] == 180
    assert out["accuracy"] == pytest.approx(180 / 231)
    assert out["ci_low"] < bm.CARD_ACCURACY < out["ci_high"]
    assert out["ci_method"] == "wilson-95"
    assert out["base_revision"] == "rev1"
    assert out["card"] == {"n": 231, "n_correct": 180, "accuracy": pytest.approx(0.7792207)}
    assert out["base_mismatch"] is False
    assert len(srv.requests) == 231
    assert json.dumps(out)


def test_reproduction_whose_ci_excludes_the_card_marks_base_mismatch(servers):
    """AC1: 150/231 has a CI wholly below 0.7792, so the run is marked base-mismatch."""
    srv = servers("stock")
    out = bm.reproduce_stock(srv.target(), _tasks(150), base_revision="rev1", memory_probe=None)
    assert out["ci_high"] < bm.CARD_ACCURACY
    assert out["base_mismatch"] is True


def test_reproduction_within_noise_is_not_a_mismatch(servers):
    """AC1: 172/231 differs from 180 but its CI still holds 0.7792."""
    srv = servers("stock")
    out = bm.reproduce_stock(srv.target(), _tasks(172), base_revision="rev1", memory_probe=None)
    assert out["base_mismatch"] is False


def test_reproduction_refuses_a_task_set_of_the_wrong_size_or_no_base(servers):
    srv = servers("stock")
    small = _tasks(10, n=20)
    with pytest.raises(CliError, match="231"):
        bm.reproduce_stock(srv.target(), small, base_revision="rev1")
    full = _tasks(180)
    with pytest.raises(CliError, match="base revision"):
        bm.reproduce_stock(srv.target(), full, base_revision="")
    assert srv.requests == []


def test_failed_calls_count_as_wrong_in_the_reproduction(servers):
    srv = servers("stock")
    srv.httpd.shutdown()
    srv.httpd.server_close()
    out = bm.reproduce_stock(srv.target(), _tasks(231), base_revision="rev1", memory_probe=None)
    assert out["n"] == 231
    assert out["n_correct"] == 0
    assert out["base_mismatch"] is True
    assert out["n_failed"] == 231


def test_tasks_file_hash_is_recorded_next_to_the_cards(servers, tmp_path):
    path = tmp_path / "jevbench.jsonl"
    dataset_rows = [json.dumps(t.to_json()) for t in _tasks(180)]
    path.write_text("\n".join(dataset_rows) + "\n")
    tasks = bm.load_jevbench(path)
    srv = servers("stock")
    out = bm.reproduce_stock(
        srv.target(), tasks, base_revision="rev1", tasks_path=path, memory_probe=None
    )
    assert len(out["tasks_sha256"]) == 64
    assert out["card_tasks_sha256"] == bm.CARD_TASKS_SHA256


def test_sealed_jevbench_file_name_is_refused(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text("")
    with pytest.raises(CliError, match="sealed"):
        bm.load_jevbench(path)


def _mismatch_record(run_dir, mismatch):
    bm.write_reproduction(
        run_dir,
        {"base_mismatch": mismatch, "n": 231, "n_correct": 150, "ci_low": 0.58, "ci_high": 0.70},
    )


def test_require_base_match_refuses_a_mismatch_without_a_deviation(tmp_path):
    """AC1: train refuses on base-mismatch unless a deviation id is given."""
    _mismatch_record(tmp_path, True)
    with pytest.raises(CliError, match="base-mismatch"):
        bm.require_base_match(tmp_path)
    with pytest.raises(CliError, match="deviation"):
        bm.require_base_match(tmp_path, "  ")
    bm.require_base_match(tmp_path, "d7")


def test_require_base_match_passes_a_match_and_an_absent_record(tmp_path):
    bm.require_base_match(tmp_path)
    _mismatch_record(tmp_path, False)
    bm.require_base_match(tmp_path)


def test_require_base_match_refuses_a_corrupt_record(tmp_path):
    path = tmp_path / bm.REPRODUCTION_REL
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(CliError, match="unreadable"):
        bm.require_base_match(tmp_path, "d7")


def test_train_refuses_on_base_mismatch_without_a_deviation(tmp_path, pin_cache, monkeypatch):
    """AC1: run_grid itself refuses before planning anything; a deviation id lets it through."""
    from jev_factory.backbones.pointer import train_distill as td
    from jev_factory.distill import train as T
    from jev_factory.distill.runspec import TrainerEnv

    path, run_dir = tt.locked(tmp_path)
    _mismatch_record(run_dir, True)
    kwargs = dict(
        prereg_path=path,
        run_dir=run_dir,
        cache_dir=pin_cache,
        pins=tt.student_pins(),
        folds=tt.folds(),
        trainer_env=TrainerEnv(python="python3", decider_repo=tmp_path),
    )
    monkeypatch.setattr(td, "plan_train", lambda **k: pytest.fail("planned"))
    with pytest.raises(CliError, match="base-mismatch"):
        T.run_grid(**kwargs)


def test_measure_stock_scores_probe_and_folds_from_the_server_never_copying(servers):
    """AC2: probe, fit and selection are all measured by the (fake) server in one session."""
    srv = servers("stock")
    probe = [_bench_task(100 + i, "fix" if i < 3 else "wait") for i in range(4)]
    fit = [_bench_task(200 + i, "fix") for i in range(5)]
    sel = [_bench_task(300 + i, "wait") for i in range(2)]
    out = bm.measure_stock(
        srv.target(), {"probe": probe, "fit": fit, "selection": sel}, memory_probe=None
    )
    assert len(srv.requests) == 11
    assert out["sets"]["probe"]["n"] == 4
    assert out["sets"]["probe"]["accuracy"] == 0.75
    assert out["sets"]["fit"]["accuracy"] == 1.0
    assert out["sets"]["selection"]["accuracy"] == 0.0
    assert out["sets"]["fit"]["ci_method"] == "wilson-95"
    assert srv.log == ["start stock", "stop stock"]


def test_probe_rows_are_private_but_folds_keep_theirs(servers):
    srv = servers("stock")
    probe = [_bench_task(100, "fix")]
    fit = [_bench_task(200, "fix")]
    out = bm.measure_stock(srv.target(), {"probe": probe, "fit": fit}, memory_probe=None)
    ids = [p["record_id"] for p in out["predictions"]]
    assert ids == ["jb200"]
    assert "jb100" not in json.dumps(out)


def test_measure_stock_refuses_sealed_unknown_and_empty(servers):
    srv = servers("stock")
    test_rec = _bench_task(1, "fix")
    test_rec = Record(**{**test_rec.__dict__, "split": "test"})
    with pytest.raises(CliError, match="test"):
        bm.measure_stock(srv.target(), {"fit": [test_rec]})
    with pytest.raises(CliError, match="unknown"):
        bm.measure_stock(srv.target(), {"held_out": [_bench_task(2, "fix")]})
    with pytest.raises(CliError, match="no sets"):
        bm.measure_stock(srv.target(), {})
    assert srv.requests == []
    assert srv.log == []
