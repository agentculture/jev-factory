"""t15: edge measurement of a served bundle through a localhost tunnel, against a fake server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import edge
from jev_factory.distill.records import Gold, Question, Record

FILES = {"model.safetensors": "a" * 64, "head.safetensors": "b" * 64}
BUNDLE = {"files": FILES, "manifest_sha256": "c" * 64}
CAPPED = {
    "rss_bytes": 700_000_000,
    "mem_available_bytes": 3_000_000_000,
    "device_model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit",
    "nv_tegra_release": "# R36 (release), REVISION: 4.3",
    "cgroup": {
        "version": 2,
        "path": "/x",
        "memory_max": 4_000_000_000,
        "effective_memory_max": 4_000_000_000,
    },
}
UNCAPPED = {
    **CAPPED,
    "cgroup": {"version": 2, "path": "/", "memory_max": "max", "effective_memory_max": None},
}


def _rec(rid: str, split: str = "train") -> Record:
    return Record(
        id=rid,
        state=f"state {rid}",
        questions={"c": Question("choice", [("fix", "fix it"), ("wait", "wait")])},
        gold={"c": Gold(label="fix")},
        split=split,
        source="src",
        licence="mit",
        gold_provenance="human",
    )


class _Server:
    def __init__(self, bundle=None, stats=None, p_fix=0.8):
        outer = self
        self.bundle = BUNDLE if bundle is None else bundle
        self.stats = CAPPED if stats is None else stats
        self.posts = 0

        class H(BaseHTTPRequestHandler):
            def _send(self, doc):
                raw = json.dumps(doc).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                self._send(outer.bundle if self.path == "/v1/bundle" else outer.stats)

            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.posts += 1
                self._send({"answers": {"c": {"probabilities": {"fix": p_fix, "wait": 1 - p_fix}}}})

            def log_message(self, *_a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(**kw):
        made.append(_Server(**kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


def _jsonl(path: Path, recs) -> None:
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in recs), encoding="utf-8")


def _write_reference(run: Path, rows, bundle=BUNDLE) -> Path:
    path = run / edge.REFERENCE_PREDICTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": "torch_int4_wo", "fold": "selection", "bundle": bundle, "predictions": rows}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path):
    run, ds = tmp_path / "run", tmp_path / "ds"
    ds.mkdir()
    (run / "distill/final/predictions").mkdir(parents=True)
    for stem in ("test", "held_out"):
        rows = [{"id": f"{stem}-1", "source": "s"}]
        (ds / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _jsonl(ds / "selection.jsonl", [_rec("s1"), _rec("s2")])
    _jsonl(tmp_path / "probe.jsonl", [_rec("p1")])
    (run / "distill/final/final.json").write_text(json.dumps({"deployed": {"bundle": BUNDLE}}))
    # the final's student answers cover test/held-out only: they never overlap edge's inputs
    final_rows = [{"record_id": "test-1", "questions": {"c": {"distribution": {"fix": 1.0}}}}]
    (run / "distill/final/predictions/student.json").write_text(json.dumps(final_rows))
    quant_rows = [
        {
            "fold": "selection",
            "record_id": "s1",
            "questions": {"c": {"distribution": {"fix": 0.9, "wait": 0.1}}},
        },
        {
            "fold": "selection",
            "record_id": "s2",
            "questions": {"c": {"distribution": {"fix": 0.2, "wait": 0.8}}},
        },
    ]
    _write_reference(run, quant_rows)
    return {"run": run, "ds": ds, "sel": ds / "selection.jsonl", "probe": tmp_path / "probe.jsonl"}


def _go(w, url, **kw):
    return edge.run_edge(
        run_dir=w["run"],
        dataset_dir=w["ds"],
        selection=w["sel"],
        probe=w["probe"],
        base_url=url,
        served_model="m",
        label="nano",
        apply=kw.pop("apply", True),
        **kw,
    )


def test_measures_and_writes_manifest(world, server):
    """AC1: latency median/p95, RSS and MemAvailable from /v1/stats; agreement (risk r11) with
    the quant stage's selection-fold predictions of the shipped bundle, not the final's."""
    srv = server()
    out = _go(world, srv.url)
    assert out["n_records"] == 3
    assert out["latency"]["median_s"] is not None
    assert out["latency"]["p95_s"] is not None
    assert out["memory"]["rss_bytes_peak"] == 700_000_000
    assert out["memory"]["mem_available_bytes_min"] == 3_000_000_000
    # the server always answers "fix": agrees with s1, disagrees with s2; the probe is not in
    # the reference (it is the selection fold only)
    assert out["agreement"]["n_compared"] == 2
    assert out["agreement"]["reference"] == edge.REFERENCE_PREDICTIONS
    assert out["agreement"]["fold"] == "selection"
    assert out["agreement"]["n_agree"] == 1
    assert out["agreement"]["agreement"] == 0.5
    assert (world["run"] / "distill/edge/nano/edge_manifest.json").is_file()


def test_manifest_records_board_l4t_and_cap(world, server):
    """AC2: the manifest records board model, L4T release and the cap."""
    out = _go(world, server().url)
    assert out["board"]["device_model"].startswith("NVIDIA Jetson Orin Nano")
    assert out["board"]["nv_tegra_release"] == CAPPED["nv_tegra_release"]
    assert out["cap"]["effective_memory_max"] == 4_000_000_000


def test_probe_ids_hashed_in_written_predictions(world, server):
    """The probe set is private: its ids are not written."""
    _go(world, server().url)
    text = (world["run"] / "distill/edge/nano/edge_predictions.json").read_text()
    assert "p1" not in json.loads(text)[2]["record_id"]
    assert json.loads(text)[2]["record_id"].startswith("sha256:")


def test_dry_run_sends_nothing(world, server):
    """Dry-run by default: no request, no files."""
    srv = server()
    out = _go(world, srv.url, apply=False)
    assert out["dry_run"] is True
    assert srv.posts == 0
    assert not (world["run"] / "distill/edge").exists()


@pytest.mark.parametrize("sealed_id", ["test-1", "held_out-1"])
def test_sealed_id_in_input_refused(world, server, sealed_id):
    """AC1: a test or held-out id in the input exits non-zero before any request."""
    _jsonl(world["sel"], [_rec("s1"), _rec(sealed_id)])
    srv = server()
    with pytest.raises(CliError) as exc:
        _go(world, srv.url)
    assert exc.value.code == 1
    assert srv.posts == 0


def test_test_split_record_in_probe_refused(world, server):
    """AC1: a record whose split is test is refused even with an unknown id."""
    _jsonl(world["probe"], [_rec("p9", split="test")])
    url = server().url
    with pytest.raises(CliError):
        _go(world, url)


def test_non_selection_fold_refused(world, server):
    """Only the selection fold (and the probe set) may be passed."""
    other = world["ds"] / "fit.jsonl"
    _jsonl(other, [_rec("f1")])
    world["sel"] = other
    url = server().url
    with pytest.raises(CliError):
        _go(world, url)


def test_bundle_mismatch_refused(world, server):
    """AC2: /v1/bundle differing from the final's sha256s refuses."""
    srv = server(bundle={"files": {**FILES, "head.safetensors": "d" * 64}})
    with pytest.raises(CliError, match="/v1/bundle"):
        _go(world, srv.url)
    assert srv.posts == 0


def test_no_cgroup_cap_refused(world, server):
    """AC2: /v1/stats with no cgroup memory cap refuses."""
    srv = server(stats=UNCAPPED)
    with pytest.raises(CliError, match="cgroup memory cap"):
        _go(world, srv.url)
    assert srv.posts == 0


def test_teacher_lock_refused_even_in_dry_run(world, server):
    """AC2: refuses while the teacher-label lock exists."""
    edge.acquire_teacher_lock(world["run"])
    srv = server()
    with pytest.raises(CliError, match="teacher-label lock"):
        _go(world, srv.url, apply=False)
    edge.release_teacher_lock(world["run"])
    assert _go(world, srv.url, apply=False)["dry_run"] is True


def test_acquire_lock_twice_refused(tmp_path):
    """The lock helper is exclusive."""
    edge.acquire_teacher_lock(tmp_path)
    with pytest.raises(CliError):
        edge.acquire_teacher_lock(tmp_path)


def test_non_localhost_url_refused(world):
    """Only a 127.0.0.1 tunnel URL is accepted."""
    with pytest.raises(CliError):
        _go(world, "http://10.0.0.5:8000")


# --------------------------------------------------------------------- t17 wiring


def test_agreement_overlap_is_not_empty_against_the_quant_selection_predictions(world, server):
    """Risk r11: the final's predictions (test/held-out) never overlap edge's inputs; the quant
    stage's selection-fold predictions do."""
    out = _go(world, server().url)
    assert out["agreement"]["n_compared"] > 0
    assert out["agreement"]["n_reference_questions"] == 2  # the reference's selection questions


def test_missing_reference_predictions_refused_before_any_request(world, server):
    (world["run"] / edge.REFERENCE_PREDICTIONS).unlink()
    srv = server()
    with pytest.raises(CliError, match="no selection-fold predictions"):
        _go(world, srv.url)
    assert srv.posts == 0


def test_reference_from_another_bundle_refused_before_any_request(world, server):
    """Risk r11: the reference must come from the same sha256 bundle the final measured."""
    other = {"files": FILES, "manifest_sha256": "e" * 64}
    _write_reference(world["run"], [], bundle=other)
    srv = server()
    with pytest.raises(CliError, match="different bundle"):
        _go(world, srv.url)
    assert srv.posts == 0


def test_reference_with_other_file_hashes_refused(world, server):
    other = {"files": {**FILES, "head.safetensors": "f" * 64}, "manifest_sha256": "c" * 64}
    _write_reference(world["run"], [], bundle=other)
    url = server().url
    with pytest.raises(CliError, match="file sha256s"):
        _go(world, url)


def test_a_cap_other_than_the_expected_one_refused(world, server):
    """The edge stage passes distill_edge_memory_cap: a different cap is refused."""
    srv = server()
    with pytest.raises(CliError, match="expects"):
        _go(world, srv.url, expected_cap=8_000_000_000)
    assert srv.posts == 0
    out = _go(world, srv.url, expected_cap=4_000_000_000)
    assert out["expected_cap"] == 4_000_000_000


def test_an_injected_getter_answers_bundle_and_stats(world, server):
    srv = server(bundle={"files": {}}, stats=UNCAPPED)  # the HTTP GETs would refuse
    seen = []

    def get(path):
        seen.append(path)
        return BUNDLE if path == edge.BUNDLE_PATH else CAPPED

    out = _go(world, srv.url, get=get)
    assert out["n_records"] == 3
    assert seen[:2] == [edge.BUNDLE_PATH, edge.STATS_PATH]


def _write_lock(run, **holder):
    path = edge.teacher_lock_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(holder) + "\n", encoding="utf-8")
    return path


def test_a_lock_whose_process_is_gone_is_replaced_and_does_not_block(tmp_path, monkeypatch):
    """A hard-killed teacher-label run leaves its lock; a dead pid on this host is stale."""
    import socket

    monkeypatch.setattr(edge, "_pid_alive", lambda pid: False)
    _write_lock(tmp_path, owner="teacher-label", pid=12345, host=socket.gethostname())
    edge.require_no_teacher_lock(tmp_path)  # stale: final and edge are not blocked
    path = edge.acquire_teacher_lock(tmp_path)
    holder = json.loads(path.read_text(encoding="utf-8"))
    assert holder["host"] == socket.gethostname()
    assert holder["pid"] != 12345


@pytest.mark.parametrize("holder", [{"pid": 12345, "host": "another-box"}, {"note": "no pid"}])
def test_a_lock_from_another_host_or_unreadable_still_blocks(tmp_path, monkeypatch, holder):
    monkeypatch.setattr(edge, "_pid_alive", lambda pid: False)
    _write_lock(tmp_path, **holder)
    with pytest.raises(CliError, match="busy labelling"):
        edge.require_no_teacher_lock(tmp_path)
    with pytest.raises(CliError, match="already held"):
        edge.acquire_teacher_lock(tmp_path)
