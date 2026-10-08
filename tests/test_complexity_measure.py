"""Characterization tests pinning measure-module branches before the d17 complexity refactor.

Each test pins a branch of a function Sonar flagged for cognitive complexity
(S3776) that the existing suite did not exercise, so the behaviour-neutral
refactor of probe.py, run.py, serve.py and snapshot.py is checked against it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.measure import probe
from jev_factory.measure import run as measure
from jev_factory.measure import serve
from jev_factory.measure import snapshot as snap
from tests.fixtures.stub_servers import free_port, write_stub
from tests.fixtures.toy_domain import DOMAIN
from tests.test_measure_probe import DOMAIN_ARG as PROBE_DOMAIN_ARG
from tests.test_measure_probe import _render as probe_render
from tests.test_measure_probe import _split_file as probe_split_file
from tests.test_measure_probe import identity
from tests.test_measure_run import PICKS, Harness, _argv, _page, _split, by_request

# ---------------------------------------------------------------------------
# probe.main
# ---------------------------------------------------------------------------


def _probe_handle(closed: list | None = None):
    def build(_args):
        return measure.ScorerHandle(
            top_k=identity("lamp_status"),
            render=probe_render,
            close=lambda: closed.append(True) if closed is not None else None,
        )

    return build


def test_probe_cli_reports_an_unloadable_domain(tmp_path, capsys):
    path = probe_split_file(tmp_path, "val.json")
    missing = str(tmp_path / "no-such-domain.json")
    got = probe.main(["--domain", missing, "--split", str(path), "--model", "m"])
    assert got == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_probe_cli_reports_an_unreadable_split(tmp_path, capsys):
    missing = tmp_path / "val.json"
    got = probe.main([*PROBE_DOMAIN_ARG, "--split", str(missing), "--model", "m"])
    assert got == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert str(missing) in err


def test_probe_cli_reports_an_unparseable_split(tmp_path, capsys):
    path = tmp_path / "val.json"
    path.write_text("{not json")
    assert probe.main([*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_probe_cli_refuses_a_split_with_no_valid_entries(tmp_path, capsys):
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"entries": []}))
    assert probe.main([*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m"]) == 1
    assert capsys.readouterr().err == f"error: {path} has no valid entries\n"


def test_probe_cli_refuses_a_malformed_paraphrase_file(tmp_path, capsys):
    path = probe_split_file(tmp_path, "val.json")
    bad = tmp_path / "para.json"
    bad.write_text(json.dumps({"lamp_on": ["only one"]}))
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--paraphrases", str(bad)]
    assert probe.main(argv) == 1
    assert capsys.readouterr().err == (
        f"error: {bad}: 'lamp_on' has fewer than 2 alternative descriptions\n"
    )


def test_probe_cli_served_scorer_needs_a_base_url(tmp_path, capsys):
    path = probe_split_file(tmp_path, "val.json")
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--scorer", "served"]
    assert probe.main(argv) == 1
    assert capsys.readouterr().err == "error: --scorer served needs --base-url\n"


def test_probe_cli_prints_the_report_without_out(tmp_path, capsys):
    path = probe_split_file(tmp_path, "val.json")
    closed: list = []
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "2"]
    assert probe.main(argv, build_scorer=_probe_handle(closed)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pooled_change_rate"] == 0.0
    assert report["per_entry"] == 2
    assert closed == [True]


def test_probe_cli_uses_a_paraphrase_file_and_the_served_scorer(tmp_path, capsys):
    path = probe_split_file(tmp_path, "val.json")
    para = tmp_path / "para.json"
    para.write_text(json.dumps({"lamp_status": ["one way", "another way"]}))
    seen: list = []

    def build(args):
        seen.append((args.scorer, args.base_url))
        return _probe_handle()(args)

    out = tmp_path / "probe.json"
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "1"]
    argv += ["--scorer", "served", "--base-url", "http://127.0.0.1:9/v1"]
    argv += ["--paraphrases", str(para), "--out", str(out)]
    assert probe.main(argv, build_scorer=build) == 0
    assert seen == [("served", "http://127.0.0.1:9/v1")]
    kinds = {row["kind"] for row in json.loads(out.read_text())["kinds"]}
    assert "paraphrase" in kinds
    assert capsys.readouterr().out == ""


def test_probe_cli_progress_name_for_an_out_without_a_parent_dir(tmp_path, monkeypatch):
    from jev_factory.factory import detach

    path = probe_split_file(tmp_path, "val.json")
    monkeypatch.chdir(tmp_path)
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "1"]
    argv += ["--out", "my.probe.json", "--progress-dir", str(tmp_path / "jobs")]
    assert probe.main(argv, build_scorer=_probe_handle()) == 0
    progress = detach.read_progress(tmp_path / "jobs", "probe-my_probe")
    assert progress["done"] == progress["total"] == 1


def test_probe_cli_progress_name_defaults_to_the_split(tmp_path):
    from jev_factory.factory import detach

    path = probe_split_file(tmp_path, "val.json")
    argv = [*PROBE_DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "1"]
    argv += ["--progress-dir", str(tmp_path / "jobs")]
    assert probe.main(argv, build_scorer=_probe_handle()) == 0
    stem = f"{tmp_path.name}-val".replace(".", "_")
    assert detach.read_progress(tmp_path / "jobs", "probe-" + stem)["total"] == 1


# ---------------------------------------------------------------------------
# probe.probe_entry / kind_report
# ---------------------------------------------------------------------------


def test_probe_entry_without_paraphrases_records_no_paraphrase_trials():
    entry = probe.CorpusEntry(
        id="e", kind="explicit", text="which lights are on", expect={"operation": "lamp_status"}
    )
    outcome = probe.probe_entry(
        DOMAIN,
        identity("lamp_status"),
        probe_render,
        entry,
        pool=DOMAIN.names(),
        per_entry=2,
        seed=0,
        paraphrases=None,
    )
    assert set(outcome.trials) == {"order", "letters", "subset", "all"}
    assert all(len(trials) == 2 for trials in outcome.trials.values())


def test_probe_entry_with_zero_trials_records_no_kind():
    entry = probe.CorpusEntry(
        id="e", kind="explicit", text="which lights are on", expect={"operation": "lamp_status"}
    )
    outcome = probe.probe_entry(
        DOMAIN,
        identity("lamp_status"),
        probe_render,
        entry,
        pool=DOMAIN.names(),
        per_entry=0,
        seed=0,
        paraphrases={"lamp_on": ["a", "b"]},
    )
    assert outcome.trials == {}
    assert outcome.baseline_choice == "lamp_status"


def test_kind_report_subset_counts_removed_baseline_choices():
    outcome = probe.EntryOutcome(
        "a",
        "x",
        "x",
        trials={
            "subset": [
                (True, False, True, False),
                (False, True, False, False),
                (False, False, False, True),
            ]
        },
    )
    other = probe.EntryOutcome("b", "x", "x", trials={"order": [(True, False, None, False)]})
    report = probe.kind_report([outcome, other], "subset", bootstrap_seed=3)
    assert report["entries"] == 1
    assert report["baseline_choice_removed"] == {"trials": 2, "removed": 1, "rate": 0.5}
    assert report["label_case"] == {
        "lowercase_trials": 1,
        "lowercase_changes": 0,
        "lowercase_rate": 0.0,
        "uppercase_trials": 1,
        "uppercase_changes": 1,
        "uppercase_rate": 1.0,
    }
    assert report["incomplete"] == {"trials": 1, "of": 3, "rate": 1 / 3}


def test_kind_report_subset_with_every_trial_incomplete():
    outcome = probe.EntryOutcome("a", "x", "x", trials={"subset": [(True, True, True, True)]})
    report = probe.kind_report([outcome], "subset", bootstrap_seed=0)
    assert report["baseline_choice_removed"] == {"trials": 0, "removed": 0, "rate": None}
    assert report["label_case"]["lowercase_rate"] is None
    assert report["label_case"]["uppercase_rate"] is None


# ---------------------------------------------------------------------------
# run.render_markdown / _check_flags / _run
# ---------------------------------------------------------------------------


def _prov(**fields) -> measure.Provenance:
    base = {
        "date": "2026-09-30",
        "label": "x",
        "command": "cmd",
        "domain": "toy",
        "surface_sha256": "s",
        "split_path": "p",
        "split_sha256": "h",
        "split_count": 1,
        "source_count": 1,
        "problems": (),
        "seed": None,
        "seed_origin": "",
        "grounding": "g",
        "final": False,
        "acceptance": False,
        "sealed_before": 0,
        "deviation": None,
        "decision_mode": "d",
        "slice_note": "full split",
    }
    base.update(fields)
    return measure.Provenance(**base)


def test_render_markdown_counts_corpus_problems():
    page = measure.render_markdown(_prov(problems=("a", "b")), [])
    assert "- Corpus problems (entries skipped): 2" in page.splitlines()
    assert "- Seed: not recorded" in page.splitlines()


def test_render_markdown_acceptance_slice_and_nothing_measured():
    records = [measure.RunRecord(model="m", revision="r", failure="boom")]
    prov = _prov(acceptance=True, slice=measure.SLICE_MISSING, sealed_before=2, seed=7)
    lines = measure.render_markdown(prov, records).splitlines()
    assert "- Seed: 7 ()" in lines
    assert measure.NOT_MEASURED_MARKER in lines
    assert (
        f"- Measurements of the {measure.HELD_OUT} side ({measure.SLICE_MISSING} slice) in this"
        " run, including this one: 2"
    ) in lines
    assert "(not captured)" in lines


def test_check_flags_refuses_a_non_positive_ctx(tmp_path, capsys):
    harness = Harness()
    argv = _argv(tmp_path, _split(tmp_path), "--ctx", "0")
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_USER
    assert capsys.readouterr().err == "error: --ctx must be a positive whole number\n"


def test_check_flags_serve_takes_one_model(tmp_path, capsys):
    harness = Harness()
    argv = _argv(tmp_path, _split(tmp_path), models=("a", "b"))
    argv[argv.index("in-process")] = "served"
    argv += ["--serve", "x.gguf", "--tokenizer", "t"]
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_USER
    assert capsys.readouterr().err == "error: --serve serves one --model per run\n"


def test_check_flags_attached_with_enough_logprobs_passes(tmp_path):
    from jev_factory.backbones.causal_lm.readout import READOUT_TOP

    args = measure._parser().parse_args(
        _argv(tmp_path, tmp_path / "s.json")
        + ["--base-url", "http://127.0.0.1:9/v1", "--max-logprobs", str(READOUT_TOP)]
    )
    args.scorer = measure.SCORER_SERVED
    assert measure._check_flags(args) is None


def test_run_refuses_a_split_with_no_valid_entries(tmp_path, capsys):
    harness = Harness()
    split = _split(tmp_path, entries=[{"id": "broken"}])
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == measure.EXIT_USER
    assert capsys.readouterr().err == "error: val.json has no valid entries\n"
    assert not _page(tmp_path).exists()


def test_run_records_an_explicit_seed_and_ctx(tmp_path):
    harness = Harness(by_request(PICKS))
    split = _split(tmp_path)
    argv = _argv(tmp_path, split, "--seed", "5", "--ctx", "4096")
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_OK
    lines = _page(tmp_path).read_text().splitlines()
    assert "- Seed: 5 (--seed)" in lines
    assert "- Context: 4096" in lines


def test_run_default_seed_and_ctx(tmp_path):
    harness = Harness(by_request(PICKS))
    assert measure.main(_argv(tmp_path, _split(tmp_path)), seams=harness.seams) == 0
    lines = _page(tmp_path).read_text().splitlines()
    assert "- Seed: 39 (from the split header)" in lines
    assert "- Context: 2048" in lines
    assert harness.guarded == [("measure", False)]


def test_run_missing_slice_with_a_structured_header(tmp_path):
    harness = Harness(by_request(PICKS))
    split = tmp_path / "splits" / "val.json"
    split.parent.mkdir(parents=True)
    entries = json.loads(_split(tmp_path, name="tmp.json").read_text())["entries"]
    header = {"split": "val", "seed": 39}
    split.write_text(json.dumps({"header": header, "entries": entries}))
    argv = _argv(tmp_path, split, "--slice", measure.SLICE_MISSING, "--seed", "1")
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_OK
    page = _page(tmp_path).read_text()
    assert f"- Slice: {measure.SLICE_MISSING} (each operation entry" in page


def _serve_argv(tmp_path: Path, *extra: str) -> list:
    argv = _argv(tmp_path, _split(tmp_path), *extra)
    argv[argv.index("in-process")] = "served"
    return argv + ["--serve", str(tmp_path / "m.gguf"), "--tokenizer", "t"]


def test_run_serve_takes_ctx_and_settings_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_MEASURE_CTX", "3072")
    monkeypatch.delenv("JEV_MEASURE_GPU_LAYERS", raising=False)
    harness = Harness(by_request(PICKS))
    argv = _serve_argv(tmp_path, "--llama-server", "/opt/llama-server")
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_OK
    assert "- Context: 3072" in _page(tmp_path).read_text().splitlines()
    settings = harness.built[0].settings
    assert settings.ctx == 3072
    assert settings.run_dir == tmp_path / "run" / "measure" / "serve"
    assert settings.llama_server == "/opt/llama-server"
    assert harness.guarded == [("measure", False)]


def test_run_serve_ctx_flag_overrides_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_MEASURE_CTX", "3072")
    monkeypatch.setenv("JEV_MEASURE_GPU_LAYERS", "0")
    monkeypatch.setenv("JEV_LLAMA_SERVER", "/env/llama-server")
    harness = Harness(by_request(PICKS))
    argv = _serve_argv(tmp_path, "--ctx", "1024", "--allow-foreign-gpu")
    assert measure.main(argv, seams=harness.seams) == measure.EXIT_OK
    assert "- Context: 1024" in _page(tmp_path).read_text().splitlines()
    settings = harness.built[0].settings
    assert settings.ctx == 1024
    assert settings.llama_server == "/env/llama-server"
    assert harness.guarded == []  # a CPU-served .gguf needs no free GPU


# ---------------------------------------------------------------------------
# serve.start_llama
# ---------------------------------------------------------------------------


@pytest.fixture
def llama_settings(tmp_path) -> serve.ServeSettings:
    from tests.fixtures.stub_servers import stub_bin

    directory = stub_bin(tmp_path / "bin")
    return serve.ServeSettings(
        run_dir=tmp_path / "run",
        llama_server=str(directory / "llama-server"),
        wait_seconds=5,
        poll_seconds=0.05,
        stop_seconds=5,
        lock_seconds=5,
    )


@pytest.fixture
def model_file(tmp_path) -> Path:
    path = tmp_path / "models" / "m.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF stub")
    return path


def test_start_llama_refuses_a_non_executable_binary(tmp_path, llama_settings, model_file):
    plain = tmp_path / "llama-server.txt"
    plain.write_text("not a program")
    settings = replace(llama_settings, llama_server=str(plain))
    port = free_port()
    with pytest.raises(serve.ServeError) as excinfo:
        serve.start_llama(model_file, port, settings)
    assert excinfo.value.message == f"llama_server={plain} is not an executable file"
    assert excinfo.value.code == 2


def test_start_llama_refuses_a_directory_binary(tmp_path, llama_settings, model_file):
    settings = replace(llama_settings, llama_server=str(tmp_path))
    port = free_port()
    with pytest.raises(serve.ServeError, match="is not an executable file"):
        serve.start_llama(model_file, port, settings)


def test_start_llama_reports_a_failing_version(tmp_path, llama_settings, model_file):
    binary = write_stub(
        tmp_path / "bad", "llama-server", "print('  boom  ')\nraise SystemExit(3)\n"
    )
    settings = replace(llama_settings, llama_server=str(binary))
    port = free_port()
    with pytest.raises(serve.ServeError) as excinfo:
        serve.start_llama(model_file, port, settings)
    assert excinfo.value.message == f"'{binary.resolve()} --version' failed: boom"
    assert excinfo.value.code == 2
    assert not settings.run_dir.exists()


def test_start_llama_reports_a_server_that_exits_before_it_is_identified(
    llama_settings, model_file, monkeypatch
):
    monkeypatch.setenv("STUB_LLAMA_FAIL", "1")
    monkeypatch.setattr(serve, "is_ours", lambda pid, starttime, argv: False)
    port = free_port()
    settings = replace(llama_settings, start_seconds=30)
    with pytest.raises(serve.ServeError) as excinfo:
        serve.start_llama(model_file, port, settings)
    assert excinfo.value.message == (
        f"llama-server exited at once; see {serve.log_file(settings, port)}"
    )
    assert excinfo.value.code == 2
    assert not serve.state_file(settings, port).exists()


def test_start_llama_stops_a_process_that_never_shows_the_recorded_argv(
    llama_settings, model_file, monkeypatch
):
    monkeypatch.setattr(serve, "is_ours", lambda pid, starttime, argv: False)
    port = free_port()
    settings = replace(llama_settings, start_seconds=0.3)
    record = settings.run_dir / "record.json"
    with pytest.raises(serve.ServeError) as excinfo:
        serve.start_llama(model_file, port, settings, record)
    binary = Path(llama_settings.llama_server).resolve()
    assert excinfo.value.message == (
        f"the launched process never ran {binary} with the recorded argv"
    )
    assert not serve.state_file(settings, port).exists()
    saved = json.loads(record.read_text())
    assert saved["served_model_name"] == "m"
    assert saved["port"] == port


def test_start_llama_uses_the_configured_model_name(llama_settings, model_file, capsys):
    port = free_port()
    settings = replace(llama_settings, model_name="alias-x")
    pid = serve.start_llama(model_file, port, settings)
    try:
        state = serve.read_state(settings, port)
        assert state["pid"] == pid
        assert state["alias"] == "alias-x"
        assert state["backend"] == serve.LLAMA_BACKEND
        assert state["argv"][-1] == "alias-x"
        assert capsys.readouterr().err == (
            f"serve: started llama-server (pid {pid}) serving m.gguf as 'alias-x'"
            f" on 127.0.0.1:{port}\n"
        )
    finally:
        serve.stop(port, settings)


# ---------------------------------------------------------------------------
# snapshot.build_snapshot
# ---------------------------------------------------------------------------


def _lookup_domain(result):
    kind = replace(DOMAIN.ground_kinds[0], lookup=lambda: result)
    return replace(DOMAIN, ground_kinds=(kind,))


@pytest.mark.parametrize("result", ["kitchen", b"kitchen", 7, None])
def test_build_snapshot_refuses_a_lookup_that_is_not_a_list(result):
    domain = _lookup_domain(result)
    with pytest.raises(snap.SnapshotError) as excinfo:
        snap.build_snapshot(domain, [], source="s", created="c", base_world={})
    assert str(excinfo.value) == "the rooms lookup did not return a list"
    assert excinfo.value.env


def test_build_snapshot_keeps_only_string_lookup_values():
    domain = _lookup_domain(("kitchen", 3, None, "study"))
    snapshot, counts = snap.build_snapshot(
        domain, [], source="s", created="c", base_world={"home": "h"}
    )
    assert snapshot["rooms"] == ["kitchen", "study"]
    assert counts == {"rooms": 2, "rooms_live": 2, "rooms_splits_only": 0}


def test_build_snapshot_skips_unusable_split_entries(tmp_path):
    entries = [
        "not a dict",
        {"id": "a"},
        {"id": "b", "expect": "nope"},
        {"id": "c", "expect": {"operation": "lamp_on", "args": "nope"}},
        {"id": "d", "expect": {"operation": "lamp_on", "args": {"room": 3}}},
        {"id": "e", "expect": {"operation": "lamp_on", "args": {"room": ""}}},
        {"id": "f", "expect": {"operation": "no_such_op", "args": {"room": "attic"}}},
        {"id": "g", "expect": {"operation": "lamp_on", "args": {"other": "attic"}}},
        {"id": "h", "expect": {"operation": "lamp_on", "args": {"room": "cellar"}}},
    ]
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"entries": entries}))
    snapshot, counts = snap.build_snapshot(
        DOMAIN, [path], source="s", created="c", base_world={"home": "h", "rooms": ["kitchen"]}
    )
    assert snapshot == {"home": "h", "rooms": ["cellar", "kitchen"], "source": "s", "created": "c"}
    assert counts == {"rooms": 2, "rooms_live": 1, "rooms_splits_only": 1}


def test_build_snapshot_ignores_an_argument_whose_ground_kind_is_unknown(tmp_path):
    domain = replace(DOMAIN, ground_kinds=())
    entries = [{"id": "a", "expect": {"operation": "lamp_on", "args": {"room": "attic"}}}]
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"entries": entries}))
    snapshot, counts = snap.build_snapshot(
        domain, [path], source="s", created="c", base_world={"home": "h", "rooms": ["x"]}
    )
    assert snapshot == {"home": "h", "rooms": ["x"], "source": "s", "created": "c"}
    assert counts == {}


def test_build_snapshot_schema_refusal_message():
    with pytest.raises(snap.SnapshotError) as excinfo:
        snap.build_snapshot(DOMAIN, [], source="s", created="c")
    message = str(excinfo.value)
    assert message.startswith("the snapshot misses the domain's world schema: ")
    assert message.endswith(" (pass --base-world with a corpus whose world has them)")
    assert not excinfo.value.env
