"""End-to-end tests for the release-gate runner: run / continue / status / smoke.

Synthetic toy-domain case sets, synthetic saved predictions and in-process
fake providers; the run dir and the private data root live under
``tmp_path``. No network, no keys, no deepeval (the default dev env does not
install the ``evals`` group; a full run passes ``--no-deepeval`` here, and
the DeepEval-backed run is exercised only when deepeval is importable).

Ported from nvsh evals/tool_jev/tests/test_run.py, test_run_manifest.py and
test_scaffold.py (batch, Track A, judge and drive cases are deferred).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jev_factory.evals import run as runner
from jev_factory.evals.__main__ import main
from tests.evals_support import EXTRA_LOCAL, Fakes, setup_private

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "jev_factory" / "evals"
REF = "openrouter/vendor/fake-sync"


def _main(args, manifest, run_dir, env, fakes, lines=None):
    lines = [] if lines is None else lines
    code = main(
        [*args, "--manifest", str(manifest), "--run-dir", str(run_dir)],
        factory=fakes,
        env=env,
        out=lines.append,
    )
    return code, lines


def _run(tmp_path, fakes=None, extra=(), **setup):
    manifest, run_dir, env = setup_private(tmp_path, **setup)
    fakes = fakes or Fakes()
    code, lines = _main(
        ["run", "--no-deepeval", "--run-id", "r1", "--date", "2026-09-30", *extra],
        manifest,
        run_dir,
        env,
        fakes,
    )
    return code, lines, manifest, run_dir, env, fakes


# ---------------------------------------------------------------------------
# o31: the smoke with the fake provider and raw policy, no secrets, no nvsh
# ---------------------------------------------------------------------------


@pytest.mark.behavioral("o31")
def test_smoke_runs_with_the_fake_provider_and_no_secrets(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")):
            monkeypatch.delenv(name, raising=False)
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    code, lines = _main(["smoke", "--cases", "3"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines
    smoke = json.loads((run_dir / runner.SMOKE_FILE).read_text())
    row = smoke["models"][REF]
    assert (row["calls"], row["flag"], row["invalid"]) == (3, "OK", 0)
    assert smoke["cases"] == 3
    assert smoke["full_run_cases"] == 5
    assert row["projected_full_run_usd"] == pytest.approx(row["cost_usd"] * 5 / 3, abs=1e-4)
    assert fakes.get(REF).sent == ["c-1", "c-2", "c-3"]
    assert any(line.startswith(f"smoke {REF}: OK") for line in lines)


@pytest.mark.behavioral("o31")
def test_full_run_with_the_fake_provider_scores_raw_and_harness_rows(tmp_path):
    code, lines, _m, run_dir, _env, fakes = _run(tmp_path)
    assert code == runner.EXIT_OK, lines
    result = json.loads((run_dir / runner.RESULT_FILE).read_text())
    rows = {(r["subject"], r["variant"]): r for r in result["rows"]}
    model_only = rows[("cand.toy-test", "model-only")]
    harness = rows[("cand.toy-test", "model+harness")]
    assert model_only["harness_policy"] is None
    assert model_only["wrong_mutations"] == 1
    assert harness["harness_policy"] == "mutating-strict-example"
    assert harness["wrong_mutations"] == 0
    (reference,) = result["reference_rows"]
    assert reference["subject"] == "openrouter.vendor-fake-sync.toy-test"
    assert reference["policy"] == "raw"
    assert reference["right"] == {"n": 2, "N": 2}
    assert reference["wrong_mutations"] == 0
    assert reference["ece"] is not None
    assert result["deepeval"] is False
    page = (run_dir / runner.PAGE_FILE).read_text()
    assert "model+harness" in page
    assert "turn on the lamps" not in page
    assert sorted(fakes.get(REF).sent) == ["c-1", "c-2", "c-3", "c-4", "c-5-nocand"]
    traces = [json.loads(x) for x in (run_dir / "traces" / f"{reference['subject']}.jsonl").open()]
    first = traces[0]["raw"]
    assert first["outcome"] == "propose"
    assert first["arguments"] == {"room": "kitchen"}
    assert "(explain)" in first["candidates"]
    assert "explain" not in first["candidates"]
    status = runner.status(run_dir)
    assert status["status"] == "complete"
    assert status["providers"]["openrouter"]["done"] == 5


@pytest.mark.behavioral("o31")
def test_the_gate_code_has_no_nvsh_import():
    offenders = []
    for path in sorted(EVALS.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(name.split(".")[0] in ("nvsh", "evals") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    assert (EVALS / "run.py").is_file()
    assert (EVALS / "policies" / "raw.json").is_file()


def test_deferred_features_are_not_imported():
    present = {p.stem for p in EVALS.rglob("*.py")}
    for deferred in ("track_a_loop", "judge", "alerts", "drive", "anthropic", "openai"):
        assert deferred not in present
    assert not (EVALS / "docker").exists()
    assert not (EVALS / "rubric").exists()


@pytest.mark.behavioral("o31")
def test_deepeval_is_imported_lazily_and_only_from_the_evals_group():
    code = (
        "import sys\n"
        "sys.modules['deepeval'] = None\n"
        "import jev_factory.evals.run, jev_factory.evals.deepeval_layer, "
        "jev_factory.evals.__main__\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok", proc.stderr


def test_a_full_run_without_deepeval_installed_is_an_environment_error(tmp_path, monkeypatch):
    monkeypatch.setattr("jev_factory.evals.deepeval_layer.deepeval_available", lambda: False)
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    code, lines = _main(["run"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_ENV
    assert "uv run --group evals" in lines[-1]
    assert "--no-deepeval" in lines[-1]
    assert fakes.made == {}  # refused before any provider was built or called


def test_a_full_run_with_deepeval_when_the_evals_group_is_installed(tmp_path):
    pytest.importorskip("deepeval")
    manifest, run_dir, env = setup_private(tmp_path)
    code, lines = _main(["run"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_OK, lines
    assert json.loads((run_dir / runner.RESULT_FILE).read_text())["deepeval"] is True
    assert list((run_dir / "deepeval").glob("*/test_run_*.json"))


# ---------------------------------------------------------------------------
# resume, stops, money
# ---------------------------------------------------------------------------


def test_interrupt_then_continue_gives_identical_outputs_and_sends_nothing_twice(tmp_path):
    _code, _lines, _m, clean_dir, _env, _f = _run(tmp_path / "clean")
    manifest, run_dir, env = setup_private(tmp_path / "cut")
    fakes = Fakes(**{REF: {}})
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).interrupt_at = 3
    code, lines = _main(
        ["run", "--no-deepeval", "--run-id", "r1", "--date", "2026-09-30"],
        manifest,
        run_dir,
        env,
        fakes,
    )
    assert code == runner.EXIT_INTERRUPTED
    code, lines = _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines
    assert sorted(fakes.get(REF).sent) == ["c-1", "c-2", "c-3", "c-4", "c-5-nocand"]
    for name in (runner.RESULT_FILE, runner.PAGE_FILE):
        assert (run_dir / name).read_text() == (clean_dir / name).read_text()


def test_a_call_in_flight_at_a_crash_is_resent_with_an_uncertain_charge(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).crash_after_send_at = 2
    code, _ = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_INTERRUPTED
    state = json.loads((run_dir / runner.RUN_FILE).read_text())
    assert len(state["reserved"]) == 1  # the in-flight marker survived the crash
    code, lines = _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines
    state = json.loads((run_dir / runner.RUN_FILE).read_text())
    assert len(state["uncertain_charges"]) == 1
    assert state["reserved"] == {}
    assert fakes.get(REF).sent.count("c-2") == 2
    assert any("uncertain charge" in line for line in lines)


def test_budget_cap_reserves_the_worst_case_before_sending(tmp_path):
    # One call's worst case (prompt + the whole 512-token output budget) is about
    # $0.0014; spent + reserved + estimate may never pass the cap.
    manifest, run_dir, env = setup_private(tmp_path, openrouter_cap=0.003)
    fakes = Fakes()
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    assert any("budget_cap_reached" in line for line in lines)
    sent = len(fakes.get(REF).sent)
    assert 0 < sent < 5
    status = runner.status(run_dir)
    assert status["providers"]["openrouter"]["spend_usd"] <= 0.003
    assert status["providers"]["openrouter"]["pending"] == 5 - sent
    assert "budget_cap" in json.dumps(status["stops"])


def test_money_stop_on_one_provider_lets_the_other_finish_then_continue_retries(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=EXTRA_LOCAL)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).fail = {2: "402"}
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    assert any("openrouter: stopping cleanly (insufficient_credit)" in line for line in lines)
    assert sorted(fakes.get("local/fake-cut").sent) == ["c-1", "c-2", "c-3", "c-4", "c-5-nocand"]
    assert "money" in json.dumps(runner.status(run_dir)["stops"])
    code, lines = _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines


def test_rejected_request_stops_only_that_model_and_is_a_capability(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=EXTRA_LOCAL)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "local", "model": "fake-cut"}), None, env)
    fakes.get("local/fake-cut").fail_always = "400"
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    state = json.loads((run_dir / runner.RUN_FILE).read_text())
    assert state["stops"]["models"]["local/fake-cut"]["kind"] == "rejected"
    assert state["capabilities"]["local/fake-cut"][0]["request_rejected"] == (
        "request_rejected:bad_request"
    )
    assert len(fakes.get(REF).sent) == 5
    # Still rejected while the same parameters are sent; retried on request.
    code, lines = _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    fakes.get("local/fake-cut").fail_always = None
    code, lines = _main(
        ["continue", "--no-deepeval", "--retry-rejected"], manifest, run_dir, env, fakes
    )
    assert code == runner.EXIT_OK, lines


def test_truncation_stop_stops_only_the_cut_model(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=EXTRA_LOCAL)
    fakes = Fakes(**{"local/fake-cut": {"cut": True}})
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    stop = json.loads((run_dir / runner.RUN_FILE).read_text())["stops"]["models"]
    assert stop["local/fake-cut"]["kind"] == "truncation"
    assert stop["local/fake-cut"]["truncated"] >= 3
    assert "max_output_tokens=64" in (stop["local/fake-cut"]["message"])
    # concurrency_cap 2: at most one more call was already in flight when the stop landed.
    assert 3 <= len(fakes.get("local/fake-cut").sent) <= 4
    assert len(fakes.get(REF).sent) == 5


def test_a_truncated_reply_is_invalid_under_every_policy(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=EXTRA_LOCAL)
    manifest.write_text(manifest.read_text().replace("min_answers = 3", "min_answers = 50"))
    fakes = Fakes(**{"local/fake-cut": {"cut": True}})
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines
    result = json.loads((run_dir / runner.RESULT_FILE).read_text())
    cut = next(r for r in result["reference_rows"] if r["subject"].startswith("local."))
    assert cut["right"]["n"] == 0
    assert cut["coverage"]["n"] == 0
    smoke_dir = tmp_path / "smoke"
    code, _ = _main(["smoke", "--cases", "2"], manifest, smoke_dir, env, fakes)
    smoke = json.loads((smoke_dir / runner.SMOKE_FILE).read_text())
    assert smoke["models"]["local/fake-cut"]["flag"] == "CAPPED"


def test_a_transient_stop_pauses_the_provider_and_continue_finishes(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).fail = {1: "429"}
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_STOPPED
    assert any("rate_limited" in line and "transient" in line for line in lines)
    code, lines = _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines


# ---------------------------------------------------------------------------
# records, scope and configuration
# ---------------------------------------------------------------------------


def test_run_record_lists_every_host_that_received_case_text(tmp_path):
    _code, _lines, _m, run_dir, _env, _f = _run(tmp_path, extra_refs=EXTRA_LOCAL)
    hosts = json.loads((run_dir / runner.RUN_FILE).read_text())["hosts"]
    assert hosts == {"local.invalid": ["local/fake-cut"], "openrouter.invalid": [REF]}
    lines = runner.render_status(runner.status(run_dir))
    assert any("hosts that received case text" in line for line in lines)


def test_status_cli_prints_counts_and_json(tmp_path):
    _code, _lines, _m, run_dir, _env, _f = _run(tmp_path)
    out = []
    assert main(["status", "--run-dir", str(run_dir), "--json"], out=out.append) == 0
    doc = json.loads(out[0])
    assert doc["models"][REF]["done"] == 5
    assert doc["mode"] == "full"
    out = []
    assert main(["status", "--run-dir", str(run_dir)], out=out.append) == 0
    assert out[0].startswith("run r1 (2026-09-30) [full]: complete")
    assert main(["status", "--run-dir", str(tmp_path / "none")], out=out.append) == 1


def test_smoke_scope_is_kept_on_continue_and_expanded_on_request(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    assert _main(["smoke", "--cases", "2"], manifest, run_dir, env, fakes)[0] == 0
    assert _main(["continue", "--no-deepeval"], manifest, run_dir, env, fakes)[0] == 0
    assert fakes.get(REF).sent == ["c-1", "c-2"]
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    assert "--expand" in lines[-1]
    code, lines = _main(["smoke", "--cases", "3"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    code, lines = _main(["run", "--no-deepeval", "--expand"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_OK, lines
    assert sorted(fakes.get(REF).sent) == ["c-1", "c-2", "c-3", "c-4", "c-5-nocand"]
    code, lines = _main(["smoke", "--cases", "2"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    assert "full run" in lines[-1]


def test_run_refuses_a_second_start_and_continue_needs_a_run(tmp_path):
    code, _lines, manifest, run_dir, env, fakes = _run(tmp_path)
    assert code == 0
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    assert "continue" in lines[-1]
    code, lines = _main(["continue", "--no-deepeval"], manifest, tmp_path / "empty", env, fakes)
    assert code == runner.EXIT_USER
    assert "holds no run" in lines[-1]


@pytest.mark.behavioral("o32")
def test_run_dir_inside_a_git_worktree_is_refused_and_an_empty_git_dir_is_not(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    manifest, _run_dir, env = setup_private(tmp_path / "a")
    repo = tmp_path / "repo"
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    code, lines = _main(["run", "--no-deepeval"], manifest, repo / "run", env, Fakes())
    assert code == runner.EXIT_USER
    assert "git worktree" in lines[-1]
    plain = tmp_path / "plain"
    (plain / ".git").mkdir(parents=True)
    code, lines = _main(["run", "--no-deepeval"], manifest, plain / "run", env, Fakes())
    assert code == runner.EXIT_OK, lines


def test_configuration_errors_are_user_or_environment_errors(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    fakes = Fakes()
    assert _main(["run", "--no-deepeval"], tmp_path / "missing.toml", run_dir, env, fakes)[0] == 1
    assert _main(["run", "--no-deepeval"], manifest, run_dir, {}, fakes)[0] == runner.EXIT_ENV
    manifest.write_text(manifest.read_text().replace("count = 5", "count = 6"))
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    assert "the file has 5" in lines[-1]
    state = (
        json.loads((run_dir / runner.RUN_FILE).read_text() or "{}")
        if (run_dir / runner.RUN_FILE).exists()
        else {}
    )
    assert not state.get("started_full")  # a configuration error never marks it started
    code, lines = _main(["smoke", "--cases", "0"], manifest, run_dir, env, fakes)
    assert code == runner.EXIT_USER
    out = []
    assert main(["run"], env=env, out=out.append) == runner.EXIT_USER
    assert "--run-dir is required" in out[-1]


def test_unknown_policy_domain_and_world_are_refused(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    text = manifest.read_text()
    manifest.write_text(text.replace('"mutating-strict-example"', '"nope"'))
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_USER
    assert "unknown policy" in lines[-1]
    manifest.write_text(text.replace("tests.fixtures.toy_domain", "no.such.domain"))
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_USER
    assert "domain" in lines[-1]
    (Path(env["JEV_EVALS_PRIVATE_ROOT"]) / "world.json").write_text('{"home": 3}')
    manifest.write_text(text)
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_USER
    assert "world snapshot" in lines[-1]


def test_a_policy_file_under_the_private_root_is_a_harness_row(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    policy = {"name": "bundle", "version": "1", "calibration": None, "gate": None}
    (Path(env["JEV_EVALS_PRIVATE_ROOT"]) / "bundle.json").write_text(json.dumps(policy))
    manifest.write_text(manifest.read_text().replace('"mutating-strict-example"', '"bundle.json"'))
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_OK, lines
    result = json.loads((run_dir / runner.RESULT_FILE).read_text())
    assert {r["harness_policy"] for r in result["rows"]} == {None, "bundle.json"}


def test_training_overlap_refuses_to_score_a_checkpoint(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    root = Path(env["JEV_EVALS_PRIVATE_ROOT"])
    (root / "train.json").write_text(json.dumps({"entries": [{"id": "c-2"}]}))
    manifest.write_text(
        manifest.read_text().replace(
            'revision = "rev-1"', 'revision = "rev-1"\ntrain_split = "train.json"'
        )
    )
    code, lines = _main(["run", "--no-deepeval"], manifest, run_dir, env, Fakes())
    assert code == runner.EXIT_USER
    assert "c-2" in lines[-1]


def test_json_output_is_one_document(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path)
    out = []
    code = main(
        ["smoke", "--cases", "1", "--json", "--manifest", str(manifest), "--run-dir", str(run_dir)],
        factory=Fakes(),
        env=env,
        out=out.append,
    )
    assert code == 0
    assert len(out) == 1
    assert json.loads(out[0])["status"] == "complete"


def test_default_factory_builds_the_adapter_without_reading_keys(monkeypatch):
    from jev_factory.evals.manifest import Budget, Reference

    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    ref = Reference(provider="local", model="m", capabilities=("logprobs",))
    budget = Budget(provider="local", usd_cap=0.0, concurrency_cap=1, requests_per_minute=30)
    provider = runner.default_factory(
        ref, budget, {"JEV_EVALS_BASE_URL_LOCAL": "http://127.0.0.1:9/v1"}
    )
    assert provider.capabilities.logprobs
    assert provider.host == "127.0.0.1"
    assert runner.shared_limiter("local", 30) is runner.shared_limiter("local", 30)
    assert runner.provider_reasoning("openrouter", "none") == "none"
    assert runner.provider_reasoning("local", "none") is None
    assert runner.provider_reasoning("local", "high") == "high"


def test_classify_exception_maps_adapter_failures():
    from jev_factory.evals.providers import openai_compat
    from jev_factory.evals.providers.base import HeldoutSplitRefused, MissingProviderKey

    assert runner.classify_exception(HeldoutSplitRefused("x"), "local") is None
    assert runner.classify_exception(MissingProviderKey("x"), "local").reason == "missing_key"
    assert (
        runner.classify_exception(openai_compat.TransportError("x"), "local").reason
        == "network_loss"
    )
    unexpected = runner.classify_exception(ValueError("x"), "local")
    assert unexpected.reason == "unexpected_error:ValueError"
    assert unexpected.retryable

    class HTTPish(Exception):
        status = 429
        body = b'{"error": {"code": "insufficient_quota"}}'

    assert runner.classify_exception(HTTPish(), "local").reason == "insufficient_credit"
    assert runner.refused_before_send(ConnectionRefusedError()) is True
    assert runner.refused_before_send(TimeoutError()) is False


def _clocked(tmp_path, clock, fail=None, extra_refs=""):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=extra_refs)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).fail = fail or {}
    lines: list[str] = []
    code = main(
        ["run", "--no-deepeval", "--manifest", str(manifest), "--run-dir", str(run_dir)],
        factory=fakes,
        env=env,
        out=lines.append,
        clock=clock,
    )
    return code, lines, run_dir, fakes


def test_a_rate_limit_pause_expires_within_the_pass(tmp_path):
    now = [1000.0]

    def clock():
        now[0] += 31.0  # every read moves time on: two reads outlast the 60 s pause
        return now[0]

    code, lines, run_dir, fakes = _clocked(tmp_path, clock, {1: "429"})
    assert code == runner.EXIT_OK, lines
    assert fakes.get(REF).sends == 6  # the rate-limited call, then all five
    assert any("rate_limited" in line for line in lines)
    assert (run_dir / runner.RESULT_FILE).exists()


def test_a_paused_provider_stays_paused_until_the_pause_runs_out(tmp_path):
    code, _lines, run_dir, _fakes = _clocked(tmp_path, lambda: 1000.0, {1: "429"})
    assert code == runner.EXIT_STOPPED
    assert runner.status(run_dir)["providers"]["openrouter"]["pending"] > 0


def test_a_sync_round_stops_claiming_after_its_time_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROUND_SECONDS", -1.0)  # every round is already over
    code, _lines, run_dir, fakes = _clocked(tmp_path, lambda: 1000.0)
    assert code == runner.EXIT_STOPPED
    assert fakes.get(REF).sends == 0  # nothing was claimed past the deadline
    assert runner.status(run_dir)["providers"]["openrouter"]["pending"] == 5


def test_interleave_takes_one_call_per_model_in_turn():
    a = type("M", (), {"label": "a"})()
    b = type("M", (), {"label": "b"})()
    work = [(a, 1), (a, 2), (a, 3), (b, 4)]
    assert [item for _m, item in runner.interleave(work)] == [1, 4, 2, 3]
