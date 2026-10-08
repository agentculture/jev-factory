"""The progress standard for long-running work (deviation d15): timed progress files and
``jev status --watch``, an update every interval (30 minutes by default) until nothing runs."""

from __future__ import annotations

import io
import json
import os

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.cli._commands import status
from jev_factory.cli._errors import CliError
from jev_factory.factory import detach


def _progress(jobs, name, done, total, *, started, updated, pid, start_done=0):
    jobs.mkdir(parents=True, exist_ok=True)
    doc = {"done": done, "total": total, "started": started, "start_done": start_done}
    doc.update(updated=updated, pid=pid)
    (jobs / f"{name}.progress.json").write_text(json.dumps(doc))


def test_progress_files_carry_timing_and_give_a_rate_and_an_eta(tmp_path):
    progress = detach.Progress(tmp_path, "augment", total=10, done=4)  # resumed at 4
    progress.advance(2)
    record = detach.read_progress(tmp_path, "augment")
    assert (record["done"], record["total"], record["start_done"]) == (6, 10, 4)
    assert record["pid"] == os.getpid()
    assert record["updated"] >= record["started"]
    timed = {"done": 6, "total": 10, "start_done": 4, "started": 100.0, "updated": 160.0}
    rate, eta = detach.rate_and_eta(timed)  # 2 items this session in 60 s
    assert rate == pytest.approx(2 / 60)
    assert eta == pytest.approx(120.0)
    assert detach.rate_and_eta({"done": 3, "total": 6}) == (None, None)  # an old file


def test_an_in_process_job_is_running_complete_or_stopped_from_its_progress(tmp_path):
    _progress(tmp_path, "live", 1, 5, started=1.0, updated=2.0, pid=os.getpid())
    _progress(tmp_path, "full", 5, 5, started=1.0, updated=2.0, pid=os.getpid())
    _progress(tmp_path, "gone", 1, 5, started=1.0, updated=2.0, pid=2**22 + 12345)
    states = {n: detach.job_status(tmp_path, n)["state"] for n in ("live", "full", "gone")}
    assert states == {"live": "running", "full": "complete", "gone": "stopped"}


@pytest.mark.parametrize(
    "text, seconds", [("30m", 1800), ("1h", 3600), ("90s", 90), ("90", 90), ("1.5m", 90)]
)
def test_every_takes_a_duration(text, seconds):
    assert status.parse_every(text) == seconds


@pytest.mark.parametrize("text", ["soon", "0s", "5d"])
def test_a_bad_interval_is_refused(text):
    with pytest.raises(CliError):
        status.parse_every(text)


def test_the_default_interval_is_thirty_minutes():
    assert status.DEFAULT_EVERY_SECONDS == 30 * 60


def test_watch_updates_every_interval_until_nothing_runs(tmp_path):
    jobs = tmp_path / "jobs"
    pid = os.getpid()
    steps = iter([(3, 10), (7, 10), (10, 10)])
    slept: list[float] = []
    clock = iter(range(1000, 100000, 1800))

    def advance(seconds: float) -> None:
        slept.append(seconds)
        done, total = next(steps)
        _progress(jobs, "augment", done, total, started=0.0, updated=60.0 * done, pid=pid)

    _progress(jobs, "augment", 0, 10, started=0.0, updated=0.0, pid=pid)
    advance(0)
    slept.clear()
    out = io.StringIO()
    rc = status.watch(tmp_path, 1800, sleep=advance, clock=lambda: next(clock), stream=out)
    blocks = out.getvalue().strip().split("\n[")
    assert rc == 0
    assert slept == [1800, 1800]
    assert len(blocks) == 3
    assert "augment: running, 3/10 items, 1.0/min, ETA 7m00s" in blocks[0]
    assert "since last update: augment: +4 items" in blocks[1]
    assert "final update" in blocks[2]
    assert "augment: running -> complete" in blocks[2]


def test_watch_with_json_emits_one_update_per_line(tmp_path):
    _progress(tmp_path / "jobs", "draft", 6, 6, started=0.0, updated=60.0, pid=os.getpid())
    out = io.StringIO()
    assert status.watch(tmp_path, 1800, json_mode=True, sleep=lambda s: None, stream=out) == 0
    (line,) = out.getvalue().splitlines()
    upd = json.loads(line)
    assert upd["running"] is False
    assert upd["jobs"][0]["job"] == "draft"
    assert upd["jobs"][0]["rate_per_min"] == pytest.approx(6.0)


def test_every_without_watch_is_a_user_error(tmp_path, capsys):
    assert cli_main(["status", str(tmp_path), "--every", "30m"]) == 1
    assert "--every needs --watch" in capsys.readouterr().err


def test_a_one_shot_status_lists_jobs_that_belong_to_no_stage(tmp_path):
    _progress(tmp_path / "jobs", "measure-final-test", 2, 8, started=0.0, updated=1.0, pid=1)
    text = status.render(status.collect(tmp_path))
    assert "jobs:" in text
    assert "measure-final-test" in text
    assert "2/8 items" in text
