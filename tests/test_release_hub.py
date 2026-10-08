"""release/hub.py: private upload with a fake hub client (no network, no token)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.release import hub as hub_module
from jev_factory.release.hub import UploadError, check_inventory, compare, upload
from tests.fixtures.toy_domain import DOMAIN
from tests.release_support import HUB_PREFIX, REPO, TOKEN, FakeHub, make_config, scanned_bundle

ENV = {"HF_TOKEN": TOKEN}


def _upload(tmp_path: Path, hub: FakeHub, **overrides):
    kwargs = dict(
        bundle=overrides.pop("bundle", None) or scanned_bundle(tmp_path),
        repo=REPO,
        repo_type="model",
        config=make_config(),
        domain=DOMAIN,
        apply=True,
        hub=hub,
        environ=ENV,
    )
    kwargs.update(overrides)
    return upload(**kwargs)


@pytest.mark.behavioral("o19")
def test_without_apply_nothing_is_sent(tmp_path, monkeypatch):
    hub = FakeHub(tmp_path / "remote")
    monkeypatch.setattr(hub_module.importlib, "import_module", lambda n: pytest.fail(n))
    result = _upload(tmp_path, hub, apply=False, environ={})  # not even a token is read
    assert hub.calls == []
    assert result["applied"] is False
    assert not (tmp_path / "remote").exists()


@pytest.mark.behavioral("o19")
def test_apply_creates_private_forces_private_fetches_back_and_compares(tmp_path, capsys):
    hub = FakeHub(tmp_path / "remote")
    result = _upload(tmp_path, hub)
    names = [call[0] for call in hub.calls]
    assert names == [
        "HfApi",
        "create_repo",
        "update_repo_visibility",
        "upload_folder",
        "snapshot_download",
        "list_repo_files",
        "repo_info",
    ]
    assert hub.calls[0] == ("HfApi", True)
    assert hub.calls[1][2]["private"] is True
    assert hub.calls[1][2]["exist_ok"] is True
    assert hub.calls[2][2]["private"] is True
    assert hub.calls[4][2]["revision"] == "c0ffee"
    assert result["private"] is True
    assert result["applied"] is True
    assert result["files"] == 8  # README, weights, tokenizer, 4 required files, scan.json
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert not list((tmp_path / "bundles").glob(".fetch-*"))
    json.dumps(result)


@pytest.mark.behavioral("o19")
def test_a_changed_file_or_a_stale_remote_file_fails_loudly(tmp_path):
    def flip(local_dir: Path) -> None:
        path = local_dir / "model.safetensors"
        data = bytearray(path.read_bytes())
        data[1] ^= 0xFF
        path.write_bytes(bytes(data))

    flipping = FakeHub(tmp_path / "r1", tamper=flip)
    with pytest.raises(UploadError, match="model.safetensors: sha256 differs"):
        _upload(tmp_path, flipping)
    dropping = FakeHub(tmp_path / "r2", tamper=lambda d: (d / "sub" / "tokenizer.json").unlink())
    bundle = scanned_bundle(tmp_path / "x")
    with pytest.raises(UploadError, match="sub/tokenizer.json: missing"):
        _upload(tmp_path / "x", dropping, bundle=bundle)
    hub = FakeHub(tmp_path / "r3")
    (tmp_path / "r3" / REPO).mkdir(parents=True)
    (tmp_path / "r3" / REPO / "old-weights.bin").write_bytes(b"old")
    bundle = scanned_bundle(tmp_path / "y")
    with pytest.raises(UploadError, match="old-weights.bin"):
        _upload(tmp_path / "y", hub, bundle=bundle)


@pytest.mark.behavioral("o19")
def test_an_extra_remote_file_under_cache_fails_the_inventory(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    stale = tmp_path / "remote" / REPO / ".cache"
    stale.mkdir(parents=True)
    (stale / "held-out.json").write_text("{}")
    with pytest.raises(UploadError, match=r"\.cache/held-out\.json"):
        _upload(tmp_path, hub)


@pytest.mark.behavioral("o19")
def test_a_symlink_in_the_bundle_is_refused_before_any_hub_call(tmp_path):
    outside = tmp_path / "sealed.json"
    outside.write_text('{"entries": []}')
    bundle = scanned_bundle(tmp_path)
    (bundle / "extra.json").symlink_to(outside)
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(UploadError, match="symlink"):
        _upload(tmp_path, hub, bundle=bundle)
    assert hub.calls == []


@pytest.mark.behavioral("o19")
def test_no_visibility_to_public_call_is_ever_made(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    _upload(tmp_path, hub)
    for call in hub.calls:
        if call[0] in ("create_repo", "update_repo_visibility", "update_repo_settings"):
            assert call[2]["private"] is True
    source = inspect.getsource(hub_module)
    assert "private=False" not in source
    assert "make_public" not in source
    assert "visibility=" not in source
    # a hub reporting public after the upload fails rather than being "fixed" to public
    public_hub = FakeHub(tmp_path / "rp", private=False)
    bundle = scanned_bundle(tmp_path / "p")
    with pytest.raises(UploadError, match="not private"):
        _upload(tmp_path / "p", public_hub, bundle=bundle)


def test_a_1x_hub_is_made_private_with_update_repo_settings():
    calls = []

    class Api:
        def update_repo_settings(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))

    hub_module._set_private(Api(), REPO, "model")  # noqa: SLF001
    assert calls == [(REPO, {"private": True, "repo_type": "model"})]


@pytest.mark.parametrize(
    "repo",
    [
        "someone-else/toy-lamps-jev-scorer",
        HUB_PREFIX,
        HUB_PREFIX + "Scorer",
        HUB_PREFIX + "a/b",
        HUB_PREFIX + "../x",
        HUB_PREFIX + "scorer ",
    ],
)
def test_a_repo_outside_the_hub_prefix_is_refused(tmp_path, repo):
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(UploadError, match="uploads go only to"):
        _upload(tmp_path, hub, repo=repo)
    assert hub.calls == []


def test_the_hub_prefix_comes_from_the_run_config_then_the_domain(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    cfg = make_config(hub_prefix="acme/fam-")
    with pytest.raises(UploadError, match="acme/fam-"):
        _upload(tmp_path, hub, config=cfg)  # the Domain's prefix does not override the config
    _upload(tmp_path, hub, config=cfg, repo="acme/fam-x")
    # no config prefix: the Domain's applies
    hub2 = FakeHub(tmp_path / "remote2")
    _upload(
        tmp_path / "d",
        hub2,
        config=make_config(hub_prefix=""),
        bundle=scanned_bundle(tmp_path / "d"),
    )
    no_prefix = make_config(hub_prefix="")
    with pytest.raises(UploadError, match="no hub_prefix"):
        _upload(tmp_path, hub2, config=no_prefix, domain=None)


def test_an_unset_token_is_a_named_error_and_nothing_is_sent(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(CliError, match="HF_TOKEN"):
        _upload(tmp_path, hub, environ={})
    assert hub.calls == []


def test_a_bundle_changed_after_its_scan_or_missing_required_files_is_refused(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    bundle = scanned_bundle(tmp_path)
    (bundle / "README.md").write_text("changed")
    with pytest.raises(UploadError, match="changed after scan"):
        _upload(tmp_path, hub, bundle=bundle)
    bundle2 = scanned_bundle(tmp_path / "m")
    (bundle2 / "gate.json").unlink()
    with pytest.raises(UploadError, match="missing gate.json"):
        _upload(tmp_path, hub, bundle=bundle2)
    assert hub.calls == []


def test_an_unknown_repo_type_is_refused(tmp_path):
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(UploadError, match="repo type"):
        _upload(tmp_path, hub, repo_type="space")


def test_a_hub_failure_never_echoes_the_token(tmp_path):
    hub = FakeHub(tmp_path / "remote")

    def boom(*a, **k):
        raise RuntimeError(f"401 for {TOKEN}")

    hub.snapshot_download = boom
    with pytest.raises(UploadError) as err:
        _upload(tmp_path, hub)
    assert TOKEN not in str(err.value)


def test_compare_ignores_the_download_cache_and_a_hub_gitattributes(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "a.txt").write_text("a")
    remote = tmp_path / "remote"
    (remote / ".cache" / "huggingface").mkdir(parents=True)
    (remote / "a.txt").write_text("a")
    (remote / ".gitattributes").write_text("x")
    (remote / ".cache" / "huggingface" / "x.lock").write_text("")
    assert compare(local, remote) == []
    (remote / "a.txt").write_text("b")
    assert compare(local, remote) == ["a.txt: sha256 differs"]
    assert check_inventory(local, [".gitattributes", "a.txt"]) == []
