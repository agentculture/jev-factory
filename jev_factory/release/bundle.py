"""Build and check model and dataset bundles: the private upload folders of a run.

A *model bundle* carries a tuned checkpoint (``bf16``, a ``gguf`` build or an
``awq`` export), the base model's ``LICENSE``, a ``NOTICE`` stating the
modification, and a model card. A *dataset bundle* carries the records the model
was trained and measured on (see :mod:`jev_factory.release.dataset_bundle`).
Every bundle, of either kind, must also carry

- ``calibration.json`` and ``gate.json``: the frozen calibration parameters and gate
  settings the reported figures use (nvsh copied these by hand);
- ``scorer-train.json``: the candidate set the scorer actually trained on;
- ``bundle.json``: the bundle's identity, including the **surface hash** of the
  domain the model was trained on. :func:`surface_mismatch` compares it with the
  running CLI's surface, which is what ``jev ask`` reports.

A bundle holding a symlink, or a GGUF whose metadata carries an absolute path (an
``imatrix`` file path would leak the builder's directory layout), is refused.
The hub prefix, licence and card prose come from the run config and the Domain, never
from constants here. Nothing in this module uploads; see :mod:`jev_factory.release.hub`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jev_factory.backbones.causal_lm.readout import READOUT_TOP
from jev_factory.domain.model import Domain
from jev_factory.factory.config import RunConfig
from jev_factory.release import dataset_bundle as _dataset
from jev_factory.release.scan import check_no_symlinks

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/release_bundle.py",
    "commit": "9debdc6",
    "adaptations": [
        "_sibling() importlib path-loading -> normal package imports; stage_cache"
        " chat_template/revision_of and drop_undeclared_mtp/_tensor_names ported here",
        "LFM Open License mode and issue-46 card prose dropped: licence and card text come"
        " from the run config and Domain.card_text",
        "calibration.json, gate.json and scorer-train.json are required; bundle.json records"
        " the domain surface sha256 (new)",
        "GGUF metadata reader added; an absolute path or a symlink refuses the bundle (new)",
        "the argparse main() is dropped",
    ],
    "licence": "Apache-2.0",
}

APACHE_LICENCE = "Apache-2.0"
KINDS = ("bf16", "gguf", "awq")
BUNDLE_FILE = "bundle.json"
CALIBRATION_FILE = "calibration.json"
GATE_FILE = "gate.json"

REQUIRED_FILES = (CALIBRATION_FILE, GATE_FILE, "scorer-train.json")
TOKENIZER_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_CONFIG = "tokenizer_config.json"


class BundleError(ValueError):
    """A bundle that cannot be built, or that fails a check and must not ship."""


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

_GGUF_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
_GGUF_MAX_STRING = 1 << 28
_ABSOLUTE_RE = re.compile(r"^(?:/|\\\\|~|[A-Za-z]:[\\/])")
_PATH_LIKE_RE = re.compile(r"^(?:/|[A-Za-z]:[\\/])\S*$")


def _read(handle, fmt: str):
    size = struct.calcsize(fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise BundleError("truncated GGUF header")
    return struct.unpack(fmt, raw)[0]


def _read_string(handle) -> str:
    length = _read(handle, "<Q")
    if length > _GGUF_MAX_STRING:
        raise BundleError("GGUF metadata string length is not plausible")
    raw = handle.read(length)
    if len(raw) != length:
        raise BundleError("truncated GGUF header")
    return raw.decode("utf-8", errors="replace")


def _read_value(handle, value_type: int):
    if value_type in _GGUF_SCALARS:
        return _read(handle, _GGUF_SCALARS[value_type])
    if value_type == _GGUF_STRING:
        return _read_string(handle)
    if value_type == _GGUF_ARRAY:
        item_type = _read(handle, "<I")
        count = _read(handle, "<Q")
        if item_type in _GGUF_SCALARS:
            handle.seek(struct.calcsize(_GGUF_SCALARS[item_type]) * count, 1)
        elif item_type == _GGUF_STRING:
            for _ in range(count):
                handle.seek(_read(handle, "<Q"), 1)
        else:
            raise BundleError(f"unsupported GGUF array item type {item_type}")
        return None  # arrays (token lists) are skipped, not returned
    raise BundleError(f"unsupported GGUF value type {value_type}")


def gguf_metadata(path: Path) -> dict[str, Any]:
    """The scalar key/value metadata of a GGUF file (arrays are skipped).

    Raises :class:`BundleError` when the file is not a GGUF v1-3 header."""
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                raise BundleError(f"{path.name}: no GGUF magic")
            version = _read(handle, "<I")
            if version not in (1, 2, 3):
                raise BundleError(f"{path.name}: unsupported GGUF version {version}")
            _read(handle, "<I" if version == 1 else "<Q")  # tensor count
            count = _read(handle, "<I" if version == 1 else "<Q")
            meta: dict[str, Any] = {}
            for _ in range(count):
                key = _read_string(handle)
                meta[key] = _read_value(handle, _read(handle, "<I"))
            return meta
    except OSError as exc:
        raise BundleError(f"{path.name}: cannot read GGUF metadata ({exc.strerror})") from None
    except (struct.error, ValueError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"{path.name}: unreadable GGUF metadata ({exc})") from None


def gguf_absolute_paths(path: Path) -> list[str]:
    """The key of every metadata string that holds an absolute path.

    Every ``quantize.imatrix.*`` string is held to the rule; any other string value that
    looks like a whole filesystem path is refused too (tokenizer keys are exempt: their
    template text is prose, not a path)."""
    bad: list[str] = []
    for key, value in gguf_metadata(path).items():
        if not isinstance(value, str) or key.startswith("tokenizer."):
            continue
        if key.startswith("quantize.imatrix"):
            if _ABSOLUTE_RE.match(value.strip()):
                bad.append(key)
        elif _PATH_LIKE_RE.match(value.strip()):
            bad.append(key)
    return bad


# ---------------------------------------------------------------------------
# Whole-bundle checks
# ---------------------------------------------------------------------------


def bundle_problems(folder: Path) -> list[str]:
    """Every reason *folder* must not ship as a bundle (empty when it may)."""
    if not folder.is_dir():
        return [f"{folder} is not a folder"]
    problems = [
        f"symlink {link}: a bundle never holds a link" for link in check_no_symlinks(folder)
    ]
    problems.extend(
        f"missing {name}"
        for name in (*REQUIRED_FILES, BUNDLE_FILE)
        if not (folder / name).is_file()
    )
    if (folder / BUNDLE_FILE).is_file() and not _records_surface(folder / BUNDLE_FILE):
        problems.append(f"{BUNDLE_FILE} records no surface_sha256")
    problems.extend(_gguf_problems(folder))
    return problems


def _records_surface(info_file: Path) -> bool:
    """Whether *info_file* (a ``bundle.json``) is an object naming a surface hash."""
    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return isinstance(info, dict) and bool(info.get("surface_sha256"))


def _gguf_problems(folder: Path) -> list[str]:
    """An absolute-path or unreadable-metadata problem per GGUF file in *folder*."""
    problems: list[str] = []
    for gguf in sorted(folder.rglob("*.gguf")):
        rel = gguf.relative_to(folder).as_posix()
        if gguf.is_symlink():
            continue  # already refused by bundle_problems; never follow it
        try:
            keys = gguf_absolute_paths(gguf)
        except BundleError as exc:
            problems.append(str(exc))
            continue
        problems.extend(f"{rel}: GGUF metadata {key} holds an absolute path" for key in keys)
    return problems


def check_bundle(folder: Path) -> None:
    """Raise :class:`BundleError` listing every problem :func:`bundle_problems` finds."""
    problems = bundle_problems(folder)
    if problems:
        raise BundleError(f"refusing bundle {folder}:\n  " + "\n  ".join(problems))


def read_bundle_info(folder: Path) -> dict[str, Any]:
    """The parsed ``bundle.json`` of *folder*."""
    try:
        return json.loads((folder / BUNDLE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BundleError(f"{folder} has no readable {BUNDLE_FILE}") from None


def surface_mismatch(folder: Path, current_sha256: str | None = None) -> str | None:
    """``None`` when the bundle's recorded surface hash equals the running CLI's, else a
    sentence saying how they differ (what ``jev ask`` reports).

    *current_sha256* defaults to the running jev CLI's surface hash."""
    recorded = read_bundle_info(folder).get("surface_sha256")
    if current_sha256 is None:
        from jev_factory.domains.jev_cli.generate import cli_surface_sha256

        current_sha256 = cli_surface_sha256()
    if recorded == current_sha256:
        return None
    return (
        f"bundle {folder.name} was trained on CLI surface {recorded}, but the running CLI's"
        f" surface is {current_sha256}: its candidates no longer match this CLI"
    )


def licence_of(domain: Domain, config: RunConfig) -> str:
    """The licence a bundle ships under: the run config's, else the Domain's, else Apache."""
    return str(config.get("licence") or domain.licence or APACHE_LICENCE)


_licence = licence_of


def _hub_prefix(domain: Domain, config: RunConfig) -> str:
    return str(config.get("hub_prefix") or domain.hub_prefix or "")


def _write_info(
    out: Path, *, kind: str, domain: Domain, config: RunConfig, repo: str, run: str
) -> dict[str, Any]:
    info = {
        "kind": kind,
        "domain": domain.name,
        "surface_sha256": domain.surface_sha256(),
        "licence": _licence(domain, config),
        "hub_prefix": _hub_prefix(domain, config),
        "repo": repo,
        "run": run,
    }
    (out / BUNDLE_FILE).write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return info


# ---------------------------------------------------------------------------
# Model-bundle helpers (ported from release_bundle.py / stage_cache.py)
# ---------------------------------------------------------------------------


def chat_template(directory: Path) -> str:
    """The chat template a model directory ships, or raise if it has none."""
    template_file = directory / _TEMPLATE_FILE
    if template_file.is_file():
        return template_file.read_text(encoding="utf-8")
    config = directory / _TOKENIZER_CONFIG
    if config.is_file():
        template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(template, str) and template:
            return template
    raise BundleError(f"{directory} ships no chat template")


def revision_of(directory: Path) -> str:
    """A stable 40-hex revision for the files in *directory* (names and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()[:40]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_table(lines: list[str], source: Path) -> str:
    table: list[str] = []
    for line in lines:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    if not table:
        raise BundleError(f"{source} holds no metric table")
    return "\n".join(table)


def results_section(results: Path, heading: str | None = None) -> tuple[str, str]:
    """``(caption, table)`` for one measurement report: the first table in the file, or
    (with *heading*, e.g. ``## Metrics``) the first table under that heading."""
    lines = results.read_text(encoding="utf-8").splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), results.stem)
    if heading is None:
        return title, _first_table(lines, results)
    if heading not in lines:
        raise BundleError(f"{results} has no '{heading}' section to quote")
    section: list[str] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("## "):
            break
        section.append(line)
    return title, _first_table(section, results)


def _tensor_names(folder: Path) -> list[str]:
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
    names: list[str] = []
    for path in sorted(folder.glob("*.safetensors")):
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        names.extend(name for name in header if name != "__metadata__")
    return names


def drop_undeclared_mtp(folder: Path) -> int | None:
    """Zero a declared multi-token-prediction head that *folder*'s weights do not hold.

    Returns the old ``mtp_num_hidden_layers`` when it changed ``config.json`` (written as a
    new file, never through a link), else ``None``. Call it on a bundle copy only."""
    path = folder / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    holders = [config]
    if isinstance(config.get("text_config"), dict):
        holders.append(config["text_config"])
    declared = [h for h in holders if int(h.get("mtp_num_hidden_layers") or 0) > 0]
    if not declared:
        return None
    if any(name.startswith("mtp.") or ".mtp." in name for name in _tensor_names(folder)):
        return None
    old = int(declared[0]["mtp_num_hidden_layers"])
    for holder in declared:
        holder["mtp_num_hidden_layers"] = 0
    path.unlink()
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return old


def _base_identity(base_snapshot: Path) -> tuple[str, str]:
    """``(owner/name, revision)`` from a cache path ``models--owner--name/snapshots/<rev>``."""
    revision = base_snapshot.name
    repo_folder = base_snapshot.parent.parent.name
    if base_snapshot.parent.name != "snapshots" or not repo_folder.startswith("models--"):
        raise BundleError(f"{base_snapshot} is not a Hugging Face cache snapshot")
    owner, _, name = repo_folder[len("models--") :].partition("--")
    return f"{owner}/{name}", revision


def _check_licence_file(licence_file: Path, licence: str) -> None:
    text = licence_file.read_text(encoding="utf-8")
    if not text.strip():
        raise BundleError(f"{licence_file} is empty")
    if licence == APACHE_LICENCE:
        head = [line for line in text.splitlines() if line.strip()][:5]
        if not (any("Apache License" in x for x in head) and any("Version 2.0" in x for x in head)):
            raise BundleError(f"{licence_file} does not appear to be the Apache License 2.0")


def notice(
    base_repo: str,
    base_revision: str,
    repo: str,
    *,
    licence: str,
    domain: Domain,
    kind: str = "bf16",
    mtp_from: int | None = None,
) -> str:
    """The NOTICE: a prominent statement that the files were modified, and how (every
    modification, including the quantization and any ``config.json`` change)."""
    extra = ""
    if kind == "gguf":
        extra += (
            "\nThis upload is a Q4_K_M GGUF quantization of that fine-tune, made with\n"
            "llama.cpp; the tokenizer files are the fine-tune's.\n"
        )
    elif kind == "awq":
        extra += (
            "\nThis upload is an INT4 AWQ quantization of that fine-tune in the\n"
            "compressed-tensors format, made with llm-compressor.\n"
        )
    if mtp_from is not None:
        extra += (
            f"\nconfig.json: mtp_num_hidden_layers was changed from {mtp_from} to 0, as the\n"
            "weights hold no multi-token-prediction (mtp.*) tensor.\n"
        )
    return (
        f"{repo}\n\n"
        f"This model is a derivative work based on {base_repo} (revision {base_revision}),\n"
        f"licensed under {licence}.\n\n"
        "Modifications: the weights were changed by LoRA fine-tuning, then merged, to teach the\n"
        f"model to score the candidate actions of the {domain.name} domain with one next-token\n"
        "read over their labels. The tokenizer and chat template are unchanged.\n"
        f"{extra}"
    )


def calibration_section() -> str:
    return (
        "\n## Calibration and gate\n\n"
        "`calibration.json` holds the temperature (and a per-label vector, all 1.0 when unused)\n"
        "fitted on the fit fold for this exact build: divide each offered label's\n"
        "log-probability by the temperature, apply the vector, and renormalise over the offered\n"
        "labels. `gate.json` holds the decision gate the reported figures use: on the calibrated\n"
        "distribution, escalate, explain, propose, or abstain when a threshold (top-1 floor,\n"
        "top-1/top-2 margin, entropy) is not met, with separate thresholds for read-only and\n"
        "mutating operations. `scorer-train.json` is the candidate set the scorer trained on.\n"
    )


def model_card(
    *,
    domain: Domain,
    licence: str,
    repo: str,
    base_repo: str,
    base_revision: str,
    run: str,
    results: Sequence[tuple[str, str, str]],
    data_summary: str,
    surface_sha256: str,
    teachers: _dataset.TeacherSummary | None = None,
    kind: str = "bf16",
    mtp_from: int | None = None,
    gguf_name: str | None = None,
    gguf_sha256: str | None = None,
    awq_serve_args: Sequence[str] = (),
    quantized_from: str | None = None,
) -> str:
    """The model card; its prose comes from ``domain.card_text`` and the run's licence."""
    if kind not in KINDS:
        raise BundleError(f"unknown bundle kind {kind!r} (one of {', '.join(KINDS)})")
    tags = {"bf16": "", "gguf": "- gguf\n", "awq": "- awq\n"}[kind]
    library = "" if kind == "gguf" else "library_name: transformers\n"
    build_note = ""
    if kind == "gguf":
        build_note = (
            f"\nThis repository holds the **Q4_K_M GGUF** build (`{gguf_name}`, sha256\n"
            f"`{gguf_sha256}`), plus the fine-tune's tokenizer files and chat template.\n"
        )
        use = (
            "Serve the GGUF with llama.cpp's `llama-server`. It is **text-only**: there is no\n"
            "vision projector, so pass no `--mmproj`. `--jinja` applies the chat template the\n"
            "GGUF carries and `--temp 0 --top-k 1` pins greedy decoding:\n\n"
            f"```bash\nllama-server --model {gguf_name} --jinja --temp 0 --top-k 1 \\\n"
            "  --host 127.0.0.1 --port 8080\n```\n"
        )
    elif kind == "awq":
        build_note = "\nThis repository holds the **INT4 AWQ** build (compressed-tensors).\n"
        args = shlex.join(awq_serve_args)
        use = (
            "Serve the compressed-tensors folder with vLLM "
            f"(`{args}`), with `--max-logprobs {READOUT_TOP}` so the label scores can be read.\n"
        )
    else:
        use = (
            f"Serve with vLLM with `--max-logprobs {READOUT_TOP}` and read the next-token\n"
            "log-probabilities over the labels.\n"
        )
    if quantized_from:
        build_note += f"The bf16 fine-tune it was quantized from is `{quantized_from}`.\n"
    if mtp_from is not None:
        build_note += (
            f"\n`config.json`: `mtp_num_hidden_layers` is set from {mtp_from} to 0 in this"
            " upload; the weights hold no `mtp.*` tensor.\n"
        )
    quoted = "\n\n".join(f"### {cap}\n\nFrom `{name}`:\n\n{table}" for cap, name, table in results)
    if teachers is not None and teachers.role_teachers:
        rows = "\n".join(f"| {n} | {lic} | {d} |" for n, lic, d in _dataset.teacher_rows(teachers))
        teacher_block = (
            f"\n{_dataset.decision_sentence(teachers.decisions)}\n\n"
            f"| Model | Licence | Role |\n|---|---|---|\n{rows}\n\n"
            "The teachers' licences do not carry over to their outputs.\n"
        )
    else:
        teacher_block = "\nNo synthetic variations were used in this run's training set.\n"
    return f"""---
{library}license: {licence.lower()}
base_model: {base_repo}
pipeline_tag: text-generation
tags:
{tags}- jev
- candidate-scoring
- {domain.name}
---

# {repo.split("/")[-1]}

{domain.card_text}

A fine-tune of [{base_repo}](https://huggingface.co/{base_repo}) (revision `{base_revision}`)
as a **jev-like candidate scorer**: every candidate is listed in the prompt under a
one-letter label and the model's next-token log-probabilities over those labels are read
once. The highest-scoring label is the choice; arguments are grounded deterministically,
never generated. Training run `{run}`. **This is a derivative work based on the base
model**: the weights were changed by LoRA fine-tuning and merged (see `NOTICE`).
{build_note}
The bundle was trained on candidate surface `{surface_sha256}` (see `bundle.json`).

## Licence

{licence}, the licence of the base model, included as `LICENSE`.

## Use

{use}
## Results

{quoted}

## Training data

{data_summary}
{teacher_block}{calibration_section()}"""


def _copy_payload(
    kind: str, merged: Path, out: Path, gguf: Path | None, awq_dir: Path | None
) -> tuple[str | None, str | None]:
    if kind == "bf16":
        shutil.copytree(merged, out, symlinks=True)  # links stay links: check_bundle refuses
        return None, None
    if kind == "awq":
        shutil.copytree(awq_dir, out, symlinks=True)
        return None, None
    sources = [merged / n for n in TOKENIZER_FILES if (merged / n).is_file()] + [gguf]
    links = [p.name for p in sources if p.is_symlink()]
    if links:
        raise BundleError(f"refusing symlink {', '.join(links)}: a bundle never holds a link")
    out.mkdir(parents=True)
    for source in sources:
        shutil.copyfile(source, out / source.name)
    return gguf.name, _sha256(out / gguf.name)


def _require_sources(**sources: Path | None) -> None:
    """Refuse before copying anything when a required input file is missing."""
    for name, path in sources.items():
        if path is None or not Path(path).is_file():
            raise BundleError(f"a bundle must carry {name}.json: pass an existing {name} file")


def _check_kind_inputs(kind: str, gguf: Path | None, awq_dir: Path | None) -> None:
    """Refuse an unknown kind, or a kind whose payload (GGUF file, AWQ folder) is missing."""
    if kind not in KINDS:
        raise BundleError(f"unknown bundle kind {kind!r} (one of {', '.join(KINDS)})")
    if kind == "gguf" and (gguf is None or not gguf.is_file()):
        raise BundleError("kind gguf needs gguf, an existing .gguf file")
    if kind == "awq" and (awq_dir is None or not awq_dir.is_dir()):
        raise BundleError("kind awq needs awq_dir, an existing export folder")


def _report_list(results: Path | Sequence[Path], data_summary: str) -> list[Path]:
    """The results reports as a list; refuse none, or an empty *data_summary*."""
    reports = [results] if isinstance(results, Path) else list(results)
    if not reports:
        raise BundleError("pass at least one results report")
    if not data_summary.strip():
        raise BundleError("data_summary must describe the training data")
    return reports


def _check_teacher_licences(teachers: _dataset.TeacherSummary | None, licence: str) -> None:
    """An Apache bundle names only Apache-2.0 teachers."""
    if teachers is None or licence != APACHE_LICENCE:
        return
    for name, lic, _ in _dataset.teacher_rows(teachers):
        if lic != APACHE_LICENCE:
            raise BundleError(
                f"teacher {name!r} ({lic}) is not {APACHE_LICENCE}; an Apache bundle"
                " names only Apache-2.0 teachers"
            )


def _check_chat_templates(
    base_snapshot: Path, merged: Path, kind: str, awq_dir: Path | None
) -> None:
    base_template = chat_template(base_snapshot)
    if chat_template(merged) != base_template:
        raise BundleError("the checkpoint's chat template differs from the base model's")
    if kind == "awq" and chat_template(awq_dir) != base_template:
        raise BundleError("the AWQ export's chat template differs from the base model's")


def _base_licence_file(base_snapshot: Path, licence: str) -> Path:
    licence_file = base_snapshot / "LICENSE"
    if not licence_file.is_file():
        raise BundleError(f"{base_snapshot} ships no LICENSE file")
    _check_licence_file(licence_file, licence)
    return licence_file


def _quoted_results(reports: list[Path], heading: str | None) -> list[tuple[str, str, str]]:
    quoted = []
    for path in reports:
        caption, table = results_section(path, heading)
        quoted.append((caption, path.name, table))
    return quoted


def _awq_serve_args(kind: str, awq_dir: Path | None) -> list[str]:
    """The vLLM serve arguments the quantize run recorded next to an AWQ export."""
    if kind != "awq":
        return []
    record = awq_dir.parent / "quantize-run.json"
    if not record.is_file():
        raise BundleError(f"no {record} next to the AWQ export; run quantize first")
    return list(json.loads(record.read_text(encoding="utf-8"))["awq_serve_args"])


def _clear_out(out: Path) -> None:
    """Remove an empty *out* folder; refuse one that holds anything."""
    if not out.exists():
        return
    if any(out.iterdir()):
        raise BundleError(f"{out} is not empty")
    out.rmdir()


def _check_card(card: str, licence: str, kind: str) -> None:
    required = (f"license: {licence.lower()}", "base_model:", "derivative work")
    missing = [p for p in required if p not in card]
    if kind == "gguf":
        missing += [p for p in ("llama-server", "--jinja") if p not in card]
    if missing:
        raise BundleError(f"model card is missing {missing}")


def build_model_bundle(
    *,
    domain: Domain,
    config: RunConfig,
    merged: Path,
    base_snapshot: Path,
    repo: str,
    run: str,
    results: Path | Sequence[Path],
    data_summary: str,
    out: Path,
    calibration: Path | None,
    gate: Path | None,
    scorer_train: Path | None,
    kind: str = "bf16",
    gguf: Path | None = None,
    awq_dir: Path | None = None,
    teachers: _dataset.TeacherSummary | None = None,
    quantized_from: str | None = None,
    results_heading: str | None = None,
) -> str:
    """Write a model bundle to *out*; return the merged checkpoint's revision.

    *calibration*, *gate* and *scorer_train* are required and ship byte for byte as
    ``calibration.json``, ``gate.json`` and ``scorer-train.json``. The hub prefix, licence
    and card prose come from *config* and *domain*. The finished folder is checked with
    :func:`check_bundle` (no symlink, no absolute GGUF path) and a failure removes it."""
    _check_kind_inputs(kind, gguf, awq_dir)
    _require_sources(calibration=calibration, gate=gate, **{"scorer-train": scorer_train})
    reports = _report_list(results, data_summary)
    licence = _licence(domain, config)
    _check_teacher_licences(teachers, licence)
    _check_chat_templates(base_snapshot, merged, kind, awq_dir)
    licence_file = _base_licence_file(base_snapshot, licence)
    quoted = _quoted_results(reports, results_heading)
    awq_serve_args = _awq_serve_args(kind, awq_dir)
    base_repo, base_revision = _base_identity(base_snapshot)
    _clear_out(out)
    try:
        gguf_name, gguf_sha256 = _copy_payload(kind, merged, out, gguf, awq_dir)
        mtp_from = drop_undeclared_mtp(out) if kind != "gguf" else None
        shutil.copyfile(licence_file, out / "LICENSE")
        (out / "NOTICE").write_text(
            notice(
                base_repo,
                base_revision,
                repo,
                licence=licence,
                domain=domain,
                kind=kind,
                mtp_from=mtp_from,
            ),
            encoding="utf-8",
        )
        for source, name in (
            (calibration, CALIBRATION_FILE),
            (gate, GATE_FILE),
            (scorer_train, "scorer-train.json"),
        ):
            shutil.copyfile(source, out / name)
        info = _write_info(out, kind="model", domain=domain, config=config, repo=repo, run=run)
        card = model_card(
            domain=domain,
            licence=licence,
            repo=repo,
            base_repo=base_repo,
            base_revision=base_revision,
            run=run,
            results=quoted,
            data_summary=data_summary.strip(),
            surface_sha256=info["surface_sha256"],
            teachers=teachers,
            kind=kind,
            mtp_from=mtp_from,
            gguf_name=gguf_name,
            gguf_sha256=gguf_sha256,
            awq_serve_args=awq_serve_args,
            quantized_from=quantized_from,
        )
        _check_card(card, licence, kind)
        (out / "README.md").write_text(card, encoding="utf-8")
        check_bundle(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return revision_of(merged)


def build_dataset_bundle(
    *,
    domain: Domain,
    config: RunConfig,
    splits: Path,
    train_augmented: Path,
    accepted: Path,
    rejected: Path | list[Path],
    licence_file: Path,
    role_models: Mapping[str, tuple[str, str]],
    out: Path,
    calibration: Path | None,
    gate: Path | None,
    scorer_train: Path | None,
    repo: str = "",
    run: str = "",
    model_repos: list[str] | None = None,
    default_source: str | None = None,
    supplement_teachers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a dataset bundle to *out*; return the counts shown in the card.

    Like a model bundle it must carry ``calibration.json``, ``gate.json`` and
    ``scorer-train.json`` plus ``bundle.json`` with the domain's surface hash, and is
    checked with :func:`check_bundle`; a failure removes the folder."""
    _require_sources(calibration=calibration, gate=gate, **{"scorer-train": scorer_train})
    licence = _licence(domain, config)
    _check_licence_file(licence_file, licence)
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"{out} is not empty")
    try:
        counts = _dataset.build(
            domain=domain,
            splits=splits,
            train_augmented=train_augmented,
            accepted=accepted,
            rejected=rejected,
            licence=licence_file,
            role_models=dict(role_models),
            scorer_train=scorer_train,
            out=out,
            apache_only=licence == APACHE_LICENCE,
            issue_refs=str(config.get("issue_refs") or ""),
            model_repos=model_repos,
            default_source=default_source,
            supplement_teachers=supplement_teachers,
        )
        for source, name in ((calibration, CALIBRATION_FILE), (gate, GATE_FILE)):
            shutil.copyfile(source, out / name)
        _write_info(out, kind="dataset", domain=domain, config=config, repo=repo, run=run)
        check_bundle(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return counts
