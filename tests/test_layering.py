"""Layering greps: the factory core knows nothing of nvsh's domain or of the jev CLI.

The Domain seam (``jev_factory.domain``) is the only place a domain's operation
table, names and prompts enter the factory. So, in code outside the domain
modules, none of these may appear (o28, o29). "Domain modules" means everything
under ``jev_factory/domains/`` (the ``jev_cli`` domain today, the nvsh Tool-Jev
one as it lands); the tests scan every other module, so they stay true as more
stages and domains arrive. ``jev_factory/cli`` and ``jev_factory/explain`` are the
CLI itself, so they (and only they) may hold the jev verb names.

What is scanned is *code*: imports, identifiers and string literals. Docstrings
and the ``NVSH_PROVENANCE`` headers (which name the nvsh files a module was
cited from) are documentation and are skipped, as are comments.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "jev_factory"
PROVENANCE = "NVSH_PROVENANCE"

#: nvsh's own operation table (nvsh/ops/table.py): names that belong to its domain module.
NVSH_OPERATION_NAMES = (
    "machine_status",
    "memory_stats",
    "gpu_stats",
    "disk_stats",
    "thermal_stats",
    "container_list",
    "network_info",
    "process_list",
    "power_get",
    "swap_status",
    "nvsh_doctor",
    "service_status",
    "service_logs",
    "power_set",
    "service_restart",
    "container_restart",
)
#: Words that mark nvsh / jetson-ai-lab in code (the hub prefix, the package, its sibling).
NVSH_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9])nvsh(?![A-Za-z0-9])|jetson[-_ ]ai[-_ ]lab|jetson_skills", re.IGNORECASE
)
#: A citation of an nvsh decision or issue ("nvsh D47", "nvsh#58") names no domain content.
NVSH_CITATION_RE = re.compile(r"\bnvsh(?: D\d+|#\d+)")
NVSH_OP_RE = re.compile(r"\b(" + "|".join(NVSH_OPERATION_NAMES) + r")\b")
#: Code where an nvsh name is the point, each with why: (relative path, the literal).
ALLOWED_NVSH_LITERALS = {
    # ISSUE46_MAPPING documents how nvsh's own outcome names map onto issue 46's action
    # vocabulary, so its first column is literally called "nvsh".
    ("core/metrics.py", "nvsh"),
    ("core/metrics.py", "| nvsh | issue 46 |"),
}
#: The jev command line's own verbs (cli/_commands/*): a list of these is the CLI surface.
JEV_VERBS = ("whoami", "learn", "explain", "overview", "doctor", "version", "status", "decide")
#: The CLI itself may hold its verb names.
CLI_PACKAGES = ("cli", "explain")


def _modules(exclude: tuple[str, ...] = ()) -> list[Path]:
    """Every module outside ``domains/`` (and outside *exclude* packages)."""
    skipped = ("domains",) + exclude
    return sorted(
        p
        for p in PKG.rglob("*.py")
        if "__pycache__" not in p.parts and p.relative_to(PKG).parts[0] not in skipped
    )


def _skipped_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line ranges of docstrings and NVSH_PROVENANCE headers: documentation, not code."""
    ranges = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # A bare string statement: a docstring or an attribute docstring.
            ranges.append((node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == PROVENANCE for t in targets):
                ranges.append((node.lineno, node.end_lineno or node.lineno))
    return ranges


def _strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string literal in code, with its line, docstrings and headers excluded."""
    skip = _skipped_ranges(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and not any(lo <= node.lineno <= hi for lo, hi in skip)
    ]


def _code_words(tree: ast.AST) -> list[tuple[int, str]]:
    """Identifiers and imported module names in code, docstrings and headers excluded."""
    skip = _skipped_ranges(tree)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if any(lo <= line <= hi for lo, hi in skip):
            continue
        if isinstance(node, ast.Name):
            out.append((line, node.id))
        elif isinstance(node, ast.Attribute):
            out.append((line, node.attr))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((line, node.module))
        elif isinstance(node, ast.Import):
            out.extend((line, alias.name) for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((line, node.name))
    return out


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _qwen_only_checks(tree: ast.AST) -> list[tuple[int, str]]:
    """Code that *tests* a model name against "qwen": a comparison or a startswith-style call.

    A default model id or a doc string that mentions Qwen is not a check; a branch
    that only works for Qwen is, and belongs to the causal-LM adapter's own seam.
    """
    found = []
    skip = _skipped_ranges(tree)
    for node in ast.walk(tree):
        if any(lo <= getattr(node, "lineno", 0) <= hi for lo, hi in skip):
            continue
        consts: list[ast.expr] = []
        if isinstance(node, ast.Compare):
            consts = [node.left, *node.comparators]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"startswith", "endswith", "match", "search", "fnmatch"}:
                consts = list(node.args)
        for c in consts:
            values = [c] + (list(c.elts) if isinstance(c, (ast.Tuple, ast.List, ast.Set)) else [])
            for v in values:
                if (
                    isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                    and "qwen" in v.value.lower()
                ):
                    found.append((node.lineno, v.value))
    return found


@pytest.mark.behavioral("o28")
def test_no_nvsh_name_or_jetson_ai_lab_prefix_outside_the_domain_modules():
    hits = []
    for path in _modules():
        rel = path.relative_to(PKG).as_posix()
        tree = _tree(path)
        for line, text in _strings(tree) + _code_words(tree):
            if (rel, text) in ALLOWED_NVSH_LITERALS:
                continue
            text = NVSH_CITATION_RE.sub("", text)
            if NVSH_NAME_RE.search(text) or NVSH_OP_RE.search(text):
                hits.append(f"{rel}:{line}: {text!r}")
    assert hits == [], "nvsh-specific names outside domains/:\n" + "\n".join(hits)


@pytest.mark.behavioral("o28")
def test_no_qwen_only_check_outside_the_domain_modules():
    hits = []
    for path in _modules():
        for line, text in _qwen_only_checks(_tree(path)):
            hits.append(f"{path.relative_to(PKG).as_posix()}:{line}: {text!r}")
    assert hits == [], "a check that only passes Qwen models:\n" + "\n".join(hits)


@pytest.mark.behavioral("o28")
def test_training_data_is_chosen_by_frozen_sha256_never_mtime():
    offenders = []
    for path in _modules():
        tree = _tree(path)
        offenders += [
            f"{path.relative_to(PKG).as_posix()}:{line}: {word}"
            for line, word in _code_words(tree)
            if "mtime" in word.lower()
        ]
    assert offenders == []
    source = (PKG / "backbones" / "causal_lm" / "train_scorer.py").read_text(encoding="utf-8")
    assert "expect_sha256" in source  # the frozen hash is what train-scorer checks


def test_the_scan_catches_what_it_is_meant_to_catch():
    tree = ast.parse(
        '"""nvsh docstring."""\n'
        'NVSH_PROVENANCE = {"upstream": "nvsh/ops/table.py"}\n'
        "import nvsh.ops\n"
        'HUB = "jetson-ai-lab/x"\n'
        'OP = "gpu_stats"\n'
        'if "qwen" in model.lower():\n    pass\n'
        'if model.startswith(("Qwen", "x")):\n    pass\n'
    )
    text = [t for _, t in _strings(tree) + _code_words(tree)]
    flagged = [t for t in text if NVSH_NAME_RE.search(t) or NVSH_OP_RE.search(t)]
    assert sorted(flagged) == ["gpu_stats", "jetson-ai-lab/x", "nvsh.ops"]
    assert NVSH_CITATION_RE.sub("", "nvsh D47 and nvsh#58") == " and "
    assert [v for _, v in _qwen_only_checks(tree)] == ["qwen", "Qwen"]


def _jev_cli_texts() -> list[str]:
    """The jev-CLI domain's own prompt text and candidate names, read from the domain."""
    from jev_factory.domains.jev_cli.generate import generate_domain

    domain = generate_domain()
    texts = [
        domain.description,
        domain.answer_policy,
        domain.persona,
        domain.card_text,
        *domain.explain_topics,
        *domain.phrasing_styles,
        *(r.description for r in domain.reasons),
        *(r.definition for r in domain.reasons),
    ]
    return [t for t in texts if t and len(t) >= 20]


@pytest.mark.behavioral("o29")
def test_no_jev_cli_prompt_outside_the_jev_tool_domain_module():
    prompts = _jev_cli_texts()
    assert prompts, "the jev_cli domain should define prompt text"
    hits = []
    for path in _modules():
        tree = _tree(path)
        for line, text in _strings(tree):
            for prompt in prompts:
                if prompt in text:
                    hits.append(f"{path.relative_to(PKG).as_posix()}:{line}: {prompt!r}")
    assert hits == [], "jev-CLI prompt text outside domains/:\n" + "\n".join(hits)


@pytest.mark.behavioral("o29")
def test_no_jev_cli_verb_list_outside_the_jev_tool_domain_module():
    from jev_factory.domains.jev_cli.generate import generate_domain

    derived = {n.removeprefix("jev.").split(".")[0] for n in generate_domain().names()}
    verbs = set(JEV_VERBS) | (derived - {"cli"})
    hits = []
    for path in _modules(exclude=CLI_PACKAGES):
        tree = _tree(path)
        skip = _skipped_ranges(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                continue
            if any(lo <= node.lineno <= hi for lo, hi in skip):
                continue
            named = {
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value in verbs
            }
            if len(named) >= 3:
                hits.append(f"{path.relative_to(PKG).as_posix()}:{node.lineno}: {sorted(named)}")
    assert hits == [], "a jev-CLI verb list outside the CLI and domains/:\n" + "\n".join(hits)


def test_the_verb_list_and_prompt_scans_see_the_cli_domain():
    """The domain the scans compare against is real: its names include the known verbs."""
    from jev_factory.domains.jev_cli.generate import generate_domain

    names = set(generate_domain().names())
    assert {"jev.whoami", "jev.learn", "jev.overview"} <= names
