"""The distillation loss: alpha * KL(teacher || student at T_kind) + (1 - alpha) * CE(gold).

Two implementations of one formula. :func:`distill_loss` is a pure-Python reference on lists,
tested in CI without torch; :func:`distill_loss_torch` is the one the trainer optimises, and
the test suite checks it against the reference wherever torch is present. torch is imported
only inside the torch functions (``importlib``), never at module import.

Per row, over the row's own options only (a pointer head emits one logit per option):

* ``KL_i = T^2 * sum_k t_k * (log t_k - log q_k)`` with ``t = soften(teacher, T)`` (the
  teacher's probabilities at temperature T, ``p^(1/T)`` renormalised, which is the teacher's
  logits divided by T) and ``q = softmax(logits / T)``. T is the row's per-question-kind
  distillation temperature. Both sides are softened and the term is scaled by T^2 (Hinton,
  Vinyals and Dean 2015), so a student equal to the teacher has KL 0 at every T and the
  gradient scale does not shrink as T grows. Teacher zeros carry no weight, exactly as
  upstream's forward ``_teacher_kl`` skips them. A row with no teacher has KL 0.
* ``CE_i = -sum_k g_k * log softmax(logits)_k`` against the gold target ``g`` at temperature
  1 (the serving temperature). For a score question with a hard gold level, ``g`` is
  upstream's ordinal smoothing (:func:`gold_target`); a noul gold is ``[1 - p_true, p_true]``.
  A row with no gold has no CE.

Batch: ``loss = alpha * mean_{rows with teacher} KL_i + (1 - alpha) * wmean_{rows with gold}
CE_i``, the CE mean weighted by row weight. These are upstream's two normalisations: the
teacher KL is a plain mean over the rows that have a teacher (``train.py``:
``_teacher_kl(...).mean()``) and the label loss is the weight-normalised mean of
``modeling.StrandsDeciderModel.forward``. With ``alpha = 0`` the loss is exactly the gold CE.

Provenance: the KL masking and the ordinal smoothing are re-expressed from
strands-labs/strands-decider @ 8b6faa553038a59e73fa7f16c4081f8ceb21d27e
(``training/train.py`` ``_kl``/``_teacher_kl``; ``training/data/collate.py``
``SystemOneCollator._target_distribution`` and the ``order[k] = canonical index at slot k``
permutation convention), Copyright the strands-decider authors, Apache License 2.0. Changes:
the per-kind temperature and T^2 scaling, the alpha mixing in place of upstream's additive
``teacher_weight``, soft (noul / probability) gold, and a list-based reference.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

#: Teacher entries at or below this are treated as zero mass (upstream's ``tea > 0`` mask).
TEACHER_ZERO = 0.0


@dataclass(frozen=True)
class LossRow:
    """One training row in slot order: logits, gold target, teacher target, temperature."""

    logits: Sequence[float]
    target: Sequence[float] | None
    teacher: Sequence[float] | None
    temperature: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class LossResult:
    loss: float
    kl: float
    ce: float
    per_row_kl: list[float]
    per_row_ce: list[float | None]
    n_teacher: int
    n_gold: int


# --------------------------------------------------------------------- targets


def gold_target(kind: str, n: int, gold: int, *, ordinal_smoothing: float) -> list[float]:
    """Canonical-order target for a hard gold index (upstream ``_target_distribution``).

    Score: ``1 - eps`` on the gold level and ``eps`` split evenly over its existing neighbours
    (all mass on gold when it has none, or when ``eps <= 0``). Choice and noul: one-hot.
    """
    if not 0 <= gold < n:
        raise ValueError(f"gold index {gold} outside {n} options")
    out = [0.0] * n
    if kind != "score" or ordinal_smoothing <= 0:
        out[gold] = 1.0
        return out
    neighbours = [lv for lv in (gold - 1, gold + 1) if 0 <= lv < n]
    if not neighbours:
        out[gold] = 1.0
        return out
    out[gold] = 1.0 - ordinal_smoothing
    for lv in neighbours:
        out[lv] += ordinal_smoothing / len(neighbours)
    return out


def noul_target(p_true: float) -> list[float]:
    """A noul gold in canonical (false, true) order."""
    if not 0.0 <= p_true <= 1.0:
        raise ValueError(f"p_true {p_true} outside [0, 1]")
    return [1.0 - p_true, p_true]


def permute(canonical: Sequence[float], order: Sequence[int] | None) -> list[float]:
    """Canonical-order values in slot order, ``order[k]`` being the canonical index at slot k."""
    if order is None:
        return list(canonical)
    return [canonical[i] for i in order]


# --------------------------------------------------------------------- pure reference


def log_softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    z = math.log(sum(math.exp(v - m) for v in logits)) + m
    return [v - z for v in logits]


def soften(probs: Sequence[float], temperature: float) -> list[float]:
    """``p^(1/T)`` renormalised; zeros stay zero."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    powered = [p ** (1.0 / temperature) if p > TEACHER_ZERO else 0.0 for p in probs]
    total = sum(powered)
    if total <= 0:
        raise ValueError("teacher distribution has no mass")
    return [p / total for p in powered]


def _check_row(row: LossRow) -> int:
    n = len(row.logits)
    if n < 2:
        raise ValueError("a row needs at least 2 options")
    for name in ("target", "teacher"):
        values = getattr(row, name)
        if values is not None and len(values) != n:
            raise ValueError(f"{name} has {len(values)} entries for {n} options")
    if row.target is None and row.teacher is None:
        raise ValueError("a row has neither a gold target nor a teacher target")
    return n


def row_kl(row: LossRow) -> float:
    """``T^2 * KL(soften(teacher, T) || softmax(logits / T))``; 0 without a teacher."""
    if row.teacher is None:
        return 0.0
    t = row.temperature
    target = soften(row.teacher, t)
    q = log_softmax([v / t for v in row.logits])
    kl = sum(p * (math.log(p) - lq) for p, lq in zip(target, q) if p > TEACHER_ZERO)
    return t * t * kl


def row_ce(row: LossRow) -> float | None:
    """``-sum g * log softmax(logits)`` at temperature 1; None without gold."""
    if row.target is None:
        return None
    q = log_softmax(list(row.logits))
    return -sum(g * lq for g, lq in zip(row.target, q) if g > 0)


def _check_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def distill_loss(rows: Sequence[LossRow], *, alpha: float) -> LossResult:
    """The batch loss on lists; the reference the torch loss is checked against."""
    _check_alpha(alpha)
    for row in rows:
        _check_row(row)
    kls = [row_kl(r) for r in rows]
    ces = [row_ce(r) for r in rows]
    teacher_rows = [k for k, r in zip(kls, rows) if r.teacher is not None]
    gold = [(c, r.weight) for c, r in zip(ces, rows) if c is not None]
    kl = sum(teacher_rows) / len(teacher_rows) if teacher_rows else 0.0
    wsum = sum(w for _, w in gold)
    ce = sum(c * w for c, w in gold) / max(wsum, 1e-8) if gold else 0.0
    return LossResult(
        loss=alpha * kl + (1.0 - alpha) * ce,
        kl=kl,
        ce=ce,
        per_row_kl=kls,
        per_row_ce=ces,
        n_teacher=len(teacher_rows),
        n_gold=len(gold),
    )


# --------------------------------------------------------------------- torch


def _torch() -> Any:
    return importlib.import_module("torch")


def to_tensors(rows: Sequence[LossRow]) -> dict[str, Any]:
    """Pad rows to the widest option count: the tensors :func:`distill_loss_torch` reads."""
    torch = _torch()
    for row in rows:
        _check_row(row)
    width = max(len(r.logits) for r in rows)
    b = len(rows)
    logits = torch.zeros((b, width), dtype=torch.float32)
    target = torch.zeros((b, width), dtype=torch.float32)
    teacher = torch.zeros((b, width), dtype=torch.float32)
    for i, r in enumerate(rows):
        n = len(r.logits)
        logits[i, :n] = torch.tensor(list(r.logits), dtype=torch.float32)
        if r.target is not None:
            target[i, :n] = torch.tensor(list(r.target), dtype=torch.float32)
        if r.teacher is not None:
            teacher[i, :n] = torch.tensor(list(r.teacher), dtype=torch.float32)
    return {
        "logits": logits,
        "n_options": torch.tensor([len(r.logits) for r in rows], dtype=torch.long),
        "target": target,
        "has_gold": torch.tensor([r.target is not None for r in rows]),
        "teacher": teacher,
        "has_teacher": torch.tensor([r.teacher is not None for r in rows]),
        "temperature": torch.tensor([float(r.temperature) for r in rows], dtype=torch.float32),
        "weights": torch.tensor([float(r.weight) for r in rows], dtype=torch.float32),
    }


def _masked_log_softmax(torch: Any, logits: Any, valid: Any) -> Any:
    masked = logits.float().masked_fill(~valid, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def distill_loss_torch(logits: Any, batch: dict[str, Any], *, alpha: float) -> dict[str, Any]:
    """The same loss on tensors. *logits* is ``[B, K]`` in slot order (raw, temperature 1).

    *batch* carries ``n_options [B]``, ``target [B, K]`` and ``has_gold [B]``, ``teacher [B, K]``
    and ``has_teacher [B]``, ``temperature [B]`` and ``weights [B]``, all in slot order.
    """
    _check_alpha(alpha)
    torch = _torch()
    device = logits.device
    b, k = logits.shape
    n_opt = batch["n_options"].to(device)
    valid = torch.arange(k, device=device).unsqueeze(0) < n_opt.unsqueeze(1)

    # CE to gold at temperature 1, weighted mean over rows with gold.
    lq = _masked_log_softmax(torch, logits, valid)
    safe = torch.where(valid, lq, torch.zeros_like(lq))
    target = batch["target"].to(device)[:, :k]
    per_ce = -(target * safe).sum(dim=-1)
    has_gold = batch["has_gold"].to(device).bool()
    w = batch["weights"].to(device).float() * has_gold.float()
    ce = (per_ce * w).sum() / w.sum().clamp_min(1e-8)

    # KL(soften(teacher, T) || softmax(logits / T)) * T^2, mean over rows with a teacher.
    t = batch["temperature"].to(device).float().clamp_min(1e-6).view(-1, 1)
    tea = batch["teacher"].to(device).float()[:, :k]
    has_teacher = batch["has_teacher"].to(device).bool()
    live = (tea > TEACHER_ZERO) & valid & has_teacher.view(-1, 1)
    log_tea = torch.where(live, tea.clamp_min(1e-30).log() / t, torch.full_like(tea, -1e30))
    log_soft = torch.log_softmax(log_tea, dim=-1)
    soft = torch.where(live, log_soft.exp(), torch.zeros_like(tea))
    lqt = _masked_log_softmax(torch, logits / t, valid)
    diff = torch.where(live, log_soft - torch.where(valid, lqt, 0.0), 0.0)
    per_kl = (soft * diff).sum(dim=-1) * (t.view(-1) ** 2)
    per_kl = torch.where(has_teacher, per_kl, torch.zeros_like(per_kl))
    n_t = has_teacher.float().sum()
    kl = per_kl.sum() / n_t.clamp_min(1.0)

    loss = alpha * kl + (1.0 - alpha) * ce
    return {"loss": loss, "kl": kl, "ce": ce, "per_row_kl": per_kl, "per_row_ce": per_ce}
