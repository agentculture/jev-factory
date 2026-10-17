"""The causal-LM backbone as a :class:`~jev_factory.backbones.base.Backbone` adapter.

A thin facade over the existing causal-LM modules; it adds no behaviour. Members with no
existing equivalent raise ``NotImplementedError`` rather than guessing. Each delegate is
imported lazily, so importing this module pulls in no heavy dependency.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from jev_factory.backbones.base import Backbone


class CausalLMBackbone(Backbone):
    """Letter-token readout over a causal decoder (Qwen3.5-0.8B, LFM2.5-350M)."""

    name = "causal_lm"

    def render(self, domain: Any, request: str, offered: Sequence[str]) -> Any:
        """The lettered chat messages (scorer.prompt_messages) for one request.

        A tokenizer's chat template (scorer.render_prompt) turns these into the prompt string
        that predict_distributions' rows carry.
        """
        from jev_factory.backbones.causal_lm import scorer

        return scorer.prompt_messages(domain, request, list(offered))

    def predict_distributions(
        self, model: Any, rendered: Sequence[Any], offered: Sequence[Sequence[str]]
    ) -> list[Mapping[str, float]]:
        """Probabilities over each row's candidates via scorer.score.

        *model* is a top-k callable (see scorer.served_top_k / InProcessTopK). Each *rendered*
        row is a mapping with ``domain``, ``prompt`` (the templated prompt string, see
        scorer.render_prompt) and ``request_text``.
        """
        from jev_factory.backbones.causal_lm import scorer

        out = []
        for row, names in zip(rendered, offered):
            pred = scorer.score(
                row["domain"], model, row["prompt"], row["request_text"], offered=list(names)
            )
            out.append(dict(pred.probabilities))
        return out

    def train_plan(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """The dry-run training plan (train.plan_train); *config* is its keyword arguments."""
        from jev_factory.backbones.causal_lm import train

        plan = train.plan_train(**config)
        if dataclasses.is_dataclass(plan) and not isinstance(plan, type):
            return dataclasses.asdict(plan)
        return plan

    def merge_verify(self, adapter_dir: str, base: str) -> Mapping[str, Any]:
        raise NotImplementedError(
            "causal_lm.merge_verify: train.merge_adapter/verify_merge need a loaded model and "
            "tokenizer, not a path pair; call them directly until this adapter is wired"
        )

    def export(self, model_dir: str, out_dir: str, *, apply: bool = False) -> Mapping[str, Any]:
        raise NotImplementedError(
            "causal_lm.export: no single existing export entry point; use "
            "jev_factory.release.bundle until this adapter is wired"
        )

    def quantize_candidates(self, config: Mapping[str, Any]) -> Sequence[str]:
        """The deployed format: GGUF Q4_K_M."""
        return ("gguf-q4_k_m",)

    # ---- the distillery's hooks: no distill path exists for this family ----------------

    @staticmethod
    def _no_distill(member: str) -> NotImplementedError:
        return NotImplementedError(
            f"causal_lm.{member}: the distillery has no causal-LM student; the causal-LM "
            "factory path is jev run"
        )

    def layout(self) -> Any:
        raise self._no_distill("layout")

    def config_files(self) -> Mapping[str, str]:
        raise self._no_distill("config_files")

    def serve_argv(
        self, python: str, bundle: Any, *, port: int, host: str = "127.0.0.1"
    ) -> list[str]:
        raise self._no_distill("serve_argv")

    def server(self) -> Any:
        raise self._no_distill("server")

    def plan_distill_train(self, **kwargs: Any) -> Mapping[str, Any]:
        raise self._no_distill("plan_distill_train")

    def run_distill_train(self, plan: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        raise self._no_distill("run_distill_train")

    def trainer(self) -> Any:
        raise self._no_distill("trainer")

    def quantizer(self) -> Any:
        raise self._no_distill("quantizer")


BACKBONE = CausalLMBackbone()
