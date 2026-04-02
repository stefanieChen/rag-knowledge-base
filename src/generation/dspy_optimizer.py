"""DSPy-based prompt optimization utilities.

This module provides an optional lightweight wrapper that can refine the
baseline LLM answer using DSPy's programmatic prompting when enabled in the
configuration.  The implementation keeps everything optional so that the
system operates smoothly even when DSPy (and its dependencies) are not
installed.
"""

from __future__ import annotations

from typing import Optional

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("generation.dspy")


class DSpyPromptOptimizer:
    """Optionally refines answers using DSPy when enabled."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        gen_cfg = config.get("generation", {})
        dspy_cfg = gen_cfg.get("dspy", {})

        self.enabled = bool(dspy_cfg.get("enabled", False))
        self._trainset_file = dspy_cfg.get("trainset_file")
        self._max_train_examples = int(dspy_cfg.get("max_train_examples", 10))
        self._metric = dspy_cfg.get("metric", "substring")

        if not self.enabled:
            self._optimizer = None
            logger.info("DSPy optimizer disabled")
            return

        try:
            import dspy  # type: ignore
            from dspy.teleprompt import BootstrapFewShot  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("DSPy not available: %s", exc)
            self.enabled = False
            self._optimizer = None
            return

        try:
            self._optimizer = _build_optimizer(
                dspy,
                BootstrapFewShot,
                trainset_file=self._trainset_file,
                max_examples=self._max_train_examples,
                metric=self._metric,
            )
            logger.info("DSPy optimizer initialized with metric=%s", self._metric)
        except Exception as exc:  # pragma: no cover - initialization failure
            logger.warning("Failed to initialize DSPy optimizer: %s", exc)
            self.enabled = False
            self._optimizer = None

    def refine_answer(self, *, question: str, context: str, baseline_answer: str) -> Optional[str]:
        if not self.enabled or not self._optimizer:
            return None
        try:
            optimized = self._optimizer(question=question, context=context, answer=baseline_answer)
        except Exception as exc:  # pragma: no cover - depends on external library
            logger.warning("DSPy refinement failed: %s", exc)
            return None
        if not optimized:
            return None
        refined = optimized.get("answer") if isinstance(optimized, dict) else str(optimized)
        if refined and refined.strip() and refined.strip() != baseline_answer.strip():
            return refined.strip()
        return None


def _build_optimizer(dspy, bootstrap_cls, *, trainset_file: Optional[str], max_examples: int, metric: str):  # type: ignore
    """Construct a BootstrapFewShot optimizer if training data is available."""

    if not trainset_file:
        logger.warning("DSPy optimizer enabled but no trainset_file provided")
        return None

    try:
        import json

        with open(trainset_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:  # pragma: no cover - IO failure
        logger.warning("Failed to load DSPy trainset: %s", exc)
        return None

    examples = []
    for item in raw_data[:max_examples]:
        question = item.get("question")
        contexts = item.get("contexts") or []
        answer = item.get("ground_truth") or item.get("answer")
        if not question or not answer:
            continue
        examples.append({
            "question": question,
            "context": "\n\n".join(contexts),
            "answer": answer,
        })

    if not examples:
        logger.warning("No usable examples found for DSPy optimizer")
        return None

    class RagModule(dspy.Module):  # type: ignore
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question: str, context: str, answer: str = ""):
            generated = self.generate(question=question, context=context)
            return {"answer": generated}

    optimizer = bootstrap_cls(metric=metric)
    optimizer.compile(RagModule(), trainset=examples)
    return optimizer
