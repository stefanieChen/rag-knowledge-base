"""Few-shot example selection utilities for generator prompts."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import load_config
from src.embedding.embedder import Embedder
from src.logging.logger import get_logger

logger = get_logger("generation.few_shot")


class FewShotSelector:
    """Selects few-shot examples using embedding similarity."""

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        embedder: Optional[Embedder] = None,
        examples: Optional[List[Dict]] = None,
    ) -> None:
        if config is None:
            config = load_config()

        gen_cfg = config.get("generation", {})
        fs_cfg = gen_cfg.get("few_shot", {})

        self.enabled: bool = bool(fs_cfg.get("enabled", False))
        self._max_examples = int(fs_cfg.get("max_examples", 2))

        if not self.enabled:
            self._examples = []
            self._embedder = None
            self._example_embeddings = []
            logger.info("Few-shot prompting disabled")
            return

        if examples is None:
            examples_file = fs_cfg.get("examples_file")
            if not examples_file:
                raise ValueError("Few-shot prompting enabled but no examples_file configured")
            examples = _load_examples(examples_file)

        if not examples:
            logger.warning("Few-shot examples unavailable; disabling few-shot prompting")
            self.enabled = False
            self._examples = []
            self._embedder = None
            self._example_embeddings = []
            return

        self._examples = examples

        try:
            self._embedder = embedder or Embedder(config=config)
            self._example_embeddings = self._embedder.embed_documents(
                [ex["question"] for ex in self._examples]
            )
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.warning("Failed to initialize few-shot embeddings: %s", exc)
            self.enabled = False
            self._examples = []
            self._embedder = None
            self._example_embeddings = []
            return

        logger.info(
            "Few-shot selector initialized: %s examples, max_examples=%s",
            len(self._examples),
            self._max_examples,
        )

    def select(self, question: str) -> List[BaseMessage]:
        if not self.enabled or not question or not self._examples or not self._embedder:
            return []

        query_embedding = self._embedder.embed_query(question)
        scores = [
            _cosine_similarity(query_embedding, emb)
            for emb in self._example_embeddings
        ]
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        selected_messages: List[BaseMessage] = []

        for idx in ranked_indices[: self._max_examples]:
            example = self._examples[idx]
            human = HumanMessage(content=_build_human_message(example))
            ai = AIMessage(content=example.get("answer", ""))
            selected_messages.extend([human, ai])

        return selected_messages


def _load_examples(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # pragma: no cover - configuration level failure
        logger.error("Failed to load few-shot examples from %s: %s", path, exc)
        return []

    examples: List[Dict] = []
    for item in raw:
        question = item.get("question")
        answer = item.get("ground_truth") or item.get("answer")
        contexts = item.get("contexts", [])
        if not question or not answer:
            continue
        examples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
        })
    return examples


def _build_human_message(example: Dict) -> str:
    contexts = example.get("contexts") or []
    context_block = "\n".join(f"- {ctx}" for ctx in contexts[:3])
    if context_block:
        context_block = f"\nRelevant Context:\n{context_block}"
    return f"Question: {example['question']}{context_block}"


@lru_cache(maxsize=None)
def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
