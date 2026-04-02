"""LLM generation module using Ollama with LangChain LCEL chains."""

import time
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config import load_config
from src.generation.dspy_optimizer import DSpyPromptOptimizer
from src.generation.few_shot_selector import FewShotSelector
from src.generation.prompt_templates import format_context, get_template
from src.logging.logger import get_logger

logger = get_logger("generation")


class Generator:
    """Wraps Ollama LLM for RAG answer generation via LCEL chain.

    Uses ChatOllama (chat model) + ChatPromptTemplate for structured
    system/human role separation, composed via LCEL pipe syntax.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        llm_cfg = config.get("llm", {})
        self._model = llm_cfg.get("model", "qwen2.5:7b")
        self._base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self._temperature = llm_cfg.get("temperature", 0.1)
        self._num_ctx = llm_cfg.get("num_ctx", 4096)

        self._llm = ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
        )
        self._output_parser = StrOutputParser()
        self._few_shot_selector = FewShotSelector(config=config)
        self._dspy_optimizer = DSpyPromptOptimizer(config=config)

        logger.info(
            f"Generator initialized: model={self._model}, "
            f"temperature={self._temperature}, num_ctx={self._num_ctx}"
        )

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        template_name: str = "default_v1",
        repo_map_text: Optional[str] = None,
    ) -> Dict:
        """Generate an answer using retrieved context and LLM via LCEL chain.

        Builds an LCEL chain: ChatPromptTemplate | ChatOllama | StrOutputParser

        Args:
            query: User's question.
            context_chunks: List of retrieved chunk dicts with 'content' and 'metadata'.
            template_name: Name of the prompt template to use.
            repo_map_text: Optional repo map text to prepend to context.

        Returns:
            Dict with keys: answer, model, prompt_template,
            context_token_count, generation_latency_ms.
        """
        start = time.perf_counter()

        # Build context string from chunks
        context_str = format_context(context_chunks)

        # Prepend repo map if provided
        if repo_map_text:
            context_str = (
                "## Repository Structure Overview\n"
                f"{repo_map_text}\n\n"
                "## Retrieved Code/Documents\n"
                f"{context_str}"
            )

        # Get ChatPromptTemplate and compose LCEL chain
        prompt_template = get_template(template_name)

        # Approximate token count (rough: 1 token ≈ 1.5 chars for mixed zh/en)
        context_token_count = len(context_str) * 2 // 3

        logger.debug(
            f"Prompt assembled: template={template_name}, "
            f"context_tokens≈{context_token_count}"
        )

        messages = prompt_template.format_messages(
            context=context_str,
            question=query,
        )

        few_shot_messages = self._few_shot_selector.select(query)
        if few_shot_messages:
            # Insert few-shot messages between context and final user question for coherence.
            if len(messages) >= 2:
                base_messages = [messages[0], messages[1]]
                base_messages.extend(few_shot_messages)
                base_messages.extend(messages[2:])
                messages = base_messages
            else:
                messages = [messages[0]] + few_shot_messages + messages[1:]
            logger.debug(
                "Few-shot examples injected: %s messages", len(few_shot_messages)
            )

        llm_result = self._llm.invoke(messages)
        answer = self._output_parser.invoke(llm_result)

        refined_answer = self._dspy_optimizer.refine_answer(
            question=query,
            context=context_str,
            baseline_answer=answer,
        )
        if refined_answer:
            answer = refined_answer
            logger.debug("Answer refined via DSPy optimizer")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Generated answer: model={self._model}, "
            f"ctx_tokens≈{context_token_count}, "
            f"answer_len={len(answer)}, {elapsed_ms}ms"
        )

        return {
            "answer": answer,
            "model": self._model,
            "prompt_template": template_name,
            "context_token_count": context_token_count,
            "generation_latency_ms": elapsed_ms,
        }

    @property
    def model_name(self) -> str:
        """Return the LLM model name."""
        return self._model
