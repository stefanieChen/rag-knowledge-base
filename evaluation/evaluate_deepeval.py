"""RAG evaluation using DeepEval framework.

Evaluates RAG pipeline quality using:
- Hallucination: Does the answer contain fabricated information?
- Answer Relevancy: Is the answer relevant to the question?
- Faithfulness: Is the answer grounded in the provided context?

Uses a custom Ollama LLM wrapper via DeepEvalBaseLLM.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("evaluation.deepeval")


class OllamaDeepEvalLLM(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper for Ollama models.

    Uses the OpenAI-compatible endpoint exposed by Ollama
    to serve as the LLM judge for DeepEval metrics.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        llm_cfg = config.get("llm", {})
        self._model_name = llm_cfg.get("model", "qwen2.5:3b")
        self._base_url = llm_cfg.get("base_url", "http://localhost:11434")

        from openai import OpenAI
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",
        )

    def load_model(self):
        """Return the OpenAI client (model is already loaded in Ollama)."""
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the Ollama model.

        Args:
            prompt: Input prompt string.

        Returns:
            Generated text response.
        """
        client = self.load_model()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Async generate — falls back to sync for Ollama.

        Args:
            prompt: Input prompt string.

        Returns:
            Generated text response.
        """
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name


def build_test_cases(
    test_data: List[Dict],
) -> List[LLMTestCase]:
    """Build DeepEval test cases from test data.

    Args:
        test_data: List of dicts with keys:
            'question', 'ground_truth', 'contexts',
            and optionally 'response' (actual LLM answer).

    Returns:
        List of LLMTestCase instances.
    """
    cases = []
    for td in test_data:
        case = LLMTestCase(
            input=td["question"],
            actual_output=td.get("response", td["ground_truth"]),
            expected_output=td["ground_truth"],
            retrieval_context=td.get("contexts", []),
            context=td.get("contexts", []),
        )
        cases.append(case)
    return cases


def build_test_cases_from_pipeline(
    test_data: List[Dict],
    pipeline,
) -> List[LLMTestCase]:
    """Run the RAG pipeline and build DeepEval test cases with actual outputs.

    Args:
        test_data: List of dicts with 'question', 'ground_truth', 'contexts'.
        pipeline: RAGPipeline instance.

    Returns:
        List of LLMTestCase instances with real pipeline outputs.
    """
    cases = []
    for td in test_data:
        question = td["question"]
        result = pipeline.query(question)

        retrieved_contexts = [
            s.get("content_preview", "") for s in result.get("sources", [])
        ]

        case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=td["ground_truth"],
            retrieval_context=retrieved_contexts,
            context=td.get("contexts", []),
        )
        cases.append(case)
        logger.info(f"Generated DeepEval case for: '{question[:50]}'")

    return cases


def run_deepeval_evaluation(
    test_cases: List[LLMTestCase],
    config: Optional[dict] = None,
    metrics_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run DeepEval evaluation on the given test cases.

    Args:
        test_cases: List of LLMTestCase instances.
        config: Application config dict.
        metrics_to_run: List of metric names. Options:
            'hallucination', 'answer_relevancy', 'faithfulness'.
            Defaults to all three.

    Returns:
        Dict with metric scores and evaluation metadata.
    """
    if config is None:
        config = load_config()

    if metrics_to_run is None:
        metrics_to_run = ["hallucination", "answer_relevancy", "faithfulness"]

    llm = OllamaDeepEvalLLM(config)

    # Build metrics
    metrics = []
    metric_names = []
    if "hallucination" in metrics_to_run:
        metrics.append(HallucinationMetric(threshold=0.5, model=llm))
        metric_names.append("hallucination")
    if "answer_relevancy" in metrics_to_run:
        metrics.append(AnswerRelevancyMetric(threshold=0.5, model=llm))
        metric_names.append("answer_relevancy")
    if "faithfulness" in metrics_to_run:
        metrics.append(FaithfulnessMetric(threshold=0.5, model=llm))
        metric_names.append("faithfulness")

    logger.info(
        f"Running DeepEval evaluation: {len(test_cases)} cases, "
        f"metrics={metric_names}"
    )

    start = time.perf_counter()

    # Evaluate each test case against each metric
    per_case_results = []
    aggregate_scores = {name: [] for name in metric_names}

    for i, test_case in enumerate(test_cases):
        case_result = {"input": test_case.input}
        for metric in metrics:
            try:
                metric.measure(test_case)
                score = metric.score
                case_result[metric.__class__.__name__] = {
                    "score": round(float(score), 4),
                    "reason": getattr(metric, "reason", ""),
                    "passed": metric.is_successful(),
                }
                aggregate_scores[metric_names[metrics.index(metric)]].append(
                    float(score)
                )
            except Exception as e:
                logger.warning(
                    f"Metric {metric.__class__.__name__} failed on case {i}: {e}"
                )
                case_result[metric.__class__.__name__] = {
                    "score": None,
                    "reason": str(e),
                    "passed": False,
                }
        per_case_results.append(case_result)

    elapsed_s = round(time.perf_counter() - start, 1)

    # Compute averages
    avg_scores = {}
    for name, scores in aggregate_scores.items():
        if scores:
            avg_scores[name] = round(sum(scores) / len(scores), 4)

    logger.info(f"DeepEval evaluation complete ({elapsed_s}s): {avg_scores}")

    return {
        "framework": "deepeval",
        "scores": avg_scores,
        "num_cases": len(test_cases),
        "evaluation_time_s": elapsed_s,
        "per_case": per_case_results,
    }


def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dict.
        output_path: Output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {output_path}")
