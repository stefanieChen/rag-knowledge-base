"""Tests for evaluation framework: Ragas + DeepEval metric creation and test set loading."""

import pytest


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

class TestTestSet:
    """Tests for evaluation test set loading."""

    def test_load_test_set(self):
        from evaluation.evaluate_ragas import load_test_set

        data = load_test_set("evaluation/test_set/sample_test_set.json")
        assert isinstance(data, list)
        assert len(data) == 5
        for tc in data:
            assert "question" in tc
            assert "ground_truth" in tc
            assert "contexts" in tc
            assert isinstance(tc["contexts"], list)


# ---------------------------------------------------------------------------
# Ragas
# ---------------------------------------------------------------------------

class TestRagas:
    """Tests for Ragas LLM, embeddings, dataset, and metric creation."""

    def test_llm_factory(self):
        from src.config import load_config
        from evaluation.evaluate_ragas import _build_ragas_llm

        config = load_config()
        llm = _build_ragas_llm(config)
        assert "InstructorLLM" in type(llm).__name__

    def test_embeddings(self):
        from src.config import load_config
        from evaluation.evaluate_ragas import _build_ragas_embeddings

        config = load_config()
        emb = _build_ragas_embeddings(config)
        assert "OpenAIEmbeddings" in type(emb).__name__

    def test_static_dataset(self):
        from evaluation.evaluate_ragas import build_evaluation_dataset_static, load_test_set

        data = load_test_set("evaluation/test_set/sample_test_set.json")
        dataset = build_evaluation_dataset_static(data)
        assert len(dataset) == 5

    def test_metric_creation(self):
        from src.config import load_config
        from evaluation.evaluate_ragas import _build_ragas_llm, _build_ragas_embeddings
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            ContextRecall,
            Faithfulness,
        )

        config = load_config()
        llm = _build_ragas_llm(config)
        emb = _build_ragas_embeddings(config)

        assert isinstance(Faithfulness(llm=llm), Faithfulness)
        assert isinstance(AnswerRelevancy(llm=llm, embeddings=emb), AnswerRelevancy)
        assert isinstance(ContextPrecisionWithoutReference(llm=llm), ContextPrecisionWithoutReference)
        assert isinstance(ContextRecall(llm=llm), ContextRecall)


# ---------------------------------------------------------------------------
# DeepEval
# ---------------------------------------------------------------------------

class TestDeepEval:
    """Tests for DeepEval custom LLM, test cases, and metric creation."""

    def test_ollama_llm(self):
        from evaluation.evaluate_deepeval import OllamaDeepEvalLLM
        from src.config import load_config

        llm = OllamaDeepEvalLLM()
        expected_model = load_config().get("llm", {}).get("model", "qwen2.5:3b")
        assert llm.get_model_name() == expected_model

    def test_build_test_cases(self):
        from evaluation.evaluate_deepeval import build_test_cases

        test_data = [
            {
                "question": "What is RAG?",
                "ground_truth": "RAG is Retrieval-Augmented Generation.",
                "contexts": ["RAG combines retrieval with generation."],
            }
        ]
        cases = build_test_cases(test_data)
        assert len(cases) == 1
        assert cases[0].input == "What is RAG?"
        assert cases[0].actual_output == "RAG is Retrieval-Augmented Generation."

    def test_metric_creation(self):
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from evaluation.evaluate_deepeval import OllamaDeepEvalLLM

        llm = OllamaDeepEvalLLM()
        assert isinstance(HallucinationMetric(threshold=0.5, model=llm), HallucinationMetric)
        assert isinstance(AnswerRelevancyMetric(threshold=0.5, model=llm), AnswerRelevancyMetric)
        assert isinstance(FaithfulnessMetric(threshold=0.5, model=llm), FaithfulnessMetric)


# ---------------------------------------------------------------------------
# Evaluation config & runner
# ---------------------------------------------------------------------------

class TestEvaluationConfig:
    """Tests for evaluation configuration and runner imports."""

    def test_config_loaded(self):
        from src.config import load_config

        config = load_config()
        eval_cfg = config.get("evaluation", {})
        assert eval_cfg.get("test_set") is not None
        assert eval_cfg.get("output_dir") is not None

    def test_runner_imports(self):
        from evaluation.run_evaluation import run_ragas, run_deepeval, print_summary

        assert callable(run_ragas)
        assert callable(run_deepeval)
        assert callable(print_summary)
