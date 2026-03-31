"""Interactive RAG Q&A entry point."""

import sys

from src.config import load_config
from src.logging.logger import setup_logging
from src.pipeline import RAGPipeline


def main() -> None:
    """Run interactive Q&A loop against the knowledge base."""
    setup_logging()
    config = load_config()
    pipeline = RAGPipeline(config)

    print("=" * 60)
    print("  Local RAG Knowledge Base Q&A")
    print(f"  LLM: {config['llm']['model']}")
    print(f"  Documents in store: {pipeline.vector_store.count}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline.query(question)

        print(f"\nAnswer: {result['answer']}")
        print(f"\n--- Sources (trace_id: {result['trace_id']}) ---")
        for i, src in enumerate(result["sources"], 1):
            print(
                f"  [{i}] {src['file']} "
                f"(score: {src['score']:.4f})"
            )
        print(f"  Total latency: {result['latency_ms']}ms")
        print()


if __name__ == "__main__":
    main()
