"""Verify that 4-list RRF gives balanced doc/code results vs old 2-list approach."""

from src.retrieval.hybrid import reciprocal_rank_fusion


def test_four_list_rrf_balances_doc_and_code():
    """With 4 separate RRF lists, doc results should not be dominated by code."""
    doc_dense = [
        {"chunk_id": "doc1", "content": "PDF about AI", "score": 0.8, "metadata": {"content_type": "text"}},
        {"chunk_id": "doc2", "content": "PDF about ML", "score": 0.6, "metadata": {"content_type": "text"}},
    ]
    code_dense = [
        {"chunk_id": "code1", "content": "def train():", "score": 0.7, "metadata": {"content_type": "code"}},
        {"chunk_id": "code2", "content": "class Pipe:", "score": 0.5, "metadata": {"content_type": "code"}},
    ]
    doc_sparse = [
        {"chunk_id": "doc1", "content": "PDF about AI", "score": 5.0, "metadata": {"content_type": "text"}},
        {"chunk_id": "doc3", "content": "PDF summary", "score": 3.0, "metadata": {"content_type": "text"}},
    ]
    code_sparse = [
        {"chunk_id": "code1", "content": "def train():", "score": 8.0, "metadata": {"content_type": "code"}},
        {"chunk_id": "code2", "content": "class Pipe:", "score": 6.0, "metadata": {"content_type": "code"}},
    ]

    # New: 4 separate lists
    fused_4 = reciprocal_rank_fusion([doc_dense, code_dense, doc_sparse, code_sparse], k=60)

    # Old: 2 merged lists
    fused_2 = reciprocal_rank_fusion([doc_dense + code_dense, doc_sparse + code_sparse], k=60)

    # In 4-list approach, doc1 and code1 both appear in 2 lists each,
    # so they should have equal RRF weight (rank-dependent only within their list)
    scores_4 = {r["chunk_id"]: r["rrf_score"] for r in fused_4}
    scores_2 = {r["chunk_id"]: r["rrf_score"] for r in fused_2}

    print("=== 4-list RRF (balanced) ===")
    for r in fused_4[:5]:
        print(f"  {r['chunk_id']}: rrf={r['rrf_score']:.6f} type={r['metadata']['content_type']}")

    print("=== 2-list RRF (old, unbalanced) ===")
    for r in fused_2[:5]:
        print(f"  {r['chunk_id']}: rrf={r['rrf_score']:.6f} type={r['metadata']['content_type']}")

    # Key assertion: in 4-list, doc1 should rank at least as high as code1
    # because doc1 is rank-1 in both its lists, code1 is rank-1 in both its lists
    assert scores_4["doc1"] >= scores_4["code1"], (
        f"doc1 ({scores_4['doc1']}) should >= code1 ({scores_4['code1']}) in balanced RRF"
    )

    # In old 2-list, code1 could outrank doc1 due to merged ranking
    # (code1 gets BM25 score 8.0 > doc1's 5.0, so code1 ranks higher in sparse list)
    print(f"\nBalanced RRF doc1={scores_4['doc1']:.6f} vs code1={scores_4['code1']:.6f}")
    print("PASS: 4-list RRF balances doc and code results correctly")


if __name__ == "__main__":
    test_four_list_rrf_balances_doc_and_code()
