from __future__ import annotations

import os
import uuid
from typing import Iterable

from sdk import Knowledge2
from sdk.types.search import SearchResult


def _print_hits(results: Iterable[SearchResult]) -> None:
    for idx, result in enumerate(results, start=1):
        text = (result.get("text") or "").strip().replace("\n", " ")
        score = result.get("score")
        display_score = f"{score:.4f}" if score is not None else "n/a"
        print(f"{idx:02d}. score={display_score} text={text[:120]}")


def main() -> None:
    api_key = os.getenv("K2_API_KEY")
    if not api_key:
        raise SystemExit("K2_API_KEY is required")

    client = Knowledge2(
        api_host=os.getenv("K2_BASE_URL", "https://api.knowledge2.ai"),
        api_key=api_key,
    )
    idempotency_suffix = os.getenv("K2_IDEMPOTENCY_SUFFIX", uuid.uuid4().hex[:8])

    def _key(base: str) -> str:
        return f"{base}-{idempotency_suffix}"

    project_id = os.getenv("K2_PROJECT_ID")
    project_name = (
        os.getenv("K2_PROJECT_NAME") or f"knowledge2-optimize-demo-{uuid.uuid4().hex[:8]}"
    )
    if not project_id:
        project = client.create_project(project_name)
        project_id = project["id"]

    corpus_name = os.getenv("K2_CORPUS_NAME") or f"{project_name}-corpus"
    corpus = client.create_corpus(
        project_id,
        corpus_name,
        description="Sample corpus for the retrieval optimization lifecycle.",
    )
    corpus_id = corpus["id"]

    docs = [
        {
            "source_uri": "doc://hybrid-overview",
            "raw_text": (
                "Knowledge² blends dense and sparse retrieval. Dense retrieval captures semantic "
                "similarity while BM25 captures exact keywords and terminology."
            ),
            "metadata": {"topic": "hybrid", "product": "knowledge2"},
        },
        {
            "source_uri": "doc://rrf",
            "raw_text": (
                "Reciprocal rank fusion combines dense and sparse rankings into a shared result "
                "set. Lower rrf_k values emphasize top-ranked hits more aggressively."
            ),
            "metadata": {"topic": "rrf", "product": "knowledge2"},
        },
        {
            "source_uri": "doc://bm25",
            "raw_text": (
                "BM25 improves retrieval for product names, acronyms, and exact operational terms. "
                "Good example queries help the platform tune sparse retrieval defaults."
            ),
            "metadata": {"topic": "bm25", "product": "knowledge2"},
        },
        {
            "source_uri": "doc://optimize",
            "raw_text": (
                "Use indexes:optimize to sample or ingest example queries and tune retrieval "
                "defaults. The optimize job evaluates candidate BM25 and RRF settings and stores "
                "the best defaults for the corpus."
            ),
            "metadata": {"topic": "optimize", "product": "knowledge2"},
        },
        {
            "source_uri": "doc://operations",
            "raw_text": (
                "The query profile stores example queries, dataset hints, and optimize state. "
                "You can read the profile before or after running retrieval optimization."
            ),
            "metadata": {"topic": "operations", "product": "knowledge2"},
        },
    ]
    ingest = client.upload_documents_batch_and_wait(
        corpus_id,
        docs,
        idempotency_key=_key("demo-ingest"),
        auto_index=False,
    )
    print("Ingest batch:", ingest["batch_id"])
    print("Ingest job:", ingest["job_id"])
    print("Indexed docs:", len(ingest.get("doc_ids") or []))

    build = client.sync_indexes(
        corpus_id,
        idempotency_key=_key("demo-sync"),
        wait=True,
    )
    print("Index sync job:", build["job_id"])

    query = "How do I improve hybrid retrieval quality?"
    baseline = client.search(
        corpus_id,
        query,
        top_k=5,
        return_config={"include_text": True, "include_scores": True},
    )
    print("\nBaseline results:")
    _print_hits(baseline["results"])

    profile = client.get_query_profile(corpus_id)
    print("\nCurrent query profile example count:", len(profile.get("example_queries") or []))

    optimize = client.optimize_indexes(
        corpus_id,
        example_queries=[
            "how does hybrid retrieval work",
            "what is bm25 tuning",
            "how does rrf combine dense and sparse search",
            "how do example queries improve retrieval defaults",
        ],
        query_count=25,
        top_k=10,
        metric="ndcg",
        idempotency_key=_key("demo-optimize"),
        wait=True,
    )
    print("Optimize job:", optimize["job_id"])

    updated_profile = client.get_query_profile(corpus_id)
    print("Updated query profile example count:", len(updated_profile.get("example_queries") or []))

    tuned = client.search(
        corpus_id,
        query,
        top_k=5,
        return_config={"include_text": True, "include_scores": True},
    )
    print("\nPost-optimize results:")
    _print_hits(tuned["results"])


if __name__ == "__main__":
    main()
