from __future__ import annotations

import os
import uuid

from sdk import Knowledge2


def main() -> None:
    api_key = os.getenv("K2_API_KEY")
    if not api_key:
        raise SystemExit("K2_API_KEY is required")

    client = Knowledge2(
        api_host=os.getenv("K2_BASE_URL", "https://api.knowledge2.ai"),
        api_key=api_key,
    )

    project_name = os.getenv("K2_PROJECT_NAME") or f"knowledge2-quickstart-{uuid.uuid4().hex[:8]}"
    project_id = os.getenv("K2_PROJECT_ID")
    if not project_id:
        project_id = client.create_project(project_name)["id"]
    corpus_name = os.getenv("K2_CORPUS_NAME") or f"{project_name}-corpus"
    corpus = client.create_corpus(project_id, corpus_name)

    docs = [
        {
            "source_uri": "doc://overview",
            "raw_text": "Knowledge2 organizes content into corpora and builds dense and sparse indexes for hybrid retrieval.",
            "metadata": {"topic": "overview"},
        },
        {
            "source_uri": "doc://search",
            "raw_text": "Hybrid retrieval combines semantic similarity with exact keyword matching so product terms and concepts both surface in search.",
            "metadata": {"topic": "search"},
        },
        {
            "source_uri": "doc://jobs",
            "raw_text": "Indexing and ingestion run as background jobs. The SDK can wait for those jobs or you can poll them explicitly.",
            "metadata": {"topic": "jobs"},
        },
    ]

    client.upload_documents_batch(corpus["id"], docs, wait=True, auto_index=False)
    client.sync_indexes(corpus["id"], wait=True)

    results = client.search(
        corpus["id"],
        "what is hybrid retrieval",
        top_k=3,
        return_config={"include_text": True, "include_scores": True},
    )
    for result in results["results"]:
        print(result["score"], (result.get("text") or "")[:100])


if __name__ == "__main__":
    main()
