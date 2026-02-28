"""Example 09: SDK-native LangChain integration for K2.

Demonstrates:
- auth via env vars
- top_k + metadata filters
- hybrid retrieval options
- optional ingestion/indexing helpers via LangChain tools

Run:
  export K2_API_KEY=...
  export K2_CORPUS_ID=...
  export K2_BASE_URL=http://api:8000   # local
  # OR export K2_BASE_URL=https://api-dev.knowledge2.ai  # dev
  python examples/09_langchain_k2_sdk/langchain_k2_sdk.py
"""

from __future__ import annotations

import os

from sdk.integrations.langchain import K2LangChainRetriever, create_k2_langchain_tools


def main() -> None:
    api_key = os.getenv("K2_API_KEY")
    corpus_id = os.getenv("K2_CORPUS_ID")
    if not api_key or not corpus_id:
        raise SystemExit("K2_API_KEY and K2_CORPUS_ID are required")

    api_host = os.getenv("K2_BASE_URL", "https://api.knowledge2.ai")

    retriever = K2LangChainRetriever(
        api_key=api_key,
        api_host=api_host,
        corpus_id=corpus_id,
        top_k=5,
        filters={"topic": "search"},
        hybrid={
            "enabled": True,
            "fusion_mode": "rrf",
            "rrf_k": 60,
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
        },
    )

    query = "How does hybrid retrieval work in Knowledge2?"
    docs = retriever.invoke(query)

    print(f"Query: {query}\n")
    print(f"Returned {len(docs)} LangChain Document objects")
    for idx, doc in enumerate(docs, start=1):
        score = doc.metadata.get("score")
        print(f"{idx:02d}. score={score} text={doc.page_content[:140].replace(chr(10), ' ')}")

    tools = create_k2_langchain_tools(
        api_key=api_key,
        api_host=api_host,
        corpus_id=corpus_id,
        default_top_k=3,
    )
    print("\nCreated LangChain tools:")
    tool_map = {tool.name: tool for tool in tools}
    for tool in tools:
        print(f"- {tool.name}")

    # Optional: grounded generation (requires server-side LLM config)
    try:
        gen = tool_map["k2_generate_answer"].invoke(
            {"query": "In one sentence, define hybrid retrieval in Knowledge2.", "top_k": 3}
        )
        answer = (gen or {}).get("answer") or ""
        if answer:
            print("\nGenerated answer:")
            print(answer)
    except Exception as exc:
        print(f"\nGeneration skipped/failed: {exc}")


if __name__ == "__main__":
    main()
