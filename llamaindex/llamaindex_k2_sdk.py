"""Example 10: SDK-native LlamaIndex integration for K2.

Demonstrates:
- K2LlamaIndexRetriever
- K2LlamaIndexVectorStore (doc-centric adapter)
- top_k + metadata filters + hybrid retrieval

Run:
  export K2_API_KEY=...
  export K2_CORPUS_ID=...
  export K2_BASE_URL=http://api:8000   # local
  # OR export K2_BASE_URL=https://api-dev.knowledge2.ai  # dev
  python examples/10_llamaindex_k2_sdk/llamaindex_k2_sdk.py
"""

from __future__ import annotations

import os

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from sdk.integrations.llamaindex import (
    K2LlamaIndexRetriever,
    K2LlamaIndexVectorStore,
    create_k2_llamaindex_tools,
)


def main() -> None:
    api_key = os.getenv("K2_API_KEY")
    corpus_id = os.getenv("K2_CORPUS_ID")
    if not api_key or not corpus_id:
        raise SystemExit("K2_API_KEY and K2_CORPUS_ID are required")

    api_host = os.getenv("K2_BASE_URL", "https://api.knowledge2.ai")

    retriever = K2LlamaIndexRetriever(
        api_key=api_key,
        api_host=api_host,
        corpus_id=corpus_id,
        top_k=5,
        hybrid={
            "enabled": True,
            "fusion_mode": "rrf",
            "rrf_k": 60,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
        },
    )

    nodes = retriever.retrieve("How does hybrid retrieval work in Knowledge2?")
    print(f"Retrieved {len(nodes)} nodes from K2LlamaIndexRetriever")
    for idx, node in enumerate(nodes, start=1):
        text = node.node.get_content().replace("\n", " ")
        print(f"{idx:02d}. score={node.score} text={text[:140]}")

    vector_store = K2LlamaIndexVectorStore(
        api_key=api_key,
        api_host=api_host,
        corpus_id=corpus_id,
        auto_index_on_add=False,
        top_k=3,
    )
    added_ids = vector_store.add(
        [
            TextNode(
                text="Knowledge2 combines dense and sparse retrieval with configurable fusion.",
                metadata={"topic": "search", "source": "llamaindex-example"},
            )
        ]
    )
    print(f"Added via vector store adapter: {added_ids}")

    result = vector_store.query(VectorStoreQuery(query_str="hybrid retrieval", similarity_top_k=2))
    print(f"Vector store query returned ids={result.ids}")

    tools = create_k2_llamaindex_tools(
        api_key=api_key,
        api_host=api_host,
        corpus_id=corpus_id,
        default_top_k=3,
    )
    print("Created LlamaIndex tools:")
    tool_map = {tool.metadata.name: tool for tool in tools}
    for tool in tools:
        print(f"- {tool.metadata.name}")

    # Optional: grounded generation (requires server-side LLM config)
    try:
        out = tool_map["k2_generate_answer"].call(
            query="In one sentence, define hybrid retrieval in Knowledge2.",
            top_k=3,
        )
        gen = getattr(out, "raw_output", None) or {}
        answer = gen.get("answer") or ""
        if answer:
            print("Generated answer:")
            print(answer)
    except Exception as exc:
        print(f"Generation skipped/failed: {exc}")


if __name__ == "__main__":
    main()
