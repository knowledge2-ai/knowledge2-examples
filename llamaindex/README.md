# Example 10: K2 SDK + LlamaIndex

This example uses first-class SDK integrations from `sdk.integrations.llamaindex`.

## What it demonstrates

- `K2LlamaIndexRetriever` for drop-in retrieval
- `K2LlamaIndexVectorStore` (doc-centric adapter)
- Metadata filters, `top_k`, and hybrid retrieval config
- LlamaIndex tools (`k2_search`, `k2_ingest_text`, `k2_build_indexes`, `k2_generate_answer`)

## Install

```bash
pip install -e ".[llamaindex]"
```

## Run against local stack

```bash
export K2_BASE_URL="http://api:8000"
export K2_API_KEY="local-admin-token"
export K2_CORPUS_ID="<your-local-corpus-id>"
python examples/10_llamaindex_k2_sdk/llamaindex_k2_sdk.py
```

## Run against dev

```bash
export K2_BASE_URL="https://api-dev.knowledge2.ai"
export K2_API_KEY="<dev-api-key>"
export K2_CORPUS_ID="<dev-corpus-id>"
python examples/10_llamaindex_k2_sdk/llamaindex_k2_sdk.py
```

## Upload files (single file or directory)

Most real apps start with file/folder ingestion, then retrieval.

```bash
export K2_BASE_URL="https://api-dev.knowledge2.ai"
export K2_API_KEY="<dev-api-key>"
export K2_CORPUS_ID="<dev-corpus-id>"

python examples/10_llamaindex_k2_sdk/llamaindex_upload_files.py --dir /path/to/docs
python examples/10_llamaindex_k2_sdk/llamaindex_upload_files.py --file /path/to/a.md --file /path/to/b.pdf
```

## Capability notes

`K2LlamaIndexVectorStore` maps LlamaIndex add/query/delete operations onto K2 document and search APIs.
Embedding-only vector lookup (`get(text_id)`) is intentionally unsupported because K2 does not expose raw vector lookup endpoints.
