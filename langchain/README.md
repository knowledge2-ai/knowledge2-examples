# Example 09: K2 SDK + LangChain

This example uses first-class SDK integrations from `sdk.integrations.langchain`.

## What it demonstrates

- Authentication via `K2_API_KEY`
- Retrieval with `top_k`, metadata `filters`, and hybrid config
- LangChain `Document` outputs via `K2LangChainRetriever`
- Framework tools (`k2_search`, `k2_ingest_text`, `k2_build_indexes`, `k2_generate_answer`)

## Install

```bash
pip install -e ".[langchain]"
```

## Run against local stack

```bash
export K2_BASE_URL="http://api:8000"
export K2_API_KEY="local-admin-token"
export K2_CORPUS_ID="<your-local-corpus-id>"
python examples/09_langchain_k2_sdk/langchain_k2_sdk.py
```

## Run against dev

```bash
export K2_BASE_URL="https://api-dev.knowledge2.ai"
export K2_API_KEY="<dev-api-key>"
export K2_CORPUS_ID="<dev-corpus-id>"
python examples/09_langchain_k2_sdk/langchain_k2_sdk.py
```

## Upload files (single file or directory)

Most real apps start with file/folder ingestion, then retrieval.

```bash
export K2_BASE_URL="https://api-dev.knowledge2.ai"
export K2_API_KEY="<dev-api-key>"
export K2_CORPUS_ID="<dev-corpus-id>"

python examples/09_langchain_k2_sdk/langchain_upload_files.py --dir /path/to/docs
python examples/09_langchain_k2_sdk/langchain_upload_files.py --file /path/to/a.md --file /path/to/b.pdf
```
