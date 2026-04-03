"""Example 10b: Upload files/folders to K2, then use LlamaIndex integration.

Typical workflow:
1) Upload real files (single file or whole directory) using the core SDK.
2) Build indexes.
3) Retrieve / search / generate using LlamaIndex adapters.

Run:
  export K2_API_KEY=...
  export K2_CORPUS_ID=...
  export K2_BASE_URL=http://api:8000   # local
  # OR export K2_BASE_URL=https://api-dev.knowledge2.ai  # dev

  python examples/10_llamaindex_k2_sdk/llamaindex_upload_files.py --dir /path/to/docs
  python examples/10_llamaindex_k2_sdk/llamaindex_upload_files.py --file /path/to/a.pdf --file /path/to/b.md

Notes:
- File ingestion is done via the core SDK (`Knowledge2.upload_document(file_path=...)`).
- LlamaIndex integrations cover retrieval, a doc-centric VectorStore adapter, and tool workflows.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from sdk import Knowledge2
from sdk.integrations.llamaindex import K2LlamaIndexRetriever, create_k2_llamaindex_tools


def _wait_for_job(
    client: Knowledge2, job_id: str, *, poll_s: float = 2.0, timeout_s: float = 900.0
) -> None:
    start = time.monotonic()
    while True:
        job = client.get_job(job_id)
        status = job.get("status")
        if status in {"succeeded", "failed", "canceled"}:
            if status != "succeeded":
                raise RuntimeError(
                    job.get("error_message") or f"job {job_id} failed: status={status}"
                )
            return
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for job {job_id}")
        time.sleep(poll_s)


def _collect_files(*, ingest_dir: str | None, ingest_files: list[str]) -> list[Path]:
    files: list[Path] = []
    if ingest_dir:
        root = Path(ingest_dir).expanduser()
        if not root.is_dir():
            raise SystemExit(f"--dir must be a directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(path)
    for item in ingest_files:
        path = Path(item).expanduser()
        if not path.is_file():
            raise SystemExit(f"--file must be a file: {path}")
        files.append(path)

    seen: set[Path] = set()
    out: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload files/dirs to K2 and query via LlamaIndex")
    parser.add_argument("--dir", dest="ingest_dir", help="Upload all files under this directory")
    parser.add_argument(
        "--file", action="append", default=[], dest="ingest_files", help="Upload a specific file"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Upload all selected files as a single batch job"
    )
    parser.add_argument(
        "--no-index", action="store_true", help="Skip building indexes after upload"
    )
    parser.add_argument(
        "--query", default="What is in this corpus?", help="Query to run after indexing"
    )
    args = parser.parse_args()

    api_key = os.getenv("K2_API_KEY")
    corpus_id = os.getenv("K2_CORPUS_ID")
    if not api_key or not corpus_id:
        raise SystemExit("K2_API_KEY and K2_CORPUS_ID are required")

    api_host = os.getenv("K2_BASE_URL", "https://api.knowledge2.ai")
    client = Knowledge2(api_key=api_key, api_host=api_host)

    files = _collect_files(ingest_dir=args.ingest_dir, ingest_files=args.ingest_files)
    if not files:
        raise SystemExit("Provide at least one --file or a --dir")

    print(f"Uploading {len(files)} file(s) to corpus {corpus_id}...")
    if args.batch:
        payload = [(p.name, p.read_bytes()) for p in files]
        resp = client.upload_files_batch(corpus_id, payload, auto_index=False, wait=False)
        job_id = (resp or {}).get("job_id")
        print(f"  batch uploaded count={(resp or {}).get('count')} job_id={job_id}")
        if job_id:
            _wait_for_job(client, job_id)
    else:
        for idx, path in enumerate(files, start=1):
            source_uri = f"file://{path.name}"
            resp = client.upload_document(
                corpus_id,
                file_path=str(path),
                source_uri=source_uri,
                metadata={"filename": path.name, "example": "llamaindex_upload_files"},
                auto_index=False,
            )
            job_id = resp.get("job_id")
            print(f"  {idx:02d}/{len(files)} uploaded filename={path.name} job_id={job_id}")
            if job_id:
                _wait_for_job(client, job_id)

    if not args.no_index:
        print("Building dense+sparse indexes...")
        index = client.build_indexes(corpus_id, dense=True, sparse=True, mode="full", wait=False)
        job_id = (index or {}).get("job_id")
        if job_id:
            _wait_for_job(client, job_id)

    retriever = K2LlamaIndexRetriever(client=client, corpus_id=corpus_id, top_k=3)
    nodes = retriever.retrieve(args.query)
    print(f"\nRetriever returned {len(nodes)} node(s) for query={args.query!r}")
    for i, node in enumerate(nodes[:3], start=1):
        text = node.node.get_content().replace("\n", " ")
        print(f"  {i:02d}. score={node.score} text={text[:160]}")

    tools = create_k2_llamaindex_tools(client=client, corpus_id=corpus_id, default_top_k=3)
    tool_map = {tool.metadata.name: tool for tool in tools}

    # Optional: grounded generation (requires server-side LLM config)
    try:
        out = tool_map["k2_generate_answer"].call(query=args.query, top_k=3)
        gen = getattr(out, "raw_output", None) or {}
        answer = gen.get("answer") or ""
        if answer:
            print("\nGenerated answer:\n" + answer)
    except Exception as exc:
        print(f"\nGeneration skipped/failed: {exc}")


if __name__ == "__main__":
    main()
