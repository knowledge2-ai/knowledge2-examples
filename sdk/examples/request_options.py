"""Per-call request options — override timeout and retry for specific calls.

Usage:
    K2_API_KEY=... python -m sdk.examples.request_options
"""

import os

from sdk import ClientTimeouts, Knowledge2, RequestOptions


def main() -> None:
    client = Knowledge2(
        api_host=os.environ.get("K2_API_HOST", "https://api.knowledge2.ai"),
        api_key=os.environ["K2_API_KEY"],
    )

    # Longer timeout for a known-slow bulk operation
    slow_opts = RequestOptions(
        timeout=ClientTimeouts(read=300),
        max_retries=5,
    )

    # Passthrough tracing headers
    traced_opts = RequestOptions(
        passthrough_headers={
            "X-Request-ID": "example-123",
            "X-Correlation-ID": "trace-abc",
        },
    )

    # Use per-call options
    corpora = client.list_corpora(request_options=traced_opts)
    print(f"Found {corpora.total} corpora (with tracing headers)")

    # Raw response access
    raw = client.with_raw_response.list_corpora()
    print(f"Status: {raw.status_code}")
    print(f"Headers: {dict(list(raw.headers.items())[:3])}...")
    print(f"Parsed: {type(raw.parsed).__name__} with {len(raw.parsed)} items")


if __name__ == "__main__":
    main()
