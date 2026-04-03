"""Bearer token factory example — dynamic auth for OAuth/OIDC workloads.

Usage:
    K2_API_HOST=http://localhost:8000 python -m sdk.examples.auth_factory
"""

import os


def get_oauth_token() -> str:
    """Simulate an OAuth token fetch (replace with your real provider)."""
    # In production: call your OAuth provider here
    return os.environ.get("K2_BEARER_TOKEN", "demo-token")


def main() -> None:
    from sdk import Knowledge2

    client = Knowledge2(
        api_host=os.environ.get("K2_API_HOST", "https://api.knowledge2.ai"),
        bearer_token_factory=get_oauth_token,
        token_cache_ttl=300,  # cache for 5 minutes
    )

    # Pre-flight check
    if not client.is_authenticated():
        print("No auth configured!")
        return

    print(f"Authenticated: {client.is_authenticated()}")


if __name__ == "__main__":
    main()
