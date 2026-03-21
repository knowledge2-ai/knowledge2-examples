"""Pagination: iter_documents vs list_documents manual pagination, iter with filters."""

from __future__ import annotations

import os

from sdk import Knowledge2, Knowledge2Error

try:
    api_key = os.environ.get("K2_API_KEY")
    if not api_key:
        raise SystemExit("K2_API_KEY is required")

    client = Knowledge2(api_key=api_key)
    corpus_id = os.environ.get("K2_CORPUS_ID", "corpus-123")

    # --- Page[T]: single-page results from list_* methods ---
    page = client.list_documents(corpus_id, limit=20, offset=0)  # returns Page[T]

    # Page attributes and protocols
    print(f"Total across all pages: {page.total}")  # page.total
    print(f"Items on this page:     {page.items}")  # page.items  (list[T])
    print(f"Page length:            {len(page)}")  # len(page)
    print(f"Page is truthy:         {bool(page)}")  # bool(page)

    for doc in page:  # iterate over items on this page
        print(doc.get("id"), doc.get("source_uri", ""))

    # Manual pagination loop using Page[T]
    offset = 0
    limit = 20
    while True:
        page = client.list_documents(corpus_id, limit=limit, offset=offset)
        if not page:
            break
        for doc in page:
            print(doc.get("id"), doc.get("source_uri", ""))
        offset += len(page)
        if len(page) < limit:
            break

    # --- SyncPager[T]: auto-pagination from iter_* methods ---
    # iter_documents returns a SyncPager that lazily fetches successive pages.
    pager = client.iter_documents(corpus_id, limit=50)  # SyncPager[T]

    for item in pager:
        print(item.get("id"), item.get("source_uri", ""))

    # iter_documents with filters
    for item in client.iter_documents(
        corpus_id,
        limit=50,
        status="indexed",
        source="doc://",
    ):
        print(item.get("id"))

    # Collect all items into a list
    all_docs = list(client.iter_documents(corpus_id, limit=100))
    print(f"Total documents: {len(all_docs)}")

except Knowledge2Error as e:
    print(f"API error: {e}")
    raise
