#!/usr/bin/env python3
"""
Interactive semantic queryer for Qdrant collection of extracted image/PDF texts.

Usage:
  python qdrant_pdf_image_query.py
Commands:
  :help         show tips
  :exit / :quit quit
  show N        request N results (e.g. "show 5")
  open <id>     print the full text payload of a returned point id (integer)
"""

import json
import time
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
data_file = "./data/extracted_pdf_images_text.json"
collection_name = "pdf_image_collection"
default_limit = 3
snippet_max_chars = 300

print("Initializing Qdrant client and sentence encoder...")
client = QdrantClient(url="http://localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load data ---
print(f"Loading data from {data_file}...")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if not data:
    raise ValueError("❌ No data found in JSON file!")

texts: List[str] = [item.get("text", "") for item in data]

# --- Prepare embeddings (only if needed) ---
# We'll check collection / points count; if empty, compute and upsert embeddings
collections = client.get_collections().collections
existing_col_names = [c.name for c in collections]
embeddings = None

# create collection if missing
if collection_name not in existing_col_names:
    print(f"Creating collection '{collection_name}'...")
    # compute embeddings to know vector size
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )
    print("Collection created.")
else:
    print(f"Collection '{collection_name}' exists. Checking points count...")
    info = client.get_collection(collection_name=collection_name)
    points_count = getattr(info, "points_count", None)
    # if points_count is 0 or None, we will compute embeddings and upload
    if points_count:
        print(f"Collection already has {points_count} points.")
    else:
        print("Collection has 0 points or unknown count — will upload embeddings.")
        embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# upload if needed
if embeddings is not None:
    print(f"Uploading {len(texts)} embeddings to '{collection_name}' ...")
    points = []
    for i, emb in enumerate(embeddings):
        # ensure plain python list for vector field if required by client
        vector = emb.tolist() if hasattr(emb, "tolist") else list(np.array(emb))
        payload = {"path": data[i].get("path"), "text": texts[i]}
        points.append(PointStruct(id=i, vector=vector, payload=payload))

    # upsert (safe whether empty or existing)
    client.upsert(collection_name=collection_name, points=points)
    print("Upload completed.")
    # small pause for consistency
    time.sleep(0.2)

print("\n" + "=" * 60)
print("🔍 Qdrant PDF/Image Text Semantic Queryer")
print("=" * 60)
print("Enter a natural-language query to search the collection.")
print("Commands: ':quit' or ':exit' to exit, ':help' for tips, 'open <id>' to see full text.")
print("=" * 60)

def print_hit(rank: int, hit, show_snippet=True):
    """Format and print a single search hit."""
    # Newer qdrant client returns objects with attributes; handle dict-like too
    score = getattr(hit, "score", None) or hit.get("score", None) or 0.0
    payload = getattr(hit, "payload", None) or hit.get("payload", {}) or {}
    pid = getattr(hit, "id", None) or hit.get("id", None)
    path = payload.get("path", "N/A")
    full_text = payload.get("text", "")
    snippet = full_text[:snippet_max_chars].replace("\n", " ") + ("..." if len(full_text) > snippet_max_chars else "")
    print(f"\n{rank}. id={pid}  (score={score:.4f})")
    print(f"   path: {path}")
    if show_snippet:
        print(f"   snippet: {snippet}")

def open_point(point_id: int):
    """Fetch the payload for a single point id and print the full text."""
    try:
        # Try to retrieve the point
        resp = client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True)
        if not resp or len(resp) == 0:
            print(f"❌ No point found with id {point_id}")
            return
        item = resp[0]
        payload = getattr(item, "payload", None) or item.get("payload", {}) or {}
        path = payload.get("path", "N/A")
        text = payload.get("text", "")
        print("\n" + "-" * 60)
        print(f"Full text for id={point_id}  (path: {path})\n")
        print(text)
        print("\n" + "-" * 60)
    except Exception as e:
        print(f"❌ Error retrieving id {point_id}: {e}")

try:
    last_results = []  # store last batch of results (so user can open ids easily)
    while True:
        user_q = input("\n📚 Query> ").strip()
        if not user_q:
            continue

        lower = user_q.lower()
        if lower in (":quit", ":exit", "quit", "exit"):
            print("Exiting. Goodbye! 👋")
            break

        if lower == ":help":
            print("\n💡 Tips:")
            print("  - Ask things like: 'alien invasion', 'OCR of kitchen receipt', 'table of contents', 'license plate text'")
            print("  - Use 'show N' (e.g. 'show 5') to request more results")
            print("  - Use 'open <id>' to print the full found text for a returned id")
            continue

        # open <id> command to display full text
        if lower.startswith("open "):
            try:
                _, id_str = user_q.split(None, 1)
                pid = int(id_str.strip())
                open_point(pid)
            except Exception:
                print("Usage: open <id>  (where <id> is an integer printed next to results)")
            continue

        # support 'show N' anywhere in the query (remove it before encoding)
        limit = default_limit
        if "show " in lower:
            # naive parse: find the token "show " and take the immediate following number
            try:
                parts = lower.split("show ")
                # take the first occurrence
                tail = parts[1]
                num_str = tail.split()[0]
                limit_candidate = int(num_str)
                if limit_candidate > 0:
                    limit = limit_candidate
                # remove the "show N" phrase from the original query (case-sensitive removal)
                # safer: remove the substring in a case-insensitive manner
                idx = user_q.lower().find(f"show {num_str}")
                if idx != -1:
                    user_q = (user_q[:idx] + user_q[idx + len(f"show {num_str}") :]).strip()
            except Exception:
                # ignore parse errors, fall back to default limit
                pass

        # Encode the query
        print("Searching...")
        q_vec = encoder.encode(user_q, convert_to_numpy=True)
        # Qdrant search call
        hits = client.search(
            collection_name=collection_name,
            query_vector=q_vec.tolist(),
            limit=limit,
            with_payload=True,
        )

        if not hits:
            print("❌ No results found. Try a different query.")
            last_results = []
            continue

        print(f"\n✨ Top {len(hits)} results for: '{user_q}'")
        print("-" * 60)
        last_results = hits
        for rank, h in enumerate(hits, start=1):
            print_hit(rank, h)
        print("-" * 60)

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Exiting. 👋")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
