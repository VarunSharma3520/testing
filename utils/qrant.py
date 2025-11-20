# testrag_qdrant_rerank.py
"""
High-quality Qdrant + Cross-Encoder RAG script.
Replaces FAISS usage with Qdrant vector DB.

Requirements:
  pip install sentence-transformers qdrant-client langchain-text-splitters langchain-community pdfplumber
Qdrant server must be running (e.g. docker run -p 6333:6333 qdrant/qdrant)
"""

import warnings
from pathlib import Path
import json
import numpy as np
import time
from collections import Counter
from typing import List

# silence the harmless pypdf "page label" warnings
warnings.filterwarnings("ignore", message="Could not reliably determine page label")
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------- CONFIG ----------------
PDF_PATH = Path("./data/F18-ABCD-000.pdf")
INDEX_DIR = Path("./data/qdrant_rerank")
METADATA_PATH = INDEX_DIR / "qdrant_metadata.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "retriever_collection"

# Embedding & reranker models
EMBEDDING_MODEL = "all-mpnet-base-v2"  # higher-quality embedding
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Retrieval / Rerank settings
QDRANT_TOP_K = 50  # retrieve many for reranking
RERANK_TOP_N = 5  # after rerank, take top N
MMR_LAMBDA = 0.7  # for MMR diversity (0.0 = diverse, 1.0 = relevance only)
USE_MMR = True  # whether to run MMR on reranked candidates (applies on embeddings)
MMR_TOP_K = 5  # final number of chunks to return after MMR

# index metadata key to check for auto-rebuild
INDEX_INFO_KEY = "index_info"
# -----------------------------------------


def try_loaders(path: Path):
    """Try PDFMinerLoader then pdfplumber and return list of langchain-like docs."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # PDFMinerLoader
    try:
        loader = PDFMinerLoader(str(path))
        docs = loader.load()
        print(f"[loader] PDFMinerLoader -> {len(docs)} pages/chunks")
        return docs
    except Exception as e:
        print(f"[loader] PDFMinerLoader failed: {e}")

    # pdfplumber fallback
    try:
        import pdfplumber

        docs = []
        with pdfplumber.open(str(path)) as pdf:
            for i, p in enumerate(pdf.pages):
                text = p.extract_text() or ""
                docs.append(
                    type("D", (), {"page_content": text, "metadata": {"page": i + 1}})()
                )
        print(f"[loader] pdfplumber -> {len(docs)} pages")
        return docs
    except Exception as e:
        print(f"[loader] pdfplumber failed: {e}")

    raise RuntimeError("No PDF loader succeeded.")


def detect_repeated_header_footer(pages: List[str], threshold: float = 0.6):
    """
    Detect header/footer lines that commonly repeat across pages.
    Returns (headers, footers) lists to remove.
    """
    first_lines = [p.splitlines()[0].strip() if p and p.strip() else "" for p in pages]
    last_lines = [p.splitlines()[-1].strip() if p and p.strip() else "" for p in pages]
    n = max(1, len(pages))
    headers = [
        line
        for line, cnt in Counter(first_lines).items()
        if cnt / n >= threshold and line
    ]
    footers = [
        line
        for line, cnt in Counter(last_lines).items()
        if cnt / n >= threshold and line
    ]
    return headers, footers


def strip_headers_footers_on_docs(docs, threshold=0.6):
    pages = [d.page_content or "" for d in docs]
    headers, footers = detect_repeated_header_footer(pages, threshold=threshold)
    if headers or footers:
        print(f"[clean] detected headers: {headers}, footers: {footers}")
    cleaned_docs = []
    for i, d in enumerate(docs):
        txt = d.page_content or ""
        lines = txt.splitlines()
        # remove matching header at start
        if headers and lines:
            for h in headers:
                if lines and lines[0].strip() == h:
                    lines = lines[1:]
                    break
        # remove matching footer at end
        if footers and lines:
            for f in footers:
                if lines and lines[-1].strip() == f:
                    lines = lines[:-1]
                    break
        new_text = "\n".join(lines).strip()
        cleaned_docs.append(
            type("D", (), {"page_content": new_text, "metadata": d.metadata or {}})()
        )
    return cleaned_docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(
        f"[split] produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def prepare_texts_and_meta(chunks):
    texts = []
    metadatas = []
    for i, c in enumerate(chunks):
        meta = c.metadata or {}
        page_info = meta.get("page", meta.get("page_number", i + 1))
        text = (c.page_content or "").replace("\n", " ").strip()
        texts.append(text)
        metadatas.append(
            {
                "id": i,
                "page": page_info,
                "text": text,
            }
        )
    return texts, metadatas


def build_embeddings(texts: List[str], model_name=EMBEDDING_MODEL, batch_size=64):
    print(f"[embed] loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings, model


# ---------------- QDRANT helpers ----------------
def init_qdrant_client(host=QDRANT_HOST, port=QDRANT_PORT):
    client = QdrantClient(host=host, port=port)
    return client


def create_or_overwrite_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create the collection if missing. If present, try to detect existing vector dimension.
    If the existing dimension is readable and mismatches `vector_size`, delete & recreate.
    If the existing dimension cannot be read (client/version mismatch), warn and keep the
    existing collection (safer than blindly deleting).
    """
    existing = client.get_collections().collections
    names = [c.name for c in existing]
    if collection_name in names:
        # try to read collection info and vector size in a couple of ways (different qdrant-client versions)
        try:
            info = client.get_collection(collection_name)
        except Exception as e:
            print(f"[qdrant] warning: failed to get collection info: {e}. Will assume collection exists and leave it as-is.")
            return

        existing_dim = None
        # try common attribute paths in order (robust to qdrant-client versions)
        try:
            # newer shape: info.params.vectors.size
            existing_dim = info.params.vectors.size
        except Exception:
            try:
                # alternate shape: info.vectors.size
                existing_dim = info.vectors.size
            except Exception:
                # fallback: try dict conversion and inspect keys (best-effort)
                try:
                    info_dict = info.model_dump() if hasattr(info, "model_dump") else info.dict() if hasattr(info, "dict") else None
                    if isinstance(info_dict, dict):
                        # navigate plausibly
                        vec_section = info_dict.get("params") or info_dict.get("vectors") or info_dict.get("payload", None)
                        if isinstance(vec_section, dict):
                            # possible nested structure
                            v = vec_section.get("vectors") or vec_section.get("vectors_config") or vec_section.get("params")
                            if isinstance(v, dict) and "size" in v:
                                existing_dim = v["size"]
                except Exception:
                    existing_dim = None

        if existing_dim is None:
            print(f"[qdrant] collection '{collection_name}' exists but vector dimension could not be determined from client response. Leaving collection intact. (If you want to force recreation, run main(build_new_index=True)).")
            return

        try:
            existing_dim = int(existing_dim)
        except Exception:
            print(f"[qdrant] couldn't parse existing vector dim '{existing_dim}'. Leaving collection intact.")
            return

        if existing_dim != vector_size:
            print(f"[qdrant] existing collection vector size {existing_dim} != {vector_size}, dropping and recreating.")
            client.delete_collection(collection_name=collection_name)
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT)
            )
        else:
            print(f"[qdrant] collection '{collection_name}' exists (dim={vector_size}).")
    else:
        print(f"[qdrant] creating collection '{collection_name}' (dim={vector_size})")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.DOT)
        )


def upload_embeddings_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    metadatas: list,
    batch_size=256,
):
    """
    embeddings: (N, D) numpy
    metadatas: list of dicts with 'id','page','text',...
    """
    n = embeddings.shape[0]
    points = []
    for i in range(n):
        vec = embeddings[i].astype("float32").tolist()
        payload = metadatas[i].copy()
        # Keep payload lightweight (we still keep text but you can choose to store less)
        # id must be unique; we'll use the integer index
        p = PointStruct(id=int(metadatas[i]["id"]), vector=vec, payload=payload)
        points.append(p)
        # upload in batches
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points)
    print(f"[qdrant] uploaded {n} vectors to collection '{collection_name}'.")


def query_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vec: np.ndarray,
    top_k: int = QDRANT_TOP_K,
):
    # query_vec is 1D numpy (already normalized)
    q_list = query_vec.astype("float32").tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=q_list,
        limit=top_k,
        with_payload=True,
    )
    # results: list of ScoredPoint (id, score, payload)
    ids = [int(r.id) for r in results]
    scores = [float(r.score) for r in results]
    payloads = [r.payload for r in results]
    return ids, scores, payloads


# ---------------- end QDRANT helpers ----------------


def save_metadata(metadatas: list, index_info: dict, meta_path: Path):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {INDEX_INFO_KEY: index_info, "metadatas": metadatas}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] metadata -> {meta_path} (n={len(metadatas)})")


def load_metadata(meta_path: Path):
    if not meta_path.exists():
        raise FileNotFoundError("Metadata file not found on disk.")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    index_info = payload.get(INDEX_INFO_KEY, {})
    metadatas = payload.get("metadatas", [])
    return metadatas, index_info


def is_index_compatible(index_info: dict):
    """Check if saved index info matches current configuration; if not, we should rebuild."""
    want = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "cross_encoder": CROSS_ENCODER_MODEL,
    }
    return index_info == want


def rerank_with_crossencoder(
    query: str,
    candidate_texts: List[str],
    cross_encoder_model: CrossEncoder,
    top_n=RERANK_TOP_N,
):
    pairs = [[query, t] for t in candidate_texts]
    scores = cross_encoder_model.predict(pairs)  # higher = more relevant
    ranked = sorted(
        list(zip(candidate_texts, scores)), key=lambda x: x[1], reverse=True
    )
    return ranked[:top_n], scores


def mmr_select(
    cand_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    candidates: List[int],
    top_n=MMR_TOP_K,
    lambda_param=MMR_LAMBDA,
):
    """
    cand_embeddings: (N, D) embeddings for candidate set (already normalized)
    query_embedding: (D,) normalized
    candidates: list of indices into cand_embeddings
    Returns list of selected candidate indices (subset of 'candidates').
    """
    selected = []
    if len(candidates) == 0:
        return selected
    sims_to_query = (
        cand_embeddings @ query_embedding.reshape(-1, 1)
    ).squeeze()  # similarity for ALL cand embeddings
    cand_set = candidates.copy()
    # pick most relevant first
    first = max(cand_set, key=lambda i: sims_to_query[i])
    selected.append(first)
    cand_set.remove(first)
    while len(selected) < top_n and cand_set:
        best_score = None
        best_idx = None
        for c in cand_set:
            sim_q = sims_to_query[c]
            sim_to_selected = (
                max((cand_embeddings[c] @ cand_embeddings[s]) for s in selected)
                if selected
                else 0.0
            )
            score = lambda_param * sim_q - (1 - lambda_param) * sim_to_selected
            if best_score is None or score > best_score:
                best_score = score
                best_idx = c
        if best_idx is None:
            break
        selected.append(best_idx)
        cand_set.remove(best_idx)
    return selected


def main(build_new_index=True):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    client = init_qdrant_client()
    need_rebuild = True
    metadatas = []
    index_info = {}

    if METADATA_PATH.exists() and not build_new_index:
        try:
            metadatas, index_info = load_metadata(METADATA_PATH)
            if is_index_compatible(index_info):
                print(
                    "[index] found compatible metadata on disk. Will try to use existing Qdrant collection."
                )
                need_rebuild = False
            else:
                print("[index] metadata config mismatch — will rebuild collection.")
                need_rebuild = True
        except Exception as e:
            print(f"[index] failed to load existing metadata: {e}. Will rebuild.")
            need_rebuild = True

    if build_new_index or need_rebuild:
        print("[build] extracting text from PDF...")
        docs = try_loaders(PDF_PATH)
        docs = strip_headers_footers_on_docs(docs, threshold=0.6)
        chunks = split_documents(docs)
        texts, metadatas = prepare_texts_and_meta(chunks)
        if len(texts) == 0:
            raise RuntimeError("No text extracted from PDF.")
        # embeddings
        embeddings, emb_model = build_embeddings(texts)
        # normalize embeddings
        vecs = embeddings.astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        # create collection and upload
        vector_dim = vecs.shape[1]
        create_or_overwrite_collection(client, COLLECTION_NAME, vector_dim)
        upload_embeddings_to_qdrant(
            client, COLLECTION_NAME, vecs, metadatas, batch_size=256
        )

        index_info = {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "cross_encoder": CROSS_ENCODER_MODEL,
            "n_chunks": len(texts),
            "built_at": time.time(),
        }
        save_metadata(metadatas, index_info, METADATA_PATH)
    else:
        # load embedding model only for queries
        emb_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[index] using existing Qdrant collection and local metadata.")

    # load cross-encoder reranker
    print(f"[rerank] loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print("\nReady for queries. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break

            # 1) Qdrant retrieval (many candidates)
            q_vec = emb_model.encode([q], convert_to_numpy=True)[0].astype("float32")
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            idxs, scores, payloads = query_qdrant(
                client, COLLECTION_NAME, q_vec, top_k=QDRANT_TOP_K
            )

            candidates = [i for i in idxs if i >= 0]
            candidate_texts = []
            # map ids -> text using payloads if present, otherwise use local metadatas
            if payloads and all(p and "text" in p for p in payloads):
                candidate_texts = [p["text"] for p in payloads]
            else:
                # fallback: lookup from metadatas file
                id_to_text = {md["id"]: md["text"] for md in metadatas}
                candidate_texts = [id_to_text.get(i, "") for i in candidates]

            if not any(candidate_texts):
                print("No candidates retrieved.")
                continue

            # 2) Cross-encoder rerank on the Qdrant candidates
            reranked, raw_scores = rerank_with_crossencoder(
                q, candidate_texts, cross_encoder, top_n=len(candidate_texts)
            )
            reranked_texts = [t for t, s in reranked]  # sorted by score desc

            # If USE_MMR: perform MMR selection on the top-M reranked candidates using their embeddings
            final_selected_texts = []
            if USE_MMR:
                # prepare candidate embeddings (in reranked order)
                # We compute embeddings for candidate_texts (not ideal but works). Normalize.
                cand_embs = emb_model.encode(candidate_texts, convert_to_numpy=True)
                cand_embs = cand_embs.astype("float32")
                cand_embs = cand_embs / (
                    np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12
                )

                # query embedding
                q_emb = emb_model.encode([q], convert_to_numpy=True)[0].astype(
                    "float32"
                )
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

                # Map reranked order back to candidate indices (we reranked candidate_texts)
                top_for_mmr = min(len(candidate_texts), 20)
                indices_for_mmr = list(
                    range(top_for_mmr)
                )  # indices into candidate_texts
                selected_rel_indices = mmr_select(
                    cand_embs,
                    q_emb,
                    indices_for_mmr,
                    top_n=MMR_TOP_K,
                    lambda_param=MMR_LAMBDA,
                )
                final_selected_texts = [
                    candidate_texts[i] for i in selected_rel_indices
                ]
            else:
                final_selected_texts = reranked_texts[:RERANK_TOP_N]

            # Print top results with provenance
            print("\nTop retrieved & selected chunks:")
            for i, txt in enumerate(final_selected_texts, 1):
                preview = txt[:400].replace("\n", " ")
                # find original metadata (approx) - search preview in metadatas
                meta_page = None
                for md in metadatas:
                    if md["text"].startswith(txt[:50]):
                        meta_page = md.get("page")
                        break
                print(f"{i}. page={meta_page} preview: {preview}...\n")

            # Build concise prompt for LLM
            context = "\n\n".join(
                [f"[DOC {i+1}]\n{t}" for i, t in enumerate(final_selected_texts)]
            )
            prompt = f"""You are an assistant for question-answering tasks.
Use ONLY the following documents to answer the question. If the answer is not contained in the documents, say "I don't know."
Be concise (max 3 sentences).

Question: {q}
Documents:
{context}

Answer:"""
            print("\nPrepared prompt (send to your LLM):\n")
            print(prompt)
            # Optionally: call your LLM here.
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError during query: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("Goodbye.")


if __name__ == "__main__":
    main(build_new_index=False)  # set True to force rebuild every run
