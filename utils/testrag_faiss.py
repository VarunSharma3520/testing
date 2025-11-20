# testrag_faiss_rerank.py
"""
High-quality FAISS + Cross-Encoder RAG script.

Requirements:
  pip install sentence-transformers faiss-cpu langchain-text-splitters langchain-community pdfplumber
(If faiss-cpu isn't available on your platform, use conda or replace with another ANN.)
"""

import warnings
from pathlib import Path
import json
import numpy as np
import faiss
import time
from collections import Counter
from typing import List

# silence the harmless pypdf "page label" warnings
warnings.filterwarnings("ignore", message="Could not reliably determine page label")

# loaders & splitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# sentence-transformers for embeddings and cross-encoder
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as e:
    raise RuntimeError(
        "Please install sentence-transformers: pip install sentence-transformers"
    )

# ---------------- CONFIG ----------------
PDF_PATH = Path("./data/F18-ABCD-000.pdf")
INDEX_DIR = Path("./data/faiss_rerank")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "faiss_metadata.json"

# Embedding & reranker models
EMBEDDING_MODEL = "all-mpnet-base-v2"                     # higher-quality embedding
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval / Rerank settings
FAISS_TOP_K = 50     # retrieve many for reranking
RERANK_TOP_N = 5     # after rerank, take top N
MMR_LAMBDA = 0.7     # for MMR diversity (0.0 = diverse, 1.0 = relevance only)
USE_MMR = True       # whether to run MMR on reranked candidates (applies on embeddings)
MMR_TOP_K = 5        # final number of chunks to return after MMR

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
                docs.append(type("D", (), {"page_content": text, "metadata": {"page": i+1}})())
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
    headers = [line for line, cnt in Counter(first_lines).items() if cnt / n >= threshold and line]
    footers = [line for line, cnt in Counter(last_lines).items() if cnt / n >= threshold and line]
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
        cleaned_docs.append(type("D", (), {"page_content": new_text, "metadata": d.metadata or {}})())
    return cleaned_docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[split] produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def prepare_texts_and_meta(chunks):
    texts = []
    metadatas = []
    for i, c in enumerate(chunks):
        meta = c.metadata or {}
        page_info = meta.get("page", meta.get("page_number", i + 1))
        text = (c.page_content or "").replace("\n", " ").strip()
        texts.append(text)
        metadatas.append({
            "id": i,
            "page": page_info,
            "text": text,
        })
    return texts, metadatas


def build_embeddings(texts: List[str], model_name=EMBEDDING_MODEL, batch_size=64):
    print(f"[embed] loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model


def build_faiss_index(embeddings: np.ndarray):
    vecs = embeddings.astype("float32")
    # L2-normalize -> inner product becomes cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def save_index_and_metadata(index: faiss.Index, metadatas: list, index_path: Path, meta_path: Path, index_info: dict):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    payload = {
        INDEX_INFO_KEY: index_info,
        "metadatas": metadatas
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] index -> {index_path} (n={index.ntotal}), metadata -> {meta_path}")


def load_index_and_metadata(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found on disk.")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    index_info = payload.get(INDEX_INFO_KEY, {})
    metadatas = payload.get("metadatas", [])
    return index, metadatas, index_info


def is_index_compatible(index_info: dict):
    """Check if saved index info matches current configuration; if not, we should rebuild."""
    want = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "cross_encoder": CROSS_ENCODER_MODEL
    }
    return index_info == want


def query_faiss(index: faiss.Index, model: SentenceTransformer, query: str, top_k=FAISS_TOP_K):
    q_vec = model.encode([query], convert_to_numpy=True)[0].astype("float32")
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    D, I = index.search(q_vec.reshape(1, -1), top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    return idxs, scores


def rerank_with_crossencoder(query: str, candidate_texts: List[str], cross_encoder_model: CrossEncoder, top_n=RERANK_TOP_N):
    pairs = [[query, t] for t in candidate_texts]
    scores = cross_encoder_model.predict(pairs)  # higher = more relevant
    ranked = sorted(list(zip(candidate_texts, scores)), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], scores


def mmr_select(cand_embeddings: np.ndarray, query_embedding: np.ndarray, candidates: List[int], top_n=MMR_TOP_K, lambda_param=MMR_LAMBDA):
    """
    cand_embeddings: (N, D) embeddings for candidate set (already normalized)
    query_embedding: (D,) normalized
    candidates: list of indices into cand_embeddings
    Returns list of selected candidate indices (subset of 'candidates').
    """
    selected = []
    if len(candidates) == 0:
        return selected
    sims_to_query = (cand_embeddings @ query_embedding.reshape(-1,1)).squeeze()  # similarity for ALL cand embeddings
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
            sim_to_selected = max((cand_embeddings[c] @ cand_embeddings[s]) for s in selected) if selected else 0.0
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

    need_rebuild = True
    if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
        try:
            index, metadatas, index_info = load_index_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)
            if is_index_compatible(index_info):
                print("[index] found compatible index on disk. Loading...")
                need_rebuild = False
            else:
                print("[index] index config mismatch — will rebuild index.")
                need_rebuild = True
        except Exception as e:
            print(f"[index] failed to load existing index: {e}. Will rebuild.")
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
        faiss_index = build_faiss_index(embeddings)
        index_info = {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "cross_encoder": CROSS_ENCODER_MODEL,
            "n_chunks": len(texts),
            "built_at": time.time()
        }
        save_index_and_metadata(faiss_index, metadatas, FAISS_INDEX_PATH, METADATA_PATH, index_info)
        index = faiss_index
    else:
        index, metadatas, index_info = load_index_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)
        embeddings = None
        # load embedding model for queries
        emb_model = SentenceTransformer(EMBEDDING_MODEL)

    # load cross-encoder reranker
    print(f"[rerank] loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print("\nReady for queries. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break

            # 1) FAISS retrieval (many candidates)
            idxs, scores = query_faiss(index, emb_model, q, top_k=FAISS_TOP_K)
            # filter negative indices if any
            candidates = [i for i in idxs if i >= 0]
            candidate_texts = [metadatas[i]["text"] for i in candidates]

            if not candidate_texts:
                print("No candidates retrieved.")
                continue

            # 2) Cross-encoder rerank on the FAISS candidates
            reranked, raw_scores = rerank_with_crossencoder(q, candidate_texts, cross_encoder, top_n=len(candidate_texts))
            reranked_texts = [t for t, s in reranked]  # sorted by score desc

            # If USE_MMR: perform MMR selection on the top-M reranked candidates using their embeddings
            final_selected_texts = []
            if USE_MMR:
                # prepare candidate embeddings:
                # if embeddings for the whole index are not loaded, we compute embeddings for the candidate_texts
                # to perform MMR we need normalized embeddings
                cand_embs = emb_model.encode(candidate_texts, convert_to_numpy=True)
                # normalize
                cand_embs = cand_embs.astype("float32")
                cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)

                # query embedding
                q_emb = emb_model.encode([q], convert_to_numpy=True)[0].astype("float32")
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

                # Map reranked order back to candidate indices (we reranked candidate_texts)
                # We'll run MMR across candidate indices in the reranked order (take top 20 candidates for MMR)
                top_for_mmr = min(len(candidate_texts), 20)
                indices_for_mmr = list(range(top_for_mmr))  # indices into candidate_texts
                # mmr selects indices in this reduced set -> map to texts
                selected_rel_indices = mmr_select(cand_embs, q_emb, indices_for_mmr, top_n=MMR_TOP_K, lambda_param=MMR_LAMBDA)
                final_selected_texts = [candidate_texts[i] for i in selected_rel_indices]
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
            context = "\n\n".join([f"[DOC {i+1}]\n{t}" for i, t in enumerate(final_selected_texts)])
            prompt = f"""You are an assistant for question-answering tasks.
Use ONLY the following documents to answer the question. If the answer is not contained in the documents, say "I don't know."
Be concise (max 3 sentences).

Question: {q}
Documents:
{context}

Answer:"""
            print("\nPrepared prompt (send to your LLM):\n")
            print(prompt)
            # Optionally: call Ollama or other LLM here. We leave it to you to enable.
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
