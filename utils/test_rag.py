# testrag_faiss_ollama.py
import warnings
from pathlib import Path
import json
import numpy as np
import faiss
import requests
import sys

# loaders & splitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# embedding model (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    )
    
# silence the harmless pypdf "page label" warnings
warnings.filterwarnings("ignore", message="Could not reliably determine page label")



# ---------- Config ----------
PDF_PATH = Path("./data/F18-ABCD-000.pdf")
FAISS_INDEX_PATH = Path("./data/faiss_index.bin")
METADATA_PATH = Path("./data/faiss_metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
TOP_K = 4

# Ollama config
OLLAMA_URL = "http://localhost:11434"
# OLLAMA_MODEL = "gemma3:1b"  # Note: gemma3 doesn't exist yet, use gemma2
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_TIMEOUT = 60
# ---------------------------


def check_ollama_connection():
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"⚠️  Ollama is not responding properly at {OLLAMA_URL}")
            return False

        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        print(f"✅ Ollama connected. Available models: {', '.join(model_names)}")

        # Check if our model exists
        if OLLAMA_MODEL not in model_names:
            print(f"⚠️  Model '{OLLAMA_MODEL}' not found.")
            print(f"   Run: ollama pull {OLLAMA_MODEL}")
            return False

        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {OLLAMA_URL}")
        print("   Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def query_ollama(prompt: str, model: str = OLLAMA_MODEL, max_tokens: int = 300) -> str:
    """
    Query Ollama with proper streaming handling.
    """
    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
            "top_k": 20,
            "top_p": 0.9,
            "stop": ["</s>", "\n\nQuestion:", "Question:"],
        },
    }

    try:
        print(f"🤖 Querying {model}...", end=" ", flush=True)

        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT, stream=True)

        if response.status_code == 404:
            return f"Error: Model '{model}' not found. Run: ollama pull {model}"

        response.raise_for_status()

        # Collect streamed response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        full_response += chunk["response"]
                        print(".", end="", flush=True)  # Progress indicator

                    if chunk.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue

        print(" ✓")
        return full_response.strip() or "No response generated."

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {OLLAMA_TIMEOUT}s"
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


def try_loaders(path: Path):
    """Try multiple loaders in order and return a list of langchain-like Documents."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # try PDFMinerLoader
    try:
        loader = PDFMinerLoader(str(path))
        docs = loader.load()
        print(f"[loader] PDFMinerLoader -> {len(docs)} pages/chunks")
        return docs
    except Exception as e:
        print(f"[loader] PDFMinerLoader failed: {e}")

    # try pdfplumber if installed
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


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[split] produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")
    return chunks


def build_embeddings(texts, model_name=EMBEDDING_MODEL_NAME, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray):
    """
    We will use cosine similarity via inner-product on L2-normalized vectors.
    IndexFlatIP works well for smaller datasets (no compression).
    """
    # ensure float32
    vecs = embeddings.astype("float32")
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(
        dim
    )  # inner product on normalized vectors = cosine similarity
    index.add(vecs)
    return index


def save_index_and_metadata(
    index: faiss.Index, metadatas, index_path: Path, meta_path: Path
):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print(f"[save] index -> {index_path} (n={index.ntotal}), metadata -> {meta_path}")


def load_index_and_metadata(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found on disk.")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas


def prepare_texts_and_meta(chunks):
    texts = []
    metadatas = []
    for i, c in enumerate(chunks):
        meta = c.metadata or {}
        page_info = meta.get("page", meta.get("page_number", i + 1))
        header = f"[page:{page_info}] "
        text = header + (c.page_content or "").replace("\n", " ").strip()
        texts.append(text)
        metadatas.append({"id": i, "page": page_info, "text_preview": text[:300]})
    return texts, metadatas


def query_index(
    index: faiss.Index, metadatas, query_embedding: np.ndarray, top_k=TOP_K
):
    # normalize query
    q = query_embedding.astype("float32")
    q = q / (np.linalg.norm(q) + 1e-12)
    D, I = index.search(q.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        md = metadatas[idx]
        results.append({"score": float(score), "metadata": md})
    return results


def main():
    print("=" * 60)
    print("FAISS + Ollama Gemma RAG System")
    print("=" * 60)

    # Check if index exists
    index_exists = FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()

    if index_exists:
        print(f"\n✅ Found existing FAISS index at {FAISS_INDEX_PATH}")
        print("   Skipping re-indexing. Loading from disk...")
        index, metadatas = load_index_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)
        print(f"   Loaded index with {index.ntotal} vectors.")
    else:
        print(f"\n📄 No existing index found. Building new index...")
        docs = try_loaders(PDF_PATH)
        chunks = split_documents(docs)
        texts, metadatas = prepare_texts_and_meta(chunks)
        if len(texts) == 0:
            raise RuntimeError("No text extracted from PDF.")
        print(
            f"[prep] embedding {len(texts)} chunks with model '{EMBEDDING_MODEL_NAME}'..."
        )
        embeddings = build_embeddings(texts)
        index = build_faiss_index(embeddings)
        save_index_and_metadata(index, metadatas, FAISS_INDEX_PATH, METADATA_PATH)

    # Check Ollama connection
    print("\n🔍 Checking Ollama connection...")
    if not check_ollama_connection():
        print("\n⚠️  Ollama is not available. You can still search documents,")
        print("   but LLM responses won't be generated.")
        print("\nTo enable LLM:")
        print("  1. Install Ollama: https://ollama.com")
        print("  2. Run: ollama serve")
        print(f"  3. Run: ollama pull {OLLAMA_MODEL}")
        ollama_available = False
    else:
        ollama_available = True

    # interactive query loop
    print("\n" + "=" * 60)
    print("Ready for queries. Type a question (or 'exit' to quit)")
    print("=" * 60)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)  # used for query embeddings

    while True:
        try:
            q = input("\n💬 Question> ").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                break

            # Get embeddings for the query
            q_emb = model.encode([q], convert_to_numpy=True)[0]

            # Search FAISS index
            results = query_index(index, metadatas, q_emb, top_k=TOP_K)

            print(f"\n📚 Top {len(results)} relevant chunks:")
            print("-" * 60)
            for i, r in enumerate(results, 1):
                print(f"{i}. Score: {r['score']:.4f} | Page: {r['metadata']['page']}")
                preview = r["metadata"]["text_preview"][:200]
                print(f"   {preview}...")
                print()

            # Build context from retrieved documents
            context_chunks = []
            for r in results:
                page = r["metadata"]["page"]
                text = r["metadata"]["text_preview"]
                context_chunks.append(f"[Page {page}] {text}")

            context = "\n\n".join(context_chunks)
            print(f"Context for LLM: {context}\n for question: {q}\n")
            # Generate answer with Ollama if available
            if ollama_available:
                print("🤖 Generating answer with Ollama...")
                print("-" * 60)

                prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question." but for normal chitchat questions, feel free to answer based on your knowledge. Be concise and accurate.
Context:
{context}
Question: {q}
Answer:"""

                answer = query_ollama(prompt, model=OLLAMA_MODEL, max_tokens=300)
                print(f"\n💡 Answer:\n{answer}\n")
            else:
                print("\n⚠️  Ollama not available. Showing context only.")
                print("   Start Ollama to get AI-generated answers.\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
