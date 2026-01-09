from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.models import models
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from pathlib import Path
import time

# =========================
# Configuration
# =========================

COLLECTION_NAME = "self_rag_collection"
DATA_FILE = Path("./data/F18-ABCD-000.txt")

QDRANT_URL = "http://qdrant_db:6333/"
OLLAMA_URL = "http://ollama_cont:11434"

EMBED_MODEL = "all-MiniLM-L6-v2"
MASTER_MODEL = "nemotron-3-nano:30b-cloud"
SLAVE_MODEL = "nemotron-3-nano:30b-cloud"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

templates = Jinja2Templates(directory="./temp")

# =========================
# App & Models
# =========================

app = FastAPI(title="Self-RAG Master–Slave API")

qdrant = QdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer(EMBED_MODEL)

master_llm = ChatOllama(
    model=MASTER_MODEL,
    temperature=0,
    base_url=OLLAMA_URL,
)

slave_llm = ChatOllama(
    model=SLAVE_MODEL,
    temperature=0,
    base_url=OLLAMA_URL,
)

# =========================
# Utilities
# =========================

def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def ensure_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )


def ingest_file():
    ensure_collection()

    info = qdrant.get_collection(COLLECTION_NAME)
    if info.points_count > 0:
        return

    text = DATA_FILE.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)

    points = []
    for i, chunk in enumerate(chunks):
        vec = encoder.encode(chunk).tolist()
        points.append(
            PointStruct(
                id=i,
                vector=vec,
                payload={
                    "text": chunk,
                    "source": DATA_FILE.name,
                },
            )
        )

    qdrant.upload_points(COLLECTION_NAME, points)


def retrieve(query: str, limit: int = TOP_K):
    qvec = encoder.encode(query).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=limit,
        with_payload=True,
    )

    results = []
    for h in hits:
        results.append({
            "id": h.id,
            "text": h.payload["text"],
            "source": h.payload["source"]
        })
    return results



# =========================
# Self-RAG (Master–Slave)
# =========================

def slave_extract(chunks, question: str) -> str:
    context = "\n\n".join(
        f"[Chunk {c['id']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""
You are a retrieval extraction assistant.

TASK:
- Extract ONLY relevant facts from the context
- Keep chunk IDs in brackets
- Do NOT answer the question
- Do NOT say "I don't know"

Context:
{context}

Question:
{question}

Extracted facts with citations:
"""
    return slave_llm.invoke(prompt).content


def master_answer(question: str) -> dict:
    chunks = retrieve(question)
    extracted = slave_extract(chunks, question)  # ✅ use it below

    prompt = f"""
You are the main assistant.

RULES:
- Answer ONLY from extracted facts
- Cite sources using [Chunk ID]
- If facts are insufficient, say "I don't know"

Extracted Facts:
{extracted}    # <-- now it's used

Question:
{question}

Final Answer with citations:
"""
    answer = master_llm.invoke(prompt).content

    return {
        "answer": answer,
        "citations": [c["id"] for c in chunks],
        "chunks": chunks,
    }

# =========================
# FastAPI Lifecycle
# =========================

@app.on_event("startup")
def startup():
    ingest_file()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "docrag.html",
        {"request": request}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    result = master_answer(message)
    return {
        "query": message,
        "response": result["answer"],
        "citations": result["citations"],
        "chunks": result["chunks"],  # ✅ expose chunks
    }



@app.post("/chat_stream")
async def chat_stream(message: str = Body(..., embed=True)):

    def generate():
        # Get full answer + citations
        result = master_answer(message)
        text = result["answer"]
        citations = result["citations"]

        # Build a header showing citations
        yield f"Citations used: {', '.join([f'[Chunk {c}]' for c in citations])}\n\n"

        # Stream the answer text in chunks
        chunk_size = 50
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
            time.sleep(0.03)

    return StreamingResponse(generate(), media_type="text/plain")




@app.get("/search")
async def search(query: str, limit: int = TOP_K):
    results = retrieve(query, limit)
    return {
        "query": query,
        "results": results,
    }
