import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# --- Configuration ---
data_file = "./data/extracted_pdf_images_text_optimised.json"
collection_name = "pdf_image_collection"

# --- Initialize Qdrant & Encoder ---
print("Initializing Qdrant client and sentence encoder...")
client = QdrantClient(url="http://localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load the extracted data ---
print(f"Loading data from {data_file}...")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if not data:
    raise ValueError("❌ No data found in JSON file!")

# --- Extract texts & generate embeddings ---
texts = [item["text"] for item in data]
print(f"Generating embeddings for {len(texts)} text entries...")
embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# --- Create collection if not exists ---
if collection_name not in [c.name for c in client.get_collections().collections]:
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")

# --- Prepare and upload points ---
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={
            "path": data[i]["path"],
            "text": data[i]["text"]
        },
    )
    for i in range(len(data))
]

print(f"Uploading {len(points)} embeddings to Qdrant...")
client.upsert(collection_name=collection_name, points=points)

print("\n✅ Done! All image texts embedded and stored in Qdrant.")
