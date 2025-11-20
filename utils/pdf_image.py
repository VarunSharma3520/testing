import os
import io
import json
import time
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# ============================
#  CONFIGURATION
# ============================
PDF_PATH = "./data/F18-ABCD-000.pdf"
IMAGE_OUTPUT_DIR = "./data/pdf_images"
JSON_OUTPUT = "./data/extracted_pdf_images_text.json"
TESSERACT_PATH = r"D:\tesrect\tesseract.exe"  # adjust this to your system
COLLECTION_NAME = "pdf_image_collection"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================
# 1. EXTRACT IMAGES FROM PDF
# ============================
def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    image_count = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    output_filename = os.path.join(output_folder, f"page{page_num+1}_{img_index+1}.{image_ext}")
                    image.save(output_filename)
                    image_count += 1
                except Exception as e:
                    print(f"❌ Could not save image {xref} on page {page_num+1}: {e}")

    doc.close()
    print(f"✅ Extracted {image_count} images to '{output_folder}'")
    return image_count


# ============================
# 2. OCR TEXT FROM IMAGES
# ============================
def extract_text_from_images(folder_path, output_json):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    data = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    text = pytesseract.image_to_string(img, lang="eng")
                data.append({"path": image_path, "text": text.strip()})
                print(f"✅ OCR extracted from: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved OCR data to {output_json}")

    return data


# ============================
# 3. LOAD DATA & STORE IN QDRANT
# ============================
def upload_to_qdrant(data, collection_name):
    print("\n🚀 Initializing Qdrant client and SentenceTransformer encoder...")
    client = QdrantClient(url=QDRANT_URL)
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # Create collection if needed
    if collection_name not in [c.name for c in client.get_collections().collections]:
        print(f"Creating Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    texts = [item["text"] for item in data]
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"path": data[i]["path"], "text": data[i]["text"]}
        )
        for i in range(len(data))
    ]

    print(f"Uploading {len(points)} OCR entries to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Upload complete!")
    return client, encoder


# ============================
# 4. SEMANTIC SEARCH INTERFACE
# ============================
def semantic_search_interface(client, encoder, collection_name):
    print("\n" + "=" * 60)
    print("🔍 Semantic Search for PDF Image Texts")
    print("=" * 60)
    print("Enter a natural-language query (e.g. 'invoice number', 'signature page')")
    print("Commands: ':quit' to exit, ':help' for tips")
    print("=" * 60)

    while True:
        query = input("\n🧠 Query> ").strip()
        if not query:
            continue
        if query.lower() in (":quit", "quit", "exit", ":exit"):
            print("👋 Exiting search.")
            break
        if query.lower() == ":help":
            print("💡 Example queries:")
            print("  'company address'  |  'total amount due'  |  'exam question'")
            continue

        # Encode and search
        print("Searching...")
        q_vec = encoder.encode(query, convert_to_numpy=True)
        hits = client.search(collection_name=collection_name, query_vector=q_vec.tolist(), limit=5, with_payload=True)

        if not hits:
            print("❌ No results found.")
            continue

        print(f"\n✨ Top {len(hits)} matches for: '{query}'")
        print("-" * 60)
        for i, h in enumerate(hits, start=1):
            score = h.score
            path = h.payload.get("path", "N/A")
            text = h.payload.get("text", "").replace("\n", " ")
            print(f"{i}. 📄 {os.path.basename(path)} | Similarity: {score:.3f}")
            print(f"   Snippet: {text[:180]}{'...' if len(text) > 180 else ''}")
        print("-" * 60)


# ============================
#  MAIN PIPELINE
# ============================
def process_pdf_pipeline():
    print("📘 Starting PDF → OCR → Qdrant pipeline...")

    # Step 1: Extract images
    extract_images_from_pdf(PDF_PATH, IMAGE_OUTPUT_DIR)

    # Step 2: OCR text
    data = extract_text_from_images(IMAGE_OUTPUT_DIR, JSON_OUTPUT)

    # Step 3: Upload to Qdrant
    client, encoder = upload_to_qdrant(data, COLLECTION_NAME)

    # Step 4: Interactive search
    semantic_search_interface(client, encoder, COLLECTION_NAME)


# ============================
#  RUN THE PIPELINE
# ============================
if __name__ == "__main__":
    process_pdf_pipeline()
