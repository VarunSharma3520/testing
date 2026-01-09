from qdrant_client.models import models, Distance, VectorParams
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
# import numpy as np
import time

# --- your data ---
documents = [
    {
        "name": "The Time Machine",
        "description": "A man travels through time and witnesses the evolution of humanity.",
        "author": "H.G. Wells",
        "year": 1895,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL24331810M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=vMJbKHUO93Q",
    },
    {
        "name": "Ender's Game",
        "description": "A young boy is trained to become a military leader in a war against an alien race.",
        "author": "Orson Scott Card",
        "year": 1985,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL24325769M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=2SRizeR4MmU",
    },
    {
        "name": "Brave New World",
        "description": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
        "author": "Aldous Huxley",
        "year": 1932,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL3301586M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=As2sMgm0Szo",
    },
    {
        "name": "The Hitchhiker's Guide to the Galaxy",
        "description": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
        "author": "Douglas Adams",
        "year": 1979,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL27284596M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=eLdiWe_HJv4",
    },
    {
        "name": "Dune",
        "description": "A desert planet is the site of political intrigue and power struggles.",
        "author": "Frank Herbert",
        "year": 1965,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL17952222M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=n9xhJrPXop4",
    },
    {
        "name": "Foundation",
        "description": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
        "author": "Isaac Asimov",
        "year": 1951,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL26774598M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=X4QYV5GTz7c",
    },
    {
        "name": "Snow Crash",
        "description": "A futuristic world where the internet has evolved into a virtual reality metaverse.",
        "author": "Neal Stephenson",
        "year": 1992,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL23230597M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=nO64ZhBYy9E",
    },
    {
        "name": "Neuromancer",
        "description": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
        "author": "William Gibson",
        "year": 1984,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL1627167M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=HJBnlZKgeUg",
    },
    {
        "name": "The War of the Worlds",
        "description": "A Martian invasion of Earth throws humanity into chaos.",
        "author": "H.G. Wells",
        "year": 1898,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL3946486M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=r-yas0yPbLU",
    },
    {
        "name": "The Hunger Games",
        "description": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
        "author": "Suzanne Collins",
        "year": 2008,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL22549594M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=mfmrPu43DF8",
    },
    {
        "name": "The Andromeda Strain",
        "description": "A deadly virus from outer space threatens to wipe out humanity.",
        "author": "Michael Crichton",
        "year": 1969,
        "image_url": "https://upload.wikimedia.org/wikipedia/en/b/bf/Big-andromedastrain.jpg",
        "video_url": "https://www.youtube.com/watch?v=YMbSpnlOOtE",
    },
    {
        "name": "The Left Hand of Darkness",
        "description": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
        "author": "Ursula K. Le Guin",
        "year": 1969,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL7524131M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=jtdJghgfGX8",
    },
    {
        "name": "The Three-Body Problem",
        "description": "Humans encounter an alien civilization that lives in a dying system.",
        "author": "Liu Cixin",
        "year": 2008,
        "image_url": "https://covers.openlibrary.org/b/OLID/OL25840917M-L.jpg",
        "video_url": "https://www.youtube.com/watch?v=SdvzhCL7vIA",
    },
]

collection_name = "test_collection"

# --- Qdrant & encoder setup ---
print("Initializing Qdrant client and sentence encoder...")
# client = QdrantClient(url="http://localhost:6333")
client = QdrantClient(url="http://qdrant:6333/")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Create or verify collection
if not client.collection_exists(collection_name=collection_name):
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )
    print("Collection created successfully.")
else:
    print(f"Collection '{collection_name}' already exists.")

# Check if collection is empty or needs upload
collection_info = client.get_collection(collection_name=collection_name)
points_count = collection_info.points_count

if points_count == 0:
    print(f"Uploading {len(documents)} documents to Qdrant...")
    points = []
    for idx, doc in enumerate(documents):
        vec = encoder.encode(doc["description"], convert_to_numpy=True)
        points.append(
            models.PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload=doc,
            )
        )

    client.upload_points(collection_name=collection_name, points=points)
    print(f"Upload completed. {len(documents)} documents indexed.")
    time.sleep(0.5)  # Allow time for indexing
else:
    print(f"Collection already contains {points_count} points. Skipping upload.")

# --- interactive natural-language query loop ---
print("\n" + "=" * 60)
print("🔍 Semantic Search Interface")
print("=" * 60)
print("Enter a natural-language query to search for books.")
print("Commands: ':quit' or ':exit' to exit, ':help' for tips")
print("=" * 60)

try:
    while True:
        user_q = input("\n📖 Query> ").strip()

        if not user_q:
            continue

        if user_q.lower() in (":quit", ":exit", "quit", "exit"):
            print("Exiting. Goodbye! 👋")
            break

        if user_q.lower() == ":help":
            print("\n💡 Search Tips:")
            print("  - Try descriptions: 'alien invasion', 'time travel story'")
            print("  - Or themes: 'dystopian future', 'virtual reality'")
            print("  - Or elements: 'hackers and computers', 'war and strategy'")
            print("  - More results: add 'show 5' to get 5 results")
            continue

        # Check if user wants more results
        limit = 3
        if "show " in user_q.lower():
            parts = user_q.lower().split("show ")
            if len(parts) > 1:
                try:
                    limit = int(parts[1].split()[0])
                    user_q = user_q.lower().replace(f"show {limit}", "").strip()
                except (ValueError, IndexError):
                    pass

        # Encode the query
        print("Searching...")
        q_vec = encoder.encode(user_q, convert_to_numpy=True)

        # Perform search
        hits = client.search(
            collection_name=collection_name,
            query_vector=q_vec.tolist(),
            limit=limit,
            with_payload=True,
        )
        print(type(hits[0]))
        if not hits:
            print("❌ No results found. Try a different query.")
            continue

        print(f"\n✨ Top {len(hits)} results for: '{user_q}'")
        print("-" * 60)

        for rank, h in enumerate(hits, start=1):
            score = h.score if hasattr(h, "score") else h.get("score", 0)
            payload = h.payload if hasattr(h, "payload") else h.get("payload", {})
            
            # Extract book details
            title = payload.get("name", "Unknown")
            author = payload.get("author", "Unknown")
            year = payload.get("year", "N/A")
            description = payload.get("description", "No description available")
            image_url = payload.get("image_url", "No image available")
            video_url = payload.get("video_url", "No video available")
            # Display result with formatting
            print(f"\n{rank}. {title}")
            print(f"   Author: {author} ({year})")
            print(f"   Similarity: {score:.3f}")
            print(f"   Description: {description}")
            print(f"   Image: {image_url}")
            # print(f"   Video: {video_url}")
            # print(f"   Page No: {h.id}")
            
        print("-" * 60)

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Exiting. 👋")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback

    traceback.print_exc()
