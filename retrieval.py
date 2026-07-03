import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_store/faiss.index"
META_PATH = "vector_store/metadata.pkl"

print("Loading resources...")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadata = data["metadata"]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def retrieve(query, top_k=8):

    top_k = min(top_k, index.ntotal)

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    if len(indices[0]) == 0:
        return None, None

    best_score = float(distances[0][0])

    print(f"\nBest Score : {best_score:.4f}")

    if best_score < 0.25:
        return None, None

    context = []
    citations = []

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        meta = metadata[idx]
        chunk = documents[idx]

        print(f"\nScore : {score:.4f}")
        print(f"Source: {meta['source']} | Page: {meta['page']}")

        context.append(
            f"""
SOURCE: {meta['source']}
PAGE: {meta['page']}
CONTENT:
{chunk}
"""
        )

        citations.append(f"{meta['source']} (Page {meta['page']})")

    return "\n".join(context), list(dict.fromkeys(citations))


while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    context, citations = retrieve(query)

    if context is None:
        print("\nI do not have sufficient information in the provided documents.")
        continue

    print("\n================ CONTEXT ================\n")
    print(context)
    print("\n============== SOURCES ==================\n")

    for c in citations:
        print("-", c)