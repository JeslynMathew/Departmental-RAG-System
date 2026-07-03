import os
import pickle
import hashlib
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

INDEX_PATH = os.path.join(VECTOR_FOLDER, "faiss.index")
META_PATH = os.path.join(VECTOR_FOLDER, "metadata.pkl")
HASH_PATH = os.path.join(VECTOR_FOLDER, "doc_hashes.pkl")

os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)


# LOAD EXISTING HASHES
if os.path.exists(HASH_PATH):
    with open(HASH_PATH, "rb") as f:
        indexed_hashes = pickle.load(f)
else:
    indexed_hashes = set()

# HELPER: FILE HASH
def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# MODEL + CHUNKER
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# STORAGE
documents = []
metadata = []

print("Reading PDFs...\n")


# READ FILES
for file in os.listdir(DATA_FOLDER):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(DATA_FOLDER, file)

    file_hash = get_file_hash(path)

    # SKIP IF ALREADY INDEXED
    if file_hash in indexed_hashes:
        print(f"Skipping duplicate: {file}")
        continue

    print(f"Processing: {file}")

    try:
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if not text:
                continue

            chunks = splitter.split_text(text)

            for chunk in chunks:

                documents.append(chunk)

                metadata.append({
                    "source": file,
                    "page": page_num + 1
                })

        indexed_hashes.add(file_hash)

    except Exception as e:
        print(f"Error in {file}: {e}")


if len(documents) == 0:
    print("\nNo new documents to index. Exiting safely.")
    exit()

print("\nTotal new chunks:", len(documents))


# EMBEDDINGS
print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")


# FAISS (COSINE SIMILARITY)
dimension = embeddings.shape[1]

new_index = faiss.IndexFlatIP(dimension)
new_index.add(embeddings)


# LOAD EXISTING INDEX 
if os.path.exists(INDEX_PATH):

    old_index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        old_data = pickle.load(f)

    # Merge NEW vectors into OLD index so vector order stays [old, new]
    old_index.merge_from(new_index)
    final_index = old_index

    # Keep documents/metadata in the SAME [old, new] order as the vectors
    documents = old_data["documents"] + documents
    metadata = old_data["metadata"] + metadata

else:
    final_index = new_index


#SAVING INDEX
faiss.write_index(final_index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump({
        "documents": documents,
        "metadata": metadata
    }, f)

with open(HASH_PATH, "wb") as f:
    pickle.dump(indexed_hashes, f)

print("\nIndex Created Successfully!")
print("Total vectors in index:", final_index.ntotal)
print("Total documents stored:", len(documents))