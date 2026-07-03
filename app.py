import streamlit as st
import os
import sys
import shutil
import subprocess
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# PAGE CONFIG
st.set_page_config(page_title="Departmental RAG System", layout="wide")

st.title("Departmental RAG System")

# FOLDERS
UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

INDEX_PATH = os.path.join(VECTOR_FOLDER, "faiss.index")
META_PATH = os.path.join(VECTOR_FOLDER, "metadata.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)


# SIDEBAR - UPLOAD SECTION
st.sidebar.title("Document Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded to staging area.")


# LOAD MODELS
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def load_llm():
    return Llama(
        model_path="models/qwen2.5-3b-instruct-q4_0.gguf",
        n_ctx=4096,
        n_threads=8,
        verbose=False
    )

embed_model = load_embed_model()
llm = load_llm()


# LOAD VECTOR STORE 
@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        data = pickle.load(f)

    return index, data["documents"], data["metadata"]

index, documents, metadata = load_vector_store()


# UPDATE KNOWLEDGE BASE
if st.sidebar.button("Update Knowledge Base"):

    with st.spinner("Updating vector database..."):

        staged_files = os.listdir(UPLOAD_FOLDER)

        if not staged_files:
            st.sidebar.warning("No files in staging area to add.")
        else:
            # Move files from uploads → data
            for file in staged_files:
                src = os.path.join(UPLOAD_FOLDER, file)
                dst = os.path.join(DATA_FOLDER, file)
                shutil.move(src, dst)

            # Run ingestion pipeline
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.sidebar.error(f"Ingestion failed:\n{result.stderr}")
            else:
                load_vector_store.clear()
                st.sidebar.success("Knowledge Base Updated Successfully!")
                st.rerun()


if index is None:
    st.warning("No knowledge base found yet. Upload PDFs and click 'Update Knowledge Base' to get started.")
    st.stop()


# RETRIEVAL FUNCTION
def retrieve(query, top_k=8):

    # Never retrieve more chunks than available
    top_k = min(top_k, index.ntotal)

    query_emb = embed_model.encode(
    [query],
    normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_emb, top_k)

    scores = distances[0]

    if len(scores) == 0:
        return None, None

    best_score = float(scores[0])


    # Reject irrelevant queries
    if best_score < 0.25:
        return None, None

    context = ""
    citations = []

    for score, idx in zip(scores, indices[0]):

        if idx == -1:
            continue

        # Ignore weak matches
        if score < max(0.50, best_score - 0.10):
            continue

        doc = documents[idx]
        meta = metadata[idx]

        context += f"""
    SOURCE: {meta['source']}
    PAGE: {meta['page']}
    CONTENT:
    {doc}

    """

        citations.append(f"{meta['source']} (Page {meta['page']})")

    return context, list(dict.fromkeys(citations))


# USER INPUT
query = st.text_input("Ask your question")

if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    context, citations = retrieve(query)
    # NO INFO CASE
    if context is None:
        st.error("I do not have sufficient information in the provided documents.")
        st.stop()

    
    prompt = f"""
You are a helpful departmental policy assistant.

Answer ONLY using the information provided in the context.

Rules:
- If the answer is present in the context, answer clearly.
- Do not use outside knowledge.
- Do not invent information.
- If the answer is NOT present, reply exactly:
I do not have sufficient information in the provided documents.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a departmental policy assistant. "
                    "Answer ONLY from the provided context. "
                    "If the answer is not present, reply exactly: "
                    "'I do not have sufficient information in the provided documents.' "
                    "Do not explain your reasoning. "
                    "Do not mention the context. "
                    "Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": f"""Context:

    {context}

    Question:
    {query}
    """,
            },
        ],
        temperature=0,
        max_tokens=100,
    )

    answer = response["choices"][0]["message"]["content"].strip()
    FALLBACK = "I do not have sufficient information in the provided documents."

    bad_patterns = [
        "Human:",
        "Assistant:",
        "SELECT AVG",
        "SELECT *",
    ]

    for pattern in bad_patterns:
        if pattern in answer:
            answer = FALLBACK
            break

    if len(answer) < 10:
        answer = FALLBACK

    # OUTPUT
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for c in sorted(citations):
        st.write("•", c)