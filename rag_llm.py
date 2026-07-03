import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# LOAD MODELS + INDEX
INDEX_PATH = "vector_store/faiss.index"
META_PATH = "vector_store/metadata.pkl"

print("Loading resources...")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadata = data["metadata"]

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

llm = Llama(
    model_path="models/qwen2.5-3b-instruct-q4_0.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)

print("System Ready!\n")


# RETRIEVAL FUNCTION

def retrieve(query, top_k=8):

    query_emb = embed_model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, top_k)

    best_score = distances[0][0]

    # STRICT FILTER (ANTI-HALLUCINATION GATE)
    if best_score < 0.25:
        return None, None

    context = []
    citations = []

    for idx in indices[0]:
        if idx==-1:
            continue
        doc = documents[idx]
        meta = metadata[idx]

        context.append(f"""
SOURCE: {meta['source']}
PAGE: {meta['page']}
TEXT: {doc}
""")

        citations.append(f"{meta['source']} (Page {meta['page']})")

    return "\n".join(context), list(set(citations))



while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    context, citations = retrieve(query)

    if context is None:
        print("\nI do not have sufficient information in the provided documents.")
        continue


    prompt =f"""<|system|>
You are an HR departmental policy assistant.

Answer ONLY using the provided context.

Do not explain your reasoning.
Do not mention the context.
Do not mention the document.
Do not say "I looked at the context".
Do not mention page numbers.
Do not say "The answer is present in the context."
Only answer the user's question.

If the answer is missing, reply exactly:
I do not have sufficient information in the provided documents.
<|end|>

<|user|>
Context:
{context}

Question:
{query}
<|end|>

<|assistant|>
"""

    response = llm(
        prompt,
        max_tokens=150,
        temperature=0,
        stop=[
             "</s>",
    "<|user|>",
    "<|system|>",
    "<|end|>",
    "Question:",
    "Human:",
    "Assistant:"
    ]
        
    )

    answer = response["choices"][0]["text"].strip()
    bad_sentences = [
    "To find the answer",
    "I looked at",
    "The answer is present in the context",
    "The answer is clearly stated",
    "I do not have sufficient information in the provided documents."
]

    for sentence in bad_sentences:
        answer = answer.replace(sentence, "")
   
    if len(answer) < 10:
        answer = "I do not have sufficient information in the provided documents."

    # OUTPUT
    print("\n" + "="*60)
    print("ANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")
    for c in citations:
        print("-", c)

    print("="*60)