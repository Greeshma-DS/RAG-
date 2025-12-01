import os
import json
import time
import requests
import pandas as pd
from pypdf import PdfReader
import chromadb

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = "datadiabetics"     # folder containing your 4 PDFs
CHROMA_DIR = "chroma_db"       # vector database folder
COLLECTION_NAME = "diabetes_rag"

MODELS = [
    "llama3.1:8b",
    "mistral:7b-instruct",
    "phi3:medium"
]

QUESTIONS = [
    "What are the early symptoms of type 2 diabetes?",
    "What are the diagnostic criteria for diabetes according to the documents?",
    "How is HbA1c used to monitor blood glucose control?",
    "What are common complications of long-term uncontrolled diabetes?",
    "What lifestyle changes are recommended for type 2 diabetes management?",
    "Which medications are commonly used as first-line treatment for type 2 diabetes?",
    "What is diabetic neuropathy and how is it typically managed?",
    "How can patients reduce their risk of hypoglycemia?",
    "What dietary patterns are recommended for people with diabetes?",
    "How does insulin resistance contribute to the development of type 2 diabetes?"
]

OUTPUT_CSV = "rag_results_diabetes.csv"


# =====================================================
# 1. LOAD PDFS
# =====================================================

def load_pdfs(folder):
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    print(f"[PDF] Found {len(pdf_files)} PDF files.")

    docs = []
    for fname in pdf_files:
        path = os.path.join(folder, fname)
        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        full_text = "\n".join(pages)
        docs.append({"id": fname, "text": full_text})

    return docs


# =====================================================
# 2. TEXT CHUNKING
# =====================================================

def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap

    return chunks


# =====================================================
# 3. OLLAMA EMBEDDING
# =====================================================

def embed_text(text):
    resp = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": text}
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# =====================================================
# 4. CHROMA VECTOR DB
# =====================================================

def get_or_create_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        coll = client.get_collection(COLLECTION_NAME)
        print(f"[DB] Loaded existing collection '{COLLECTION_NAME}'.")
    except:
        coll = client.create_collection(COLLECTION_NAME)
        print(f"[DB] Created new collection '{COLLECTION_NAME}'.")
    return coll


def build_or_load_index():
    coll = get_or_create_collection()

    if coll.count() > 0:
        print(f"[DB] Index already contains {coll.count()} chunks. Skipping rebuild.")
        return coll

    print("[DB] Building vector index from PDFs...")
    docs = load_pdfs(DATA_DIR)

    ids, embeddings, documents = [], [], []

    for doc in docs:
        doc_id = doc["id"]
        chunks = chunk_text(doc["text"])

        print(f"[DB] {doc_id} → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            vec = embed_text(chunk)

            ids.append(chunk_id)
            embeddings.append(vec)
            documents.append(chunk)

    coll.add(ids=ids, embeddings=embeddings, documents=documents)

    print(f"[DB] Finished indexing. Total chunks: {coll.count()}")
    return coll


def retrieve_context(coll, query, k=4):
    q_vec = embed_text(query)
    res = coll.query(query_embeddings=[q_vec], n_results=k)
    ids = res["ids"][0]
    docs = res["documents"][0]
    return list(zip(ids, docs))


# =====================================================
# 5. STREAMING-SAFE OLLAMA GENERATION
# =====================================================

def call_ollama_generate(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}

    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        full_output = ""

        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    full_output += obj["response"]
            except:
                continue

    return full_output.strip()


# =====================================================
# 6. BUILD PROMPT
# =====================================================

def build_prompt(question, context_chunks):
    ctx = "\n\n---\n\n".join([f"[{cid}]\n{text}" for cid, text in context_chunks])

    prompt = f"""
You are a medical assistant specialized in diabetes.

Use ONLY the information in the CONTEXT below.
If the answer is not contained in the context, say:
"The provided documents do not contain enough information."

CONTEXT:
{ctx}

QUESTION:
{question}

Provide a clear, medically accurate answer.
"""
    return prompt.strip()


# =====================================================
# 7. RUN RAG FOR ONE QUESTION + ONE MODEL
# =====================================================

def run_rag(coll, question, model):
    start_r = time.time()
    retrieved = retrieve_context(coll, question, k=4)
    retrieval_time = round(time.time() - start_r, 3)

    prompt = build_prompt(question, retrieved)

    start_g = time.time()
    answer = call_ollama_generate(model, prompt)
    generation_time = round(time.time() - start_g, 3)

    return {
        "question": question,
        "model": model,
        "retrieved_ids": ";".join([cid for cid, _ in retrieved]),
        "answer": answer,
        "retrieval_time_sec": retrieval_time,
        "generation_time_sec": generation_time,
        "answer_length_tokens": len(answer.split()),
    }


# =====================================================
# 8. FULL EXPERIMENT
# =====================================================

def run_full_experiment():
    coll = build_or_load_index()
    results = []

    for q in QUESTIONS:
        print(f"\n[Q] {q}")
        for m in MODELS:
            print(f"   → Model: {m}")
            res = run_rag(coll, q, m)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[RESULTS] Saved results to {OUTPUT_CSV}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    run_full_experiment()
