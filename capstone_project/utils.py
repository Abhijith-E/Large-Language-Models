# backend/utils.py
import os
import json
import math
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
from pathlib import Path

# Config
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"       # summarization
QA_MODEL = "deepset/roberta-base-squad2"          # extractive QA

INDEX_DIR = Path("./index")
UPLOAD_DIR = Path("./uploads")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load models (singletons)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device= -1)  # device=-1 uses CPU; set 0 for GPU
qa_pipeline = pipeline("question-answering", model=QA_MODEL, device= -1)

def extract_text_with_pages(pdf_path: str) -> List[Dict]:
    """
    Returns list of {"page": i, "text": page_text}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        t = page.get_text("text")
        t = t.strip()
        if t:
            pages.append({"page": i, "text": t})
    return pages

def chunk_text_with_metadata(pages: List[Dict], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
    """
    Chunk text keeping page metadata.
    Returns list of {"content": str, "page": page_num, "chunk_id": id}
    """
    chunks = []
    cid = 0
    for p in pages:
        text = p["text"]
        tokens = text.split()
        if len(tokens) <= chunk_size:
            chunks.append({"content": text, "page": p["page"], "chunk_id": cid})
            cid += 1
        else:
            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = " ".join(chunk_tokens)
                chunks.append({"content": chunk_text, "page": p["page"], "chunk_id": cid})
                cid += 1
                start = end - chunk_overlap
                if start < 0:
                    start = 0
    return chunks

def build_faiss_index(chunks: List[Dict], index_path: str = str(INDEX_DIR/"faiss.index"), meta_path: str = str(INDEX_DIR/"meta.json")):
    """
    Build FAISS index and save it along with metadata (list of chunks)
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    # create index
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
    # normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    # save meta (so we can map indices back to chunks)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    return {"index_path": index_path, "meta_path": meta_path, "n_chunks": len(chunks)}

def load_faiss_index(index_path: str = str(INDEX_DIR/"faiss.index"), meta_path: str = str(INDEX_DIR/"meta.json")):
    if not Path(index_path).exists() or not Path(meta_path).exists():
        raise FileNotFoundError("Index not found. Upload and process a PDF first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"]

def retrieve(query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Returns list of (chunk_dict, score)
    """
    index, chunks = load_faiss_index()
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)  # D: distances, I: indices
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append((chunk, float(dist)))
    return results

def generate_summary(chunks: List[Dict], max_length: int = 250, min_length: int = 80) -> str:
    """
    Create a doc-level summary. Approach: join top-n chunks (by length / significance) and summarize.
    For a small project, we'll concat the first N substantive chunks and ask the summarizer to condense.
    """
    # pick chunks to summarize: for speed, take the longest chunks or first N pages
    texts = [c["content"] for c in chunks]
    # join first 8 chunks (or fewer)
    n = min(8, len(texts))
    joined = "\n\n".join(texts[:n])
    # summarizer expects moderate-length input; if very long, we chunk the input to summarizer too
    if len(joined.split()) > 1000:
        # break into pieces of ~800 tokens for summarizer and join summaries
        pieces = []
        tokens = joined.split()
        start = 0
        piece_size = 800
        while start < len(tokens):
            piece = " ".join(tokens[start:start+piece_size])
            pieces.append(piece)
            start += piece_size
        partial_summaries = []
        for p in pieces:
            s = summarizer(p, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            partial_summaries.append(s)
        # then summarize concatenation
        final = summarizer(" ".join(partial_summaries), max_length= max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        return final
    else:
        s = summarizer(joined, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        return s

def answer_question(query: str, top_k: int = 5) -> Dict:
    """
    Retrieval + extractive QA.
    Returns: {"answer": str, "source_chunks": [ {page, snippet, score} , ... ] }
    """
    results = retrieve(query, top_k=top_k)
    context_text = "\n\n".join([r[0]["content"] for r in results])
    # run QA pipeline (extractive) with combined context; we can also run QA per chunk and pick best
    qa_input = {"question": query, "context": context_text}
    try:
        res = qa_pipeline(qa_input)
        answer = res.get("answer", "")
        score = float(res.get("score", 0.0))
    except Exception as e:
        answer = "Error in QA model: " + str(e)
        score = 0.0

    sources = []
    for chunk, sc in results:
        snippet = chunk["content"][:400].replace("\n", " ")
        sources.append({"page": chunk["page"], "snippet": snippet, "score": sc})

    return {"answer": answer, "confidence": score, "sources": sources}
