# backend/processor.py
import os, json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import fitz  # PyMuPDF
from prompts import SUMMARY_PROMPT, QA_PROMPT
from dotenv import load_dotenv
load_dotenv()

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_CHUNK_SIZE = 1000

INDEX_PATH = "./index/faiss_index"
DOC_STORE_PATH = "./index/docs.json"

embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

def extract_text_from_pdf(path: str) -> str:
    # Use PyMuPDF for robust extraction
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n\n".join(texts)

def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=EMBEDDING_CHUNK_SIZE, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    return docs

def process_pdf_and_build_index(path: str) -> Dict:
    text = extract_text_from_pdf(path)
    docs = chunk_text(text)

    # Build FAISS vectorstore (overwrite each upload for demo)
    os.makedirs("./index", exist_ok=True)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(INDEX_PATH)
    # persist docs
    with open(DOC_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump({"doc_count": len(docs)}, f)

    # compute a map-reduce summary
    chat = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=False)
    summary = chain.run(docs)

    # save summary
    with open("./index/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    return {"chunks": len(docs), "summary_len": len(summary)}

def load_vectorstore():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not found. Upload a PDF first.")
    vs = FAISS.load_local(INDEX_PATH, embeddings)
    return vs

def summarize_document() -> str:
    p = "./index/summary.txt"
    if not os.path.exists(p):
        return "No summary available. Upload & process a PDF first."
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def answer_query(query: str, top_k=5) -> Dict:
    vs = load_vectorstore()
    docs_and_scores = vs.similarity_search_with_score(query, k=top_k)
    retrieved_docs = [d[0] for d in docs_and_scores]

    # Build prompt with retrieved context
    chat = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.0)
    context_texts = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
    full_prompt = QA_PROMPT.format(context=context_texts, question=query)

    resp = chat.generate([{"role":"user", "content": full_prompt}])
    answer = resp.generations[0][0].text if resp.generations else "No answer"
    sources = [{"score": s, "snippet": d.page_content[:400]} for d, s in docs_and_scores]
    return {"answer": answer, "sources": sources}
