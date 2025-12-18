import sys
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# --------------------------
# 1. Load PDF
# --------------------------
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --------------------------
# 2. Split Text
# --------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# --------------------------
# 3. Create Vector Store
# --------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    return vectordb

# --------------------------
# 4. Create LLM + QA Chain
# --------------------------
def create_qa_chain(vectordb):
    # Free small HuggingFace LLM (flan-t5-base)
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0, "max_length":512})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k":3}),
        chain_type="stuff"
    )
    return qa

# --------------------------
# Main Function
# --------------------------
def main(pdf_path):
    print("\n📄 Loading annual report...")
    text = load_pdf(pdf_path)

    print("✂️ Splitting text...")
    chunks = split_text(text)

    print("📦 Creating vector store...")
    vectordb = create_vector_store(chunks)

    print("🤖 Creating QA chain...")
    qa = create_qa_chain(vectordb)

    # Generate a quick summary
    print("\n📑 Summary of the Annual Report:\n")
    summary = qa.run("Summarize the key highlights of this annual report in simple terms.")
    print(summary)

    # Interactive Q&A
    print("\n💬 You can now ask questions about the report! Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting. Goodbye!")
            break
        answer = qa.run(query)
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 app.py <annual_report.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
        sys.exit(1)

    main(pdf_path)
