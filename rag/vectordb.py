from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_pdfs(pdf_dir="rag/papers", db_dir="rag/vectorstore"):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_dir, filename))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
    )
    chunks = splitter.split_documents(docs)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_dir)
    print("Vectorstore saved to", db_dir)

if __name__ == "__main__":
    ingest_pdfs()
