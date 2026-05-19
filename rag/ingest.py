"""
Ingest pipeline: turns PDFs in data/documents/ into a searchable vector store.
Run once: python -m rag.ingest
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():
    """Load all PDFs from the documents directory."""
    print(f"Loading PDFs from {DOCUMENTS_DIR}...")
    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages from PDFs.")
    return docs


def split_documents(docs):
    """Split docs into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and persist them in Chroma."""
    print(f"Embedding chunks with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"  Vector store saved to {VECTORSTORE_DIR}")
    return vectorstore


def main():
    if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.glob("*.pdf")):
        print(f"ERROR: Add some travel PDFs to {DOCUMENTS_DIR} first.")
        return

    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("\nDone. The vector store is ready.")


if __name__ == "__main__":
    main()
