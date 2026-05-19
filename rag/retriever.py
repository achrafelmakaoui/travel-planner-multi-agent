"""
Retriever module: gives agents a tool to query the knowledge base.
This is the heart of the AGENTIC RAG: agents call this tool when they decide
they need external information, with their own queries.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, RETRIEVER_K

_vectorstore = None

def get_vectorstore():
    """Lazy-load the vector store (singleton pattern)."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    return _vectorstore


@tool
def search_travel_knowledge(query: str) -> str:
    """
    Search the travel knowledge base for information about destinations,
    attractions, accommodations, costs, food, culture, transport, etc.

    Use this whenever you need factual or contextual information that
    isn't in your general knowledge — for example, specific opening hours,
    average local prices, or details about a particular city's neighborhoods.

    Returns:
        A string containing the most relevant excerpts from the knowledge base.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=RETRIEVER_K)

    if not results:
        return "No relevant information found in the knowledge base."

    formatted = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )
    return formatted
