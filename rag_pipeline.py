"""
RAG Pipeline: Document Q&A using Pinecone (hosted embeddings) + Groq (LLM)

This matches the exact stack proven to work in Colab:
- No OpenAI (avoids billing)
- No sentence-transformers/torch (avoids numpy/torch binary conflicts)
- Pinecone's hosted inference API handles embeddings server-side
- Groq (Llama 3.1) handles answer generation, free tier
"""

import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-document-qa")
EMBED_MODEL = "multilingual-e5-large"
EMBED_DIM = 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_pc = None
_index = None


def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def ensure_index_exists():
    """Create the Pinecone index if it doesn't already exist, return the index handle."""
    global _index
    pc = get_pinecone_client()
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    _index = pc.Index(INDEX_NAME)
    return _index


def load_and_chunk_pdf(file_path: str) -> List[str]:
    """Load a PDF and split it into overlapping text chunks. Returns list of chunk texts."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(pages)
    return [d.page_content for d in docs]


def embed_and_store(texts: List[str]) -> int:
    """Embed chunks via Pinecone's hosted model and upsert them. Returns number of chunks stored."""
    pc = get_pinecone_client()
    index = ensure_index_exists()

    # Clear old vectors so each new document upload starts fresh (demo simplicity)
    try:
        index.delete(delete_all=True)
    except Exception:
        pass  # index may already be empty

    embed_response = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )

    vectors = [
        {"id": str(i), "values": embed_response[i].values, "metadata": {"text": texts[i]}}
        for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)
    return len(vectors)


def retrieve(query: str, k: int = 4) -> List[str]:
    """Embed the query and retrieve the top-k most similar chunks."""
    pc = get_pinecone_client()
    index = ensure_index_exists()

    q_embed = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    results = index.query(vector=q_embed[0].values, top_k=k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate an answer grounded in the retrieved context using Groq's Llama model."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    response = llm.invoke(prompt)
    return response.content


def answer_question(query: str, k: int = 4) -> dict:
    """Full RAG call: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve(query, k=k)
    answer = generate_answer(query, chunks)
    return {"answer": answer, "sources": chunks}
