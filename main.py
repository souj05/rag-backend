"""
FastAPI backend for the RAG Document Q&A system (Pinecone + Groq stack).

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import load_and_chunk_pdf, embed_and_store, answer_question

load_dotenv()

app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's domain before production use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF document into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        texts = load_and_chunk_pdf(tmp_path)
        count = embed_and_store(texts)
    finally:
        os.unlink(tmp_path)

    return {"status": "indexed", "chunks": count}


@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    """Ask any question against the currently indexed document."""
    result = answer_question(payload.question, k=payload.k)
    return result


@app.get("/health")
async def health_check():
    return {"status": "ok"}
