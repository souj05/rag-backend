# RAG Document Q&A — Backend

A Retrieval-Augmented Generation (RAG) system that answers questions grounded
in uploaded PDF documents. Built with LangChain-style RAG logic, Pinecone
(vector database), and Groq (LLM), exposed through both a Streamlit UI and a
FastAPI REST API.

**Live demo (Streamlit Cloud):** https://rag-backend-fafjrsqtictxrmq5pehvr4.streamlit.app/

## What it does

1. Upload a PDF
2. The document is chunked into smaller passages
3. Each chunk is embedded (converted to a vector) using Pinecone's hosted
   embedding model and stored in a Pinecone vector index
4. When you ask a question, the question is embedded too, and Pinecone
   retrieves the most semantically similar chunks
5. Those chunks are passed as context to an LLM (Groq — Llama 3.1), which
   generates an answer grounded in the document rather than relying on
   general training knowledge alone

## Tech Stack

- **Vector Database:** Pinecone (serverless, cosine similarity)
- **Embeddings:** Pinecone hosted inference (`multilingual-e5-large`)
- **LLM:** Groq (`llama-3.1-8b-instant`)
- **Backend/API:** FastAPI
- **UI:** Streamlit
- **Frontend (separate repo):** React (Vite) — calls the FastAPI endpoints

## Project Structure

```
.
├── app.py             # Streamlit UI
├── main.py             # FastAPI REST API
├── rag_pipeline.py     # Core RAG logic (chunking, embeddings, retrieval, generation)
├── requirements.txt
├── Dockerfile
├── .env.example
└── sample_manual.pdf   # Sample document for testing
```

## Running locally

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Add your API keys
Copy `.env.example` to `.env` and fill in:
```
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
PINECONE_INDEX_NAME=rag-document-qa
```

### 3a. Run the Streamlit app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

### 3b. Or run the FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```

- Interactive API docs (Swagger UI): **http://localhost:8000/docs**
- Alternative API docs (ReDoc): **http://localhost:8000/redoc**

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF (multipart form, field `file`) to chunk, embed, and index it |
| POST | `/ask` | Send `{"question": "..."}` to get a grounded answer + source chunks |
| GET | `/health` | Health check |

## Deployment

- **Streamlit UI** → deployed on [Streamlit Community Cloud](https://share.streamlit.io), secrets configured via the app's Advanced Settings
- **FastAPI backend** → designed for containerized deployment (see `Dockerfile`); can be deployed to any cloud platform supporting Docker (AWS EC2, GCP Cloud Run, Azure Container Apps, etc.)

## Docker

```bash
docker build -t rag-backend .
docker run -p 8501:8501 --env-file .env rag-backend
```

## Notes

- `sample_manual.pdf` is a sample "Remote Work & Cloud Security Policy Manual"
  included for quick testing — try asking "What is the main point of this
  document?" or "Summarize the conclusion."
- Never commit real API keys in `.env` to a public repository — use
  platform-specific secrets management instead (Streamlit Secrets, GitHub
  Actions secrets, cloud provider secret managers, etc.)
