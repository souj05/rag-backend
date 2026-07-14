"""
Streamlit UI for the RAG Document Q&A system (Pinecone + Groq stack).

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import load_and_chunk_pdf, embed_and_store, answer_question

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")
st.title("📄 RAG Document Q&A")
st.caption(
    "Upload a PDF and ask any question — answers are grounded in the document "
    "using retrieval-augmented generation (Pinecone vector DB + Groq LLM)."
)

if "ready" not in st.session_state:
    st.session_state.ready = False
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Chunking document and building vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            texts = load_and_chunk_pdf(tmp_path)
            count = embed_and_store(texts)
            st.session_state.chunks_count = count
            st.session_state.ready = True
            os.unlink(tmp_path)

        st.success(f"Document processed into {count} chunks and indexed in Pinecone.")

if st.session_state.ready:
    st.divider()

    st.markdown("**Try one of these, or type your own:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What is the main point of this document?"):
            st.session_state.preset_question = "What is the main point of this document?"
    with col2:
        if st.button("Summarize the conclusion"):
            st.session_state.preset_question = "Summarize the conclusion of this document."

    default_q = st.session_state.get("preset_question", "")
    question = st.text_input("Ask any question about the document:", value=default_q)

    if question:
        with st.spinner("Retrieving relevant context and generating answer..."):
            result = answer_question(question)

        st.markdown("### Answer")
        st.write(result["answer"])

        with st.expander("Retrieved source chunks"):
            for i, src in enumerate(result["sources"], start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(src[:300] + "...")
else:
    st.info("Upload and process a PDF above to get started. Then ask as many different questions as you like.")
st.markdown("---")
st.markdown("this app is built by sowjanya 💕 using Streamlit")