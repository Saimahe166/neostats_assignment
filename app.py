import streamlit as st
import os, traceback
from typing import List

from config.config import DEFAULT_PROVIDER, DEFAULT_MODEL, TOP_K, VECTOR_DIR
from models.llm import LLMClient
from models.embeddings import load_embedding_model
from utils.rag import ingest_documents, retrieve_context
from utils.web_search import web_search
from utils.text import build_system_prompt, build_user_prompt, render_search_hits

st.set_page_config(page_title="NeoStats ShipAssist Bot", page_icon="📦", layout="wide")

st.title("📦 NeoStats ShipAssist – Contextual RAG + Live Search Chatbot")
st.caption("Imagine, Build, Solve — A blueprint chatbot with RAG, web search, and response modes.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["openai", "groq", "gemini"], index=["openai","groq","gemini"].index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in ["openai","groq","gemini"] else 0)
    model = st.text_input("Model name", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    mode = st.radio("Response Mode", ["Concise", "Detailed"], index=0, help="Toggle concise vs detailed responses")

    st.markdown("---")
    st.subheader("RAG Ingestion")
    uploaded = st.file_uploader("Upload policy docs, manuals, PDFs, or text", type=["txt","md","pdf"])
    build_btn = st.button("Build/Update Vector Store")
    key_state = st.session_state.get("vector_key", "")

    if build_btn and uploaded is not None:
        try:
            import pdfminer.high_level
            raw_text = ""
            if uploaded.name.lower().endswith(".pdf"):
                raw_text = pdfminer.high_level.extract_text(uploaded)
            else:
                raw_text = uploaded.read().decode("utf-8", errors="ignore")
            vkey = ingest_documents([raw_text])
            st.session_state["vector_key"] = vkey
            st.success(f"Vector store built ✅ Key: {vkey}")
        except Exception as e:
            st.error(f"Failed to ingest: {e}")
            st.exception(e)

    if key_state:
        st.info(f"Active vector key: `{key_state}` stored under `{VECTOR_DIR}/`")

    st.markdown("---")
    st.subheader("Live Web Search")
    search_example = st.text_input("Try a quick search", value="latest delivery delays for BlueDart in Mumbai")
    if st.button("Search"):
        hits = web_search(search_example, num=5)
        st.write(render_search_hits(hits))

st.markdown("### Ask a question")
question = st.text_input("e.g., \"Track number STB100349244098109\" or \"What is the return policy for damaged items?\"")

col1, col2 = st.columns([1,1])
with col1:
    use_case = st.text_input("Use-case focus", value="Universal Shipment Tracking & Warranty Support")
with col2:
    top_k = st.slider("Top-K RAG chunks", 1, 10, TOP_K)

if st.button("Ask"):
    try:
        # Gather RAG context if available
        vkey = st.session_state.get("vector_key", "")
        rag_chunks: List[str] = retrieve_context(vkey, question, top_k=top_k) if vkey else []

        # If nothing retrieved or user asks for 'latest', do a web search too
        do_search = ("latest" in question.lower()) or ("track" in question.lower()) or (len(rag_chunks) == 0)
        web_hits = web_search(question, num=5) if do_search else []

        # Compose prompts
        system = build_system_prompt(use_case)
        user = build_user_prompt(question, rag_chunks, web_hits, mode)

        # Init LLM
        llm = LLMClient(provider=provider, model=model, temperature=temperature)
        answer = llm.chat(system, user)

        st.markdown("#### Answer")
        st.write(answer)

        # Show retrieved context and search hits for transparency
        with st.expander("RAG Chunks Used"):
            if rag_chunks:
                for i, c in enumerate(rag_chunks, 1):
                    st.markdown(f"**{i}.** {c}")
            else:
                st.write("No RAG context.")
        with st.expander("Web Results Used"):
            if web_hits:
                st.write(render_search_hits(web_hits))
            else:
                st.write("No web results.")
    except Exception as e:
        st.error("Something went wrong.")
        st.exception(e)

st.markdown("---")
st.caption("Tip: Upload your courier policy PDFs or warranty docs to enable RAG. Then ask shipment or warranty questions.")
