"""
Configuration and API keys.

IMPORTANT:
- Do NOT hardcode real keys. On Streamlit Cloud, add them as Secrets.
- Locally, set environment variables before running:
    export OPENAI_API_KEY="..."
    export GROQ_API_KEY="..."
    export GEMINI_API_KEY="..."
    export SERPAPI_API_KEY="..."   # or TAVILY_API_KEY
"""

import os

# LLM providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Web search providers (choose one you have)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# App defaults
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")  # openai | groq | gemini
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")   # sensible default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector store
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
