"""
Embedding model utilities.
All embedding logic is invoked from app.py via functions defined here.
"""

from typing import List
import numpy as np

# SentenceTransformers wrapped in try/except to avoid crash on missing dependency
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from config.config import EMBEDDING_MODEL


_model_cache = None

def load_embedding_model():
    global _model_cache
    if _model_cache is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. Add to requirements.txt")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedding_model()
    return np.array(model.encode(texts, normalize_embeddings=True))
