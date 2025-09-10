"""
RAG utilities: chunking, vector store (FAISS), and retrieval.
"""

import os, re, io, pickle, hashlib, traceback
from typing import List, Tuple
import numpy as np

from config.config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from models.embeddings import embed_texts

# Vector store implemented with FAISS if available, else cosine search fallback
try:
    import faiss
except Exception:
    faiss = None


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = _clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _hash_docs(docs: List[str]) -> str:
    m = hashlib.sha256()
    for d in docs:
        m.update(d.encode("utf-8", errors="ignore"))
    return m.hexdigest()[:12]


class SimpleVectorStore:
    """
    Uses FAISS if available; else falls back to a NumPy cosine search.
    Persists to disk under VECTOR_DIR with a key.
    """
    def __init__(self, key: str):
        self.key = key
        self.dir = os.path.join(VECTOR_DIR, key)
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, "index.faiss")
        self.meta_path = os.path.join(self.dir, "meta.pkl")
        self.embeddings = None
        self.texts = []
        self.index = None

        self._load()

    def _load(self):
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.texts = meta.get("texts", [])
            if faiss and os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            elif os.path.exists(os.path.join(self.dir, "embeddings.npy")):
                self.embeddings = np.load(os.path.join(self.dir, "embeddings.npy"))
        except Exception:
            # start clean if corrupted
            self.texts, self.index, self.embeddings = [], None, None

    def _save(self):
        with open(self.meta_path, "wb") as f:
            pickle.dump({"texts": self.texts}, f)
        if faiss and self.index is not None:
            faiss.write_index(self.index, self.index_path)
        elif self.embeddings is not None:
            np.save(os.path.join(self.dir, "embeddings.npy"), self.embeddings)

    def build(self, docs: List[str]):
        self.texts = docs
        embs = embed_texts(docs).astype("float32")
        if faiss:
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
            self.embeddings = None
        else:
            self.embeddings = embs
            self.index = None
        self._save()

    def add(self, docs: List[str]):
        if not docs:
            return
        embs = embed_texts(docs).astype("float32")
        if faiss:
            if self.index is None:
                self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
        else:
            if self.embeddings is None:
                self.embeddings = embs
            else:
                self.embeddings = np.vstack([self.embeddings, embs])
        self.texts.extend(docs)
        self._save()

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        if not self.texts:
            return []
        q_emb = embed_texts([query]).astype("float32")
        if faiss and self.index is not None:
            D, I = self.index.search(q_emb, top_k)
            idxs = I[0].tolist()
            scores = D[0].tolist()
        else:
            # cosine similarity
            A = self.embeddings
            q = q_emb[0]
            sims = (A @ q) / (np.linalg.norm(A, axis=1) * np.linalg.norm(q) + 1e-8)
            idxs = np.argsort(sims)[::-1][:top_k].tolist()
            scores = [float(sims[i]) for i in idxs]

        return [(self.texts[i], float(scores[n])) for n, i in enumerate(idxs)]


def ingest_documents(docs: List[str]) -> str:
    """
    Ingest raw docs (already-decoded strings). Splits into chunks and stores.
    Returns a vector store key that the app can use for retrieval.
    """
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d))
    key = _hash_docs(chunks)
    store = SimpleVectorStore(key)
    store.build(chunks)
    return key


def retrieve_context(key: str, query: str, top_k: int = TOP_K) -> List[str]:
    store = SimpleVectorStore(key)
    hits = store.search(query, top_k=top_k)
    return [t for t, _ in hits]
