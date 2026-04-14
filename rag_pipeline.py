"""
rag_pipeline.py — Retrieval-Augmented Generation Pipeline
==========================================================
This module implements the RAG (Retrieval-Augmented Generation) pipeline:

  1. CHUNK   — Split code into overlapping token chunks
  2. EMBED   — Convert each chunk into a vector via EmbeddingModel
  3. INDEX   — Store vectors in a FAISS index (in-memory)
  4. RETRIEVE— Given a query, find the top-k most similar chunks
  5. FORMAT  — Format retrieved chunks as context for the LLM

Additionally, a small built-in knowledge base of programming tips and
documentation snippets is included to enrich RAG results.
"""

import re
import numpy as np
from typing import List, Tuple, Optional
from embeddings import EmbeddingModel


# ─────────────────────────────────────────────
# BUILT-IN KNOWLEDGE BASE
# A small corpus of programming best-practice snippets.
# In a real project you'd load these from files or a database.
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # Python
    ("python", "List comprehensions are more Pythonic and faster than for-loops for building lists. "
               "Example: squares = [x**2 for x in range(10)]"),
    ("python", "Use 'with' statements for file I/O to ensure files are properly closed, even on errors. "
               "Example: with open('file.txt') as f: data = f.read()"),
    ("python", "Generator functions use 'yield' to produce values lazily, saving memory for large datasets."),
    ("python", "f-strings (f'Hello {name}') are the fastest and most readable string formatting method."),
    ("python", "Use enumerate() instead of range(len()) when you need both index and value in a loop."),
    ("python", "The 'collections' module provides Counter, defaultdict, deque — use them before reinventing."),
    # JavaScript
    ("javascript", "Prefer 'const' for variables that won't be reassigned, 'let' for those that will. "
                   "Avoid 'var' due to function-scoped hoisting issues."),
    ("javascript", "Use arrow functions (=>) for concise callbacks. They also preserve the outer 'this'."),
    ("javascript", "Promise.all() runs async tasks in parallel; await them together for performance."),
    ("javascript", "Optional chaining (?.) and nullish coalescing (??) prevent undefined/null errors cleanly."),
    # Java
    ("java", "Use StringBuilder instead of string concatenation in loops to avoid O(n²) memory allocation."),
    ("java", "The enhanced for-each loop is cleaner and safer than index-based iteration over collections."),
    ("java", "Always override equals() and hashCode() together when using objects as Map keys or Set elements."),
    ("java", "Try-with-resources (try(Resource r = ...){}) automatically closes AutoCloseable objects."),
    # C++
    ("c++", "Prefer smart pointers (unique_ptr, shared_ptr) over raw pointers to avoid memory leaks."),
    ("c++", "Use range-based for loops (for(auto& x : vec)) for cleaner iteration over containers."),
    ("c++", "Pass large objects by const reference (&) to avoid expensive copies in function parameters."),
    ("c++", "Reserve vector capacity upfront with vec.reserve(n) when the size is known to avoid reallocations."),
    # General algorithms
    ("algorithms", "Binary search reduces O(n) linear search to O(log n) — only works on sorted data."),
    ("algorithms", "Hash maps provide O(1) average-case lookup. Use them to trade space for time."),
    ("algorithms", "Dynamic programming stores subproblem results to avoid redundant computation (memoisation)."),
    ("algorithms", "Divide and conquer splits problems recursively; merge sort and quicksort are classic examples."),
    # Debugging
    ("debugging", "Add print/log statements at function entry/exit to trace execution flow."),
    ("debugging", "Check boundary conditions: empty input, single element, maximum values."),
    ("debugging", "Use a debugger with breakpoints instead of excessive print statements for complex bugs."),
    ("debugging", "Rubber-duck debugging: explain your code line-by-line out loud — bugs often become obvious."),
    # Optimization
    ("optimization", "Profile before optimizing. Use cProfile (Python) or perf (Linux) to find real bottlenecks."),
    ("optimization", "Cache expensive function results with memoisation (@lru_cache in Python)."),
    ("optimization", "Prefer set/dict lookups (O(1)) over list searches (O(n)) for membership tests."),
    ("optimization", "Lazy evaluation — compute values only when needed — reduces wasted work."),
]


# ─────────────────────────────────────────────
# CHUNKING HELPERS
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text       : The source text (code) to chunk.
    chunk_size : Approximate number of words per chunk.
    overlap    : Number of words to overlap between consecutive chunks.

    Returns
    -------
    List of chunk strings.
    """
    words  = text.split()
    chunks = []
    step   = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_by_function(code: str, language: str) -> List[str]:
    """
    Attempt to split code at function/class boundaries for better chunking.
    Falls back to word-level chunking if no boundaries are found.

    Parameters
    ----------
    code     : Source code string.
    language : Programming language name.

    Returns
    -------
    List of code sections (each is a function or class block, roughly).
    """
    patterns = {
        "python":     r"(?=\n(?:def |class )\w)",
        "javascript": r"(?=\n(?:function |class |const \w+ ?= ?\(|let \w+ ?= ?\())",
        "java":       r"(?=\n\s*(?:public|private|protected|static)\s)",
        "c++":        r"(?=\n\w[\w\s*&]+\w\s*\()",
    }
    lang_lower = language.lower()
    pattern    = patterns.get(lang_lower)

    if pattern:
        sections = re.split(pattern, code)
        sections = [s.strip() for s in sections if s.strip()]
        if len(sections) > 1:
            return sections

    # Fallback: word-level chunks
    return chunk_text(code, chunk_size=200, overlap=20)


# ─────────────────────────────────────────────
# RAG PIPELINE CLASS
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline:
      - Embeds user code and knowledge base entries
      - Indexes them in a FAISS vector store
      - Retrieves top-k relevant chunks given a query
    """

    def __init__(self, api_key: str = ""):
        self.embedder   = EmbeddingModel(api_key=api_key, prefer_local=True)
        self.index      = None       # FAISS index (built lazily)
        self.chunks     : List[str] = []   # raw text of each indexed chunk
        self.chunk_meta : List[dict] = []  # metadata per chunk
        self._kb_indexed = False

        # Pre-index the knowledge base
        self._index_knowledge_base()

    # ── Knowledge Base ────────────────────────

    def _index_knowledge_base(self):
        """Embed and index the built-in knowledge base on startup."""
        kb_texts = [entry[1] for entry in KNOWLEDGE_BASE]
        kb_meta  = [{"source": "knowledge_base", "language": entry[0]}
                    for entry in KNOWLEDGE_BASE]
        self._add_to_index(kb_texts, kb_meta)
        self._kb_indexed = True

    # ── Indexing ─────────────────────────────

    def index_code(self, code: str, language: str = "Unknown", chunk_size: int = 256):
        """
        Chunk, embed, and index the given source code.

        Parameters
        ----------
        code       : Raw source code string.
        language   : Programming language (for smarter chunking).
        chunk_size : Approx words per chunk.
        """
        # Remove old user-code chunks (keep KB chunks)
        kb_count = sum(1 for m in self.chunk_meta if m.get("source") == "knowledge_base")
        self.chunks     = self.chunks[:kb_count]
        self.chunk_meta = self.chunk_meta[:kb_count]

        # Rebuild FAISS index from scratch with KB + new code
        if kb_count > 0:
            # Re-embed KB (already done; retrieve stored vectors if possible)
            # For simplicity, just re-index from stored text chunks
            kb_texts = self.chunks[:]
            kb_meta  = self.chunk_meta[:]
        else:
            kb_texts, kb_meta = [], []

        # Chunk the new code
        code_chunks = chunk_by_function(code, language)
        if chunk_size:
            # Further split large function chunks
            fine_chunks = []
            for c in code_chunks:
                if len(c.split()) > chunk_size:
                    fine_chunks.extend(chunk_text(c, chunk_size=chunk_size, overlap=24))
                else:
                    fine_chunks.append(c)
            code_chunks = fine_chunks

        code_meta = [{"source": "user_code", "language": language, "chunk_idx": i}
                     for i in range(len(code_chunks))]

        all_texts = kb_texts + code_chunks
        all_meta  = kb_meta  + code_meta

        # Rebuild index
        self.chunks     = []
        self.chunk_meta = []
        self.index      = None
        self._add_to_index(all_texts, all_meta)

    def _add_to_index(self, texts: List[str], meta: List[dict]):
        """Embed texts and add them to the FAISS index."""
        if not texts:
            return

        embeddings = self.embedder.encode(texts)  # (N, dim)

        try:
            import faiss
            dim = embeddings.shape[1]
            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        except ImportError:
            # FAISS not installed — use numpy cosine similarity as fallback
            if self.index is None:
                self.index = {"type": "numpy", "vectors": embeddings}
            else:
                self.index["vectors"] = np.vstack([self.index["vectors"], embeddings])

        self.chunks.extend(texts)
        self.chunk_meta.extend(meta)

    # ── Retrieval ─────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Find the top-k chunks most semantically similar to the query.

        Parameters
        ----------
        query : The search query (e.g. the user's question or code action).
        top_k : Number of chunks to retrieve.

        Returns
        -------
        A formatted string of retrieved context snippets.
        """
        if self.index is None or not self.chunks:
            return ""

        query_vec = self.embedder.encode_one(query).reshape(1, -1)
        top_k     = min(top_k, len(self.chunks))

        try:
            import faiss
            distances, indices = self.index.search(query_vec, top_k)
            indices = indices[0].tolist()
        except ImportError:
            # Numpy cosine similarity fallback
            vectors  = self.index["vectors"]              # (N, dim)
            # Normalise
            norms    = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
            norm_q   = np.linalg.norm(query_vec) + 1e-9
            sims     = (vectors / norms) @ (query_vec / norm_q).T  # (N,1)
            indices  = np.argsort(-sims[:, 0])[:top_k].tolist()

        # Format retrieved chunks
        parts = []
        seen  = set()
        for idx in indices:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if chunk in seen:
                continue
            seen.add(chunk)
            meta = self.chunk_meta[idx]
            src  = meta.get("source", "unknown")
            lang = meta.get("language", "")
            label = f"[{src} | {lang}]" if lang else f"[{src}]"
            parts.append(f"{label}\n{chunk}")

        return "\n\n---\n\n".join(parts) if parts else ""