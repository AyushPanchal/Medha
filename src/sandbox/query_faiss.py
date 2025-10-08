#!/usr/bin/env python3
"""
Query a local FAISS vector store built from Markdown chunks.

- Embeddings: sentence-transformers/all-mpnet-base-v2 (HuggingFaceEmbeddings)
- Vector store: FAISS.load_local(...)
- Query: similarity_search_with_score + pretty print
"""

from typing import List, Tuple
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Hardcoded config: adjust as needed
PERSIST_DIR = r"../../data/indexes/faiss"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5


def get_vectorstore():
    # Create the same embedding function used at index time
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Many LangChain versions require allow_dangerous_deserialization=True to load local FAISS (pickle-backed) stores
    if not os.path.isdir(PERSIST_DIR):
        raise FileNotFoundError(f"FAISS directory not found: {PERSIST_DIR}")

    vectordb = FAISS.load_local(
        PERSIST_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe if index was created locally and is trusted
    )
    return vectordb


def query_vectorstore(question: str, k: int = TOP_K) -> List[Tuple[str, float, dict]]:
    db = get_vectorstore()
    docs_and_scores = db.similarity_search_with_score(question, k=k)
    results = []
    for doc, score in docs_and_scores:
        results.append((doc.page_content, float(score), doc.metadata or {}))
    return results


def pretty_print(results: List[Tuple[str, float, dict]]):
    print(f"Top {len(results)} results:")
    for i, (content, score, meta) in enumerate(results, start=1):
        print("=" * 80)
        print(f"[{i}] Score (L2, lower is better): {score:.4f}")
        print(f"Source: {meta.get('source', 'N/A')}")
        print("-" * 80)
        snippet = content.strip().replace("\n", " ")
        print(snippet[:800] + ("..." if len(snippet) > 800 else ""))


if __name__ == "__main__":
    question = "Who is the hod?"
    results = query_vectorstore(question, k=TOP_K)
    pretty_print(results)
