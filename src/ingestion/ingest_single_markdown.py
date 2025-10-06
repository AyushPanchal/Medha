# src\ingestion\ingest_single_markdown.py
"""
Hardcoded ingestion of a single Markdown file into a FAISS vector store.

- Loader: UnstructuredMarkdownLoader
- Splitter: RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=150)
- Embeddings: sentence-transformers/all-mpnet-base-v2 (HuggingFaceEmbeddings)
- Vector store: FAISS
"""

import os
import pathlib
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Hardcoded paths
MD_PATH = r"../../data/processed/markdowns/computer-faculty.md"  # change if needed
PERSIST_DIR = r"../../data/indexes/faiss"  # change if needed


def main():
    if not os.path.isfile(MD_PATH):
        raise FileNotFoundError(f"Markdown file not found: {MD_PATH}")

    # 1) Load markdown as Documents
    loader = UnstructuredMarkdownLoader(MD_PATH, mode="single", strategy="fast")
    docs = loader.load()

    # Optional: attach source metadata
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = str(pathlib.Path(MD_PATH).resolve())

    # 2) Chunk with RecursiveCharacterTextSplitter (hardcoded size/overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # 3) Embeddings: sentence-transformers/all-mpnet-base-v2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 4) Build FAISS from chunks
    vectordb = FAISS.from_documents(chunks, embeddings)

    # 5) Persist FAISS index
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectordb.save_local(PERSIST_DIR)

    print(f"Saved FAISS index to: {PERSIST_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
