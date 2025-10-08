# src/ingestion/ingestion_pipeline.py
import os
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class IngestionPipeline:
    """
    Ingest a directory of Markdown files into a FAISS vector store.

    Defaults:
    - Loader: UnstructuredMarkdownLoader(mode="single", strategy="fast")
    - Splitter: RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    - Embeddings: sentence-transformers/all-mpnet-base-v2 (HuggingFaceEmbeddings)
    - Vector store: FAISS (save_local/load_local)

    Parameters:
        md_dir: Root directory containing .md files (searched recursively).
        persist_dir: Directory path where FAISS index will be saved/loaded.
        glob_pattern: Glob for markdown discovery; defaults to "**/*.md".
        chunk_size: Chunk size for text splitter; default 1000.
        chunk_overlap: Chunk overlap for text splitter; default 150.
        loader_mode: Unstructured mode; "single" or "elements"; default "single".
        loader_strategy: Unstructured strategy; often "fast"; default "fast".
        embedding_model_name: HuggingFace sentence-transformers model; default "sentence-transformers/all-mpnet-base-v2".
    """

    def __init__(
        self,
        md_dir: Union[str, Path],
        persist_dir: Union[str, Path],
        glob_pattern: str = "**/*.md",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        loader_mode: str = "single",
        loader_strategy: str = "fast",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        self.md_dir = Path(md_dir)
        self.persist_dir = Path(persist_dir)
        self.glob_pattern = glob_pattern
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader_mode = loader_mode
        self.loader_strategy = loader_strategy
        self.embedding_model_name = embedding_model_name

        # Set up the splitter once; reuse it throughout the pipeline.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Initialize the embedding model; this downloads/loads the HF model on first use.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vectordb: Optional[FAISS] = None

    def _discover_files(self) -> List[Path]:
        # Quick sanity check: fail early if the root doesn't exist.
        if not self.md_dir.exists():
            raise FileNotFoundError(f"Markdown root not found: {self.md_dir}")

        print(f"[1/6] Scanning for markdown files in: {self.md_dir} ...")
        paths = sorted(self.md_dir.glob(self.glob_pattern))
        print(f"      Found {len(paths)} file(s) matching '{self.glob_pattern}'.")
        return paths

    def _load_file(self, path: Path) -> List[Document]:
        # Load a single markdown file using the Unstructured loader.
        print(f"      Loading: {path.name}")
        loader = UnstructuredMarkdownLoader(
            str(path),
            mode=self.loader_mode,
            strategy=self.loader_strategy,
        )
        docs = loader.load()

        # Attach minimal, useful metadata for tracing chunks back to the source.
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.update({
                "source": str(path.resolve()),
                "filename": path.name,
                "relpath": str(path.relative_to(self.md_dir)),
            })

        return docs

    def _load_all(self, paths: List[Path]) -> List[Document]:
        # Iterate through discovered files and load each into Documents.
        print(f"[2/6] Loading documents with Unstructured (mode='{self.loader_mode}', strategy='{self.loader_strategy}') ...")
        all_docs: List[Document] = []
        for idx, p in enumerate(paths, start=1):
            print(f"      ({idx}/{len(paths)})", end=" ")
            docs = self._load_file(p)
            all_docs.extend(docs)
        print(f"      Loaded {len(all_docs)} document(s) (pre-splitting).")
        return all_docs

    def _split(self, docs: List[Document]) -> List[Document]:
        # Split into manageable chunks for better retrieval quality.
        print(f"[3/6] Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap}) ...")
        chunks = self.splitter.split_documents(docs)
        print(f"      Produced {len(chunks)} chunk(s) from {len(docs)} document(s).")
        return chunks

    def ingest(self) -> FAISS:
        """
        Discovers markdown files, loads, splits, embeds, and builds FAISS in memory.
        Returns the FAISS vector store instance.
        """
        # Discover files
        paths = self._discover_files()
        if not paths:
            raise FileNotFoundError(f"No markdown files found under: {self.md_dir}")

        # Load all documents
        raw_docs = self._load_all(paths)

        # Split into chunks
        chunks = self._split(raw_docs)

        # Build vector store from chunks with embeddings.
        print(f"[4/6] Building FAISS index with embeddings ('{self.embedding_model_name}') ...")
        self.vectordb = FAISS.from_documents(chunks, self.embeddings)
        print("      FAISS index built in memory.")
        print(f"      Total chunks indexed: {len(chunks)}")
        return self.vectordb

    def persist(self) -> None:
        """
        Saves the FAISS index to persist_dir using save_local.
        """
        if self.vectordb is None:
            raise RuntimeError("Vector store not built. Call ingest() first.")

        print(f"[5/6] Saving FAISS index to: {self.persist_dir} ...")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectordb.save_local(str(self.persist_dir))
        print("      Save complete.")

    def load(self) -> FAISS:
        """
        Loads an existing FAISS index from persist_dir using the same embeddings.
        """
        print(f"[6/6] Loading FAISS index from: {self.persist_dir} ...")
        self.vectordb = FAISS.load_local(
            str(self.persist_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,  # only if the files are trusted
        )
        print("      Load complete.")
        return self.vectordb

    def build_and_persist(self) -> FAISS:
        """
        Convenience: run ingest and then persist to disk.
        """
        self.ingest()
        self.persist()
        return self.vectordb
