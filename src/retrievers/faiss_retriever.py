# faiss_retriever_node.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


class FaissRetriever:
    """
    A reusable, extensible FAISS retriever class.
    Handles loading, retrieval, reranking, updates, and evaluation.
    """

    def __init__(self, index_path: str, model_name: str, k: int = 5, use_open_ai_embeddings=True):
        """
        Initialize the FAISS retriever.
        """
        self.index_path = index_path
        self.model_name = model_name
        self.k = k

        if use_open_ai_embeddings:
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            # Load the embedding model used during ingestion
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        # Load FAISS index from disk
        self.db = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Create retriever interface
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

        print(f"[INFO] FAISS retriever initialized with model '{self.model_name}'")

    # ------------------------------------------------------
    # 🔹 Basic Retrieval
    # ------------------------------------------------------
    def query(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a given user query.
        """
        results = self.retriever.invoke(question)
        formatted = [
            {"rank": i + 1, "content": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(results)
        ]
        return formatted

    def preview(self, question: str, max_chars: int = 200):
        """
        Print a human-readable preview of retrieved results.
        """
        results = self.query(question)
        print(f"\n[Query] {question}")
        for res in results:
            snippet = res["content"][:max_chars].replace("\n", " ")
            print(f"\n🔹 Result {res['rank']}: {snippet}...")
            print(f"   Metadata: {res['metadata']}")

    # ------------------------------------------------------
    # 🔹 Dynamic Index Updates
    # ------------------------------------------------------
    def update_index(self, new_docs: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Dynamically add new documents to the FAISS index.

        Args:
            new_docs (List[str]): List of new text chunks to add.
            metadatas (List[Dict[str, Any]]): Optional metadata for each chunk.
        """
        print(f"[INFO] Adding {len(new_docs)} new documents to FAISS index...")

        self.db.add_texts(texts=new_docs, metadatas=metadatas)
        self.db.save_local(self.index_path)
        print("[INFO] Index updated and saved successfully!")

    # ------------------------------------------------------
    # 🔹 Cross-Encoder Reranking
    # ------------------------------------------------------
    def rerank_results(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Use a cross-encoder model to re-rank retrieved passages for better accuracy.

        Args:
            question (str): Query to rerank for.
            top_k (int): Optional number of results to rerank.

        Returns:
            List[Dict[str, Any]]: Re-ranked retrieval results.
        """
        if top_k is None:
            top_k = self.k

        results = self.query(question)

        # Load cross-encoder model (MS MARCO fine-tuned)
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(question, r["content"]) for r in results]
        scores = reranker.predict(pairs)

        for i, s in enumerate(scores):
            results[i]["rerank_score"] = float(s)

        # Sort by rerank score (descending)
        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return ranked

    # ------------------------------------------------------
    # 🔹 Evaluation Metrics
    # ------------------------------------------------------
    def evaluate(self, queries: List[str], ground_truth: Dict[str, List[str]], k: int = None):
        """
        Evaluate retriever performance using manual ground truth labels.

        Args:
            queries (List[str]): Test queries.
            ground_truth (Dict[str, List[str]]): Mapping query -> list of relevant doc IDs or substrings.
            k (int): Number of retrieved docs to consider.

        Returns:
            pd.DataFrame: DataFrame with per-query precision, recall, MRR.
        """
        if k is None:
            k = self.k

        metrics = []

        for query in queries:
            results = self.query(query)
            retrieved_texts = [r["content"] for r in results[:k]]
            relevant_texts = ground_truth.get(query, [])

            # Simple matching (string containment)
            matches = [
                any(gt.lower() in doc.lower() for gt in relevant_texts)
                for doc in retrieved_texts
            ]

            precision = sum(matches) / len(retrieved_texts) if retrieved_texts else 0
            recall = (
                sum(matches) / len(relevant_texts) if relevant_texts else 0
            )

            # MRR calculation
            mrr = 0
            for i, match in enumerate(matches):
                if match:
                    mrr = 1 / (i + 1)
                    break

            metrics.append({"query": query, "precision": precision, "recall": recall, "mrr": mrr})

        df = pd.DataFrame(metrics)
        print("\n[Evaluation Summary]")
        print(df.describe().round(3))
        return df
