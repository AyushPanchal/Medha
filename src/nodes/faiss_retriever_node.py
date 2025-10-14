"""
FaissRetrieverNode
------------------
Retriever node for a LangGraph-based RAG system.
It loads a FAISS vector store and retrieves the most relevant documents
for a user's question. Works seamlessly with ChatbotState objects.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.states.chatbot_state import ChatbotState
import os


class FaissRetrieverNode:
    def __init__(
        self,
        index_path: str = "../../data/indexes/faiss",
        model_name: str = "text-embedding-3-large",
        k: int = 5,
        use_open_ai_embeddings: bool = True
    ):
        """
        Initialize FAISS retriever node.

        Args:
            index_path (str): Path to the FAISS index directory.
            model_name (str): Name of embedding model used for query encoding.
            k (int): Number of top documents to retrieve.
        """
        self.index_path = index_path
        self.model_name = model_name
        self.k = k

        if use_open_ai_embeddings:
            self.embeddings = OpenAIEmbeddings(model=self.model_name)
        else:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        # Load FAISS store
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ FAISS index not found at: {index_path}")

        self.vectorstore = FAISS.load_local(
            index_path, self.embeddings, allow_dangerous_deserialization=True
        )

        print(f"✅ FaissRetrieverNode initialized with model: {self.model_name}")

    # ---------------------------------------------------------------------- #
    #  🔹 Common Interface: process(state)
    # ---------------------------------------------------------------------- #
    def process(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve top-k relevant documents for the question in ChatbotState.

        Args:
            state (ChatbotState): The current chatbot state containing the question.

        Returns:
            ChatbotState: Updated state with retrieved context and documents.
        """
        if not state.question:
            raise ValueError("❌ Missing 'question' in ChatbotState.")

        # Perform similarity search
        docs = self.vectorstore.similarity_search(state.question, k=self.k)
        context = "\n\n".join([doc.page_content for doc in docs])

        print(f"🔍 Retrieved {len(docs)} documents for query: '{state.question[:60]}...'")
        return ChatbotState(context=context, source_docs=docs, question=state.question, messages=state.messages)


