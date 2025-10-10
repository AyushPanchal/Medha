"""
ChatbotState
------------
Defines a strongly-typed state object for LangGraph-based RAG workflows.

Every node in the graph receives and returns this same state instance,
ensuring consistent data flow between retriever, generator, reranker, etc.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ChatbotState(BaseModel):
    """
    Represents the state of the chatbot pipeline at any point in time.
    This object flows through all LangGraph nodes.

    Attributes:
        question (str): User's query or input.
        context (Optional[str]): Retrieved context from FAISS or other sources.
        answer (Optional[str]): Final or intermediate answer from the LLM.
        source_docs (Optional[List[Any]]): Raw retrieved documents (for debugging or citations).
        metadata (Optional[dict]): Optional info like model names, confidence, etc.
    """

    question: str = Field(..., description="User's input question.")
    context: Optional[str] = Field(default=None, description="Retrieved context text.")
    answer: Optional[str] = Field(default=None, description="Generated answer from LLM.")
    source_docs: Optional[List[Any]] = Field(default=None, description="Retrieved documents from retriever.")
    metadata: Optional[dict] = Field(default_factory=dict, description="Optional metadata for diagnostics.")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "ignore"

    def summary(self) -> str:
        """Return a human-readable summary of current state."""
        context_preview = (self.context[:200] + "...") if self.context else "None"
        return (
            f"🧩 ChatbotState Summary:\n"
            f" - Question: {self.question}\n"
            f" - Has Context: {self.context is not None}\n"
            f" - Context Preview: {context_preview}\n"
            f" - Has Answer: {self.answer is not None}\n"
        )
