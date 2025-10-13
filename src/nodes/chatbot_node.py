"""
ChatbotNode
------------
Uses retrieved context + user question to generate an answer.
Supports multi-turn chat via LangChain message objects.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.states.chatbot_state import ChatbotState
import re


class ChatbotNode:
    def __init__(
            self,
            llm,
    ):
        self.llm = llm

    def process(self, state: ChatbotState) -> ChatbotState:
        """Generate LLM response and update conversation history."""
        if not state.question:
            raise ValueError("❌ Missing 'question' in ChatbotState.")
        if not state.context:
            raise ValueError("❌ Missing 'context' in ChatbotState (did you run retriever first?).")

        system_msg = SystemMessage(
            content=(
                "You are Medha, a helpful chatbot for the Computer Science Department at SVNIT, Surat. "
                "Use only the context provided to answer accurately. "
                "If unsure, politely mention that the context doesn't contain the information."
            )
        )

        # Current turn user message
        user_msg = HumanMessage(
            content=f"Context:\n{state.context}\n\nQuestion:\n{state.question}"
        )

        # Combine previous conversation + new turn
        conversation = [system_msg] + state.messages + [user_msg]

        # Generate response
        ai_response = self.llm.invoke(conversation)
        answer = re.sub(r"<think>.*?</think>", "", ai_response.content, flags=re.DOTALL).strip()
        # answer = ai_response.content.strip()

        # Update state
        new_messages = state.messages + [user_msg, AIMessage(content=answer)]

        chatbot_state = ChatbotState(
            question=state.question,
            context=state.context,
            answer=answer,
            source_docs=state.source_docs,
            messages=new_messages,
        )

        # print(chatbot_state.summary())

        return chatbot_state
