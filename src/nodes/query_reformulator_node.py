import re

from src.states.chatbot_state import ChatbotState


class QueryReformulatorNode:
    def __init__(self, llm):
        self.llm = llm  # a language model instance you control

    def process(self, state: ChatbotState) -> ChatbotState:
        # Compose a prompt for reformulation
        history = "\n".join(msg.content for msg in state.messages) if state.messages else ""
        prompt = (
            "Given the conversation history and the last user question, rewrite the question "
            "to be fully self-contained without pronouns.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User's last question: {state.question}\n\n"
            "Rewritten question:"
        )

        # Use LLM to rewrite question
        response = self.llm.invoke(prompt)

        reformulated_question = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        print(reformulated_question)
        # Return updated state with the reformulated question for downstream nodes
        return ChatbotState(
            question=reformulated_question,
            context=state.context,
            messages=state.messages,
            source_docs=state.source_docs if hasattr(state, "source_docs") else None
        )
