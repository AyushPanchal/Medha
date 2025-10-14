import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.prompts.prompt_loader import load_prompt
from src.states.chatbot_state import ChatbotState


class ChatbotNode:
    def __init__(self, llm):
        self.llm = llm

    def process(self, state: ChatbotState) -> ChatbotState:
        """Generate LLM response and update conversation history."""

        # --- Validation ---
        if not state.question:
            raise ValueError("❌ Missing 'question' in ChatbotState.")
        if not state.context:
            raise ValueError("❌ Missing 'context' in ChatbotState (did you run retriever first?).")

        # --- Include last answer in context for pronoun resolution ---
        enhanced_context = state.context
        if state.answer:
            enhanced_context += f"\n\nPrevious answer: {state.answer}"

        # --- Load and format system prompt ---
        system_prompt_template = load_prompt(r"../prompts/system/main/v1.md")

        # Format chat history for the placeholder
        formatted_history = ""
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                formatted_history += f"👤 Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                formatted_history += f"🤖 Medha: {msg.content}\n"
        if not formatted_history:
            formatted_history = "No previous conversation."

        # Inject context and chat history placeholders
        system_prompt = system_prompt_template.format(
            context=enhanced_context or "None",
            chat_history=formatted_history
        )

        # --- Construct messages ---
        system_msg = SystemMessage(content=system_prompt)
        user_msg = HumanMessage(content=state.question)

        # Combine system prompt, chat history, and new user query
        conversation = [system_msg] + state.messages + [user_msg]

        # --- Invoke the model ---
        ai_response = self.llm.invoke(conversation)
        clean_answer = re.sub(r"<think>.*?</think>", "", ai_response.content, flags=re.DOTALL).strip()

        # --- Update state ---
        new_messages = state.messages + [user_msg, AIMessage(content=clean_answer)]
        updated_state = ChatbotState(
            question=state.question,
            context=state.context,
            answer=clean_answer,
            source_docs=state.source_docs,
            messages=new_messages,
        )


        return updated_state
