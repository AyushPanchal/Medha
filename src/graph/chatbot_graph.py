from langgraph.graph import StateGraph, END

from src.llms.groq_llm import GroqLLM
from src.nodes.faiss_retriever_node import FaissRetrieverNode
from src.nodes.chatbot_node import ChatbotNode
from src.states.chatbot_state import ChatbotState

from dotenv import load_dotenv

load_dotenv()


class ChatbotGraph:
    def __init__(self):
        self.graph = StateGraph(ChatbotState)
        self.llm = GroqLLM().get_qwen32b(temperature=0.0)

    def build_graph(self):
        retriever = FaissRetrieverNode(index_path="../../data/indexes/open_ai/faiss")

        chatbot = ChatbotNode(llm=self.llm)

        self.graph.add_node("retriever", retriever.process)
        self.graph.add_node("chatbot", chatbot.process)

        self.graph.set_entry_point("retriever")
        self.graph.add_edge("retriever", "chatbot")
        self.graph.add_edge("chatbot", END)

        return self.graph.compile()


if __name__ == "__main__":
    app = ChatbotGraph().build_graph()

    # 🗨️ First turn
    state = ChatbotState(question="Who is head of the department?")
    result = app.invoke(state)

    print("\n💬 Assistant:", result["answer"])
