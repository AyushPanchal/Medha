from langgraph.graph import StateGraph, END
from src.nodes.faiss_retriever_node import FaissRetrieverNode
from src.states.chatbot_state import ChatbotState

from dotenv import load_dotenv

load_dotenv()


class ChatbotGraph:
    def __init__(self):
        # Initialize the graph with the ChatbotState
        self.graph = StateGraph(ChatbotState)

    def build_graph(self):
        faiss_retriever_node = FaissRetrieverNode(index_path="data/indexes/open_ai/faiss",
                                                  model_name="text-embedding-3-large")

        # Add nodes
        self.graph.add_node("retriever", faiss_retriever_node.process)

        # Define edges
        self.graph.set_entry_point("retriever")
        self.graph.add_edge("retriever", END)

        # Compile the graph
        compiled_graph = self.graph.compile()
        return compiled_graph


# ================================
# Top-level graph for LangGraph
# ================================
builder = ChatbotGraph()
app = builder.build_graph()  # <- Exposed at top-level

