from flask import Flask, request, jsonify
from src.graph.chatbot_graph import ChatbotGraph
from src.states.chatbot_state import ChatbotState
from langchain_core.runnables import RunnableConfig
import uuid

app = Flask(__name__)

# Build chatbot graph
chatbot_graph = ChatbotGraph().build_graph()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question'"}), 400

        question = data["question"]
        thread_id = data.get("thread_id")  # ✅ reuse if provided
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Create initial state
        state = ChatbotState(question=question, messages=data.get("messages", []))
        config = RunnableConfig(configurable={"thread_id": thread_id})

        result = chatbot_graph.invoke(state, config=config)
        if isinstance(result, dict):
            result = ChatbotState(**result)

        docs = []
        if result.source_docs:
            for doc in result.source_docs:
                if hasattr(doc, "page_content"):
                    docs.append({
                        "page_content": doc.page_content,
                        "metadata": getattr(doc, "metadata", {})
                    })

        response = {
            "thread_id": thread_id,
            "question": result.question,
            "answer": result.answer,
            "context": result.context,
            "source_docs": docs,
            "messages": result.messages,  # Updated chat history
            "summary": result.summary()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
