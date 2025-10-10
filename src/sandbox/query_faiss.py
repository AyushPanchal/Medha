from src.retrievers.faiss_retriever import FaissRetriever
from dotenv import load_dotenv

load_dotenv()

retriever = FaissRetriever(
    index_path=r"../../data/indexes/open_ai/faiss",
    model_name="text-embedding-3-large",
    use_open_ai_embeddings=True,
    k=5
)

# 1️Retrieve normally
retriever.preview("Who is the hod ? ")


