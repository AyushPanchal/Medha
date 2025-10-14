# scripts/build_index.py
from src.ingestion.ingestion_pipeline import IngestionPipeline


pipeline = IngestionPipeline(
    md_dir="../../data/processed/markdowns",
    persist_dir="../../data/indexes/open_ai/faiss",
    glob_pattern="**/*.md",
    chunk_size=1000,
    chunk_overlap=150,
    loader_mode="single",
    loader_strategy="fast",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    use_open_ai_embeddings=True
)

# Build and persist
vectordb = pipeline.build_and_persist()

# Later load and query
vectordb = pipeline.load()
results = vectordb.similarity_search("head of the department ? ", k=5)
for r in results:
    print(r.metadata.get("relpath"), "->", r.page_content[:120])
