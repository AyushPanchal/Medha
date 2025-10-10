from src.ingestion.ingestion_pipeline_with_metadata import IngestionPipelineWithMetadata

# pipeline = IngestionPipelineWithMetadata(
#     md_dir="../../data/processed/markdowns",
#     persist_dir="../../data/indexes/hugging_face_all_mpnet_base_v2/faiss",
#     use_open_ai_embeddings=True,
# )

pipeline = IngestionPipelineWithMetadata(
    md_dir="../../data/processed/markdowns",
    persist_dir="../../data/indexes/open_ai/faiss",
    use_open_ai_embeddings=True,
)

# Build and persist vector store
vectordb = pipeline.build_and_persist()
vectordb = pipeline.load()
results = vectordb.similarity_search("head of the department ? ", k=5)
for r in results:
    print(r.metadata.get("relpath"), "->", r.page_content[:120])