import json
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.ingestion_pipeline import IngestionPipeline

# Load your LLM-generated metadata once
METADATA_JSON_PATH = "../../data/artifacts/metadata_output.json"
with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

# Map filename -> metadata for quick lookup



for i, m in enumerate(metadata_list):
    if "filename" not in m:
        print(f"Missing 'filename' in metadata_list[{i}]: {m}")

metadata_map = {m["filename"]: m for m in metadata_list}


class IngestionPipelineWithMetadata(IngestionPipeline):
    """
    Extends the existing IngestionPipeline to attach LLM-generated metadata
    (title, tags, summary) to each Document chunk during ingestion.
    """

    def _load_file(self, path: Path) -> list[Document]:
        # Call the original loader
        docs = super()._load_file(path)

        # Get LLM-generated metadata for this file
        llm_meta = metadata_map.get(path.name, {})

        for d in docs:
            # Merge LLM metadata into the Document metadata
            d.metadata.update({
                "title": llm_meta.get("title", "Unknown"),
                "tags": llm_meta.get("tags", []),
                "summary": llm_meta.get("summary", "")
            })

        return docs
