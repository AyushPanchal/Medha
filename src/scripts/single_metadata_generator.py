import math
import re
import json
from pathlib import Path
from src.llms.groq_llm import GroqLLM
from dotenv import load_dotenv

load_dotenv()
# Initialize Groq LLM
groq_llm = GroqLLM().get_qwen32b()


def chunk_text(text: str, max_chars: int = 2000):
    """
    Splits text into chunks of maximum max_chars characters.
    Returns a list of text chunks.
    """
    chunks = []
    total_length = len(text)
    num_chunks = math.ceil(total_length / max_chars)

    for i in range(num_chunks):
        start = i * max_chars
        end = start + max_chars
        chunks.append(text[start:end])
    return chunks


def generate_metadata_for_large_file(file_path: str) -> dict:
    """
    Generates metadata (title, tags, summary) for a large Markdown file by:
    1. Chunking the text into smaller parts.
    2. Generating metadata for each chunk with the LLM.
    3. Combining chunk metadata into a final unified metadata dictionary.
    """
    md_file = Path(file_path)
    if not md_file.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, max_chars=2000)  # Adjust based on token estimate
    titles = []
    tags = set()
    summaries = []

    for idx, chunk in enumerate(chunks, start=1):
        prompt = f"""
You are a document metadata generator.
Read the following text and output JSON metadata with 3 fields:
1. title: a short descriptive title
2. tags: 3-5 keywords relevant to the document
3. summary: 2-3 sentence concise summary

Document:
\"\"\"{chunk}\"\"\"
"""
        response = groq_llm.invoke(prompt)
        text_resp = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        # Remove markdown code block if present
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_resp, re.DOTALL)
        if code_block_match:
            text_resp = code_block_match.group(1)

        # Parse JSON safely
        try:
            metadata = json.loads(text_resp)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for chunk {idx}. Using raw text as summary.")
            metadata = {
                "title": f"Unknown chunk {idx}",
                "tags": [],
                "summary": text_resp
            }

        titles.append(metadata.get("title", f"Chunk {idx}"))
        tags.update(metadata.get("tags", []))
        summaries.append(metadata.get("summary", ""))

        print(f"Processed chunk {idx}/{len(chunks)}")

    # Combine chunk metadata into final metadata
    final_metadata = {
        "title": " | ".join(titles[:3]) if titles else "Unknown",  # take first 3 chunk titles
        "tags": list(tags),
        "summary": " ".join(summaries[:3])  # take first 3 chunk summaries to make concise
    }

    return final_metadata


if __name__ == "__main__":
    large_md_file = "../../data/processed/markdowns/Dipti-Rana.md"  # Change to your file path
    metadata = generate_metadata_for_large_file(large_md_file)
    print("\nFinal Metadata:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
