import json
from pathlib import Path
from dotenv import load_dotenv
import re
load_dotenv()

from src.llms.groq_llm import GroqLLM

# Initialize Groq LLM
groq_llm = GroqLLM().get_qwen32b()


def generate_metadata(file_text: str) -> dict:
    """
    Uses Groq LLM to generate title, tags, and summary for a given document text.
    Handles cases where LLM wraps JSON in markdown-style code blocks.
    """
    prompt = f"""
You are a document metadata generator.
Read the following text and output JSON metadata with 3 fields:
1. title: a short descriptive title
2. tags: 3-5 keywords relevant to the document
3. summary: 2-3 sentence concise summary

Document:
\"\"\"{file_text}\"\"\"
"""
    response = groq_llm.invoke(prompt)

    # Access content from AIMessage object
    text = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

    # Remove markdown code block if present
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # Parse JSON safely
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON. Returning raw text as summary.")
        metadata = {
            "title": "Unknown",
            "tags": [],
            "summary": text
        }

    return metadata


def estimate_tokens(text: str) -> int:
    """
    Rough estimate: 1 token ≈ 4 characters (adjustable).
    Used to skip files exceeding Groq token limit.
    """
    return len(text) // 4


def process_markdown_files(directory: str):
    """
    Walks through Markdown files in a directory, generates metadata, and stores results.
    Skips files with estimated tokens > 5900.
    """
    markdown_dir = Path(directory)
    all_docs_metadata = []

    for md_file in markdown_dir.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as file:
            text = file.read()

        tokens = estimate_tokens(text)
        if tokens > 5900:
            print(f"Skipped {md_file.name} (estimated tokens: {tokens} > 5900)")
            continue

        metadata = generate_metadata(text)
        # Add file-related info
        metadata.update({
            "source": str(md_file.resolve()),
            "filename": md_file.name,
            "relpath": str(md_file.relative_to(markdown_dir))
        })

        all_docs_metadata.append(metadata)
        print(f"Processed {md_file.name}: {metadata['title']}")

    return all_docs_metadata


if __name__ == "__main__":
    directory_path = "../../data/processed/markdowns"  # Change to your Markdown folder
    docs_metadata = process_markdown_files(directory_path)

    # Save metadata to JSON file
    with open("../../data/artifacts/metadata_output.json", "w", encoding="utf-8") as f:
        json.dump(docs_metadata, f, indent=4, ensure_ascii=False)

    print("Metadata generation complete. Saved to metadata_output.json")
