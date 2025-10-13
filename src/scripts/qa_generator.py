import json
import os
import re
from typing import List

from langchain.schema import HumanMessage
from dotenv import load_dotenv
from src.llms.groq_llm import GroqLLM

load_dotenv()


# Q&A Generator class
def save_qa_pairs(qa_pairs: List[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)


class QAGenerator:
    def __init__(self, groq_llm: GroqLLM):
        self.llm = groq_llm.get_qwen32b()

    def generate_qa_pairs(self, markdown_text: str, k: int) -> List[dict]:
        prompt = f'''
        Generate exactly {k} question and answer pairs ONLY from the markdown content below.
        The questions should cover faculty names, positions, research areas, education qualifications, contact details, and key research topics.
        Output the result as a JSON array of objects with keys "question" and "answer".
        Do NOT output any explanations, reasoning steps, or extra text besides the JSON.

        Markdown content:
        {markdown_text}
        '''

        response = self.llm.invoke([HumanMessage(content=prompt)])

        clean = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if code_block_match:
            clean = code_block_match.group(1)
        try:
            qa_pairs = json.loads(clean)
            return qa_pairs
        except json.JSONDecodeError:
            return [{"question": "N/A", "answer": clean}]


def main(markdown_file_path: str, k: int):
    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    groq_llm = GroqLLM()
    generator = QAGenerator(groq_llm)

    qa_pairs = generator.generate_qa_pairs(markdown_text, k)

    base_filename = os.path.basename(markdown_file_path).replace(".md", "")
    output_path = f"../../data/processed/qa_jsons/{base_filename}_qa.json"

    save_qa_pairs(qa_pairs, output_path)

    print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")


main('../../data/processed/markdowns/computer-faculty.md', 15)
