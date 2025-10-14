
import os
import re

from langchain.schema import HumanMessage
from dotenv import load_dotenv
from src.llms.groq_llm import GroqLLM

load_dotenv()


# Save markdown Q&A to file
def save_qa_markdown(markdown_text: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


class QAGenerator:
    def __init__(self, groq_llm: GroqLLM):
        self.llm = groq_llm.get_qwen32b()

    def generate_qa_pairs(self, markdown_text: str, k: int) -> str:
        prompt = f'''
        Generate exactly {k} question and answer pairs ONLY from the markdown content below.
        The questions should be designed from the perspective of a student seeking information about faculty.
        Each question and its answer should be numbered consecutively (e.g., 1. Question, 1. Answer, 2. Question, 2. Answer, etc.).
        The answers must be complete sentences, detailed with at least 2-3 lines, not one-word or brief phrases.
        The questions should cover faculty names, positions, research areas, education qualifications, contact details, and key research topics.
        Output the result as a markdown formatted numbered list of questions and answers.
        Do NOT output any explanations, reasoning steps, or extra text besides the markdown list.

        Markdown content:
        {markdown_text}
        '''

        response = self.llm.invoke([HumanMessage(content=prompt)])

        clean = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

        # Extract content if inside any markdown code block
        code_block_match = re.search(r"``````", clean, flags=re.DOTALL)
        if code_block_match:
            clean = code_block_match.group(1)

        return clean


def main(markdown_file_path: str, k: int):
    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    groq_llm = GroqLLM()
    generator = QAGenerator(groq_llm)

    qa_markdown = generator.generate_qa_pairs(markdown_text, k)

    base_filename = os.path.basename(markdown_file_path).replace(".md", "")
    output_path = f"../../data/processed/qa_markdowns/{base_filename}_qa.md"

    save_qa_markdown(qa_markdown, output_path)

    print(f"Saved Q&A markdown to {output_path}")


main('../../data/processed/markdowns/computer-faculty.md', 25)
