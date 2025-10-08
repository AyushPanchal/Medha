from src.llms.groq_llm import GroqLLM
import re
from dotenv import load_dotenv

load_dotenv()

llm = GroqLLM().get_qwen32b(temperature=0.1)
response = llm.invoke("Summarize LangChain + Groq in one sentence.")

clean = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
print(clean)
