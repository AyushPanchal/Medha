import os
import pathlib
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv

load_dotenv()

MD_PATH = r"../../data/processed/qa_markdowns/computer-faculty_qa.md"
PERSIST_DIR = r"../../data/indexes/open_ai/faiss"

def main():
    if not os.path.isfile(MD_PATH):
        raise FileNotFoundError(f"Markdown file not found: {MD_PATH}")

    # Load new documents from markdown
    loader = UnstructuredMarkdownLoader(MD_PATH, mode="single", strategy="fast")
    docs = loader.load()
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = str(pathlib.Path(MD_PATH).resolve())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Load existing FAISS index if available
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Existing index found, load it
        vectordb = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
        # Create new FAISS index for new chunks
        new_vectordb = FAISS.from_documents(chunks, embeddings)
        # Merge new index into existing index
        vectordb.merge_from(new_vectordb)
    else:
        # No existing index, create new one
        vectordb = FAISS.from_documents(chunks, embeddings)

    # Save merged or new index back to disk
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectordb.save_local(PERSIST_DIR)

    print(f"Saved FAISS index to: {PERSIST_DIR}")
    print(f"Total chunks indexed including previous ones: {len(vectordb.docstore)}")

if __name__ == "__main__":
    main()
