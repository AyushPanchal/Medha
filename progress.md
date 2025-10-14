# Date : 01/10/2025
- Created folder structure and github repo for Medha
- Setting up virtual environment, installed necessery packages.

# Date : 03/10/2025
- Data Collection for Medha, Completed.
- Raw html links converted into each individual clean markdown files
- Manually cleared all the markdown data.

# Date : 06/10/2025
- Collected scholar pages of all the CS Department faculties.
- Scrapped all the scholar pages and stored them into respective faculty name markdowns.
- Created faiss index, and ingested a single markdown into faiss vector store with `sentence-transformers/all-mpnet-base-v2`.
- Also ran few queries on faiss vector store.

# Date : 08/10/2025
- Ingested all the markdown files into faiss vector store. Created an Ingestion Pipeline for that
- Created and tested Groq LLM.
- Created a FAISS Retriever and tested it.

# Date : 09/10/2025
- Ingested all the markdown files into faiss vector store. Created an Ingestion Pipeline for that.
- Created metadata for all the markdowns using llm, and ingested it into faiss database.
- Retrieval quality was still lower after ingesting the metadata.
- Created langgraph dev environment.

# Date : 10/10/2025
- Changed the embeddings with OpenAI Embeddings
- Retrieval quality is good with open ai.

# Date : 11/10/2025
- Created langgraph json file.
- updated chatbot state.
- created faiss retriever node to attach it in graph.

# Date : 13/10/2025
- Working on generating questions and answers pair from the data, for better retrieval.

# Date : 14/10/2025
- Created prompt loader and converted 1 prompt into markdown file.
- Introduced memory to our chatbot.
- Introduced query reformulation for user asked questions for standalone similarity search.