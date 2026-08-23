# RAG PDF Chatbot

A Streamlit chatbot that answers questions grounded in a specific PDF document, using a retrieval-augmented generation pipeline.

## How it works
- Splits the PDF into chunks and embeds them with HuggingFace embeddings
- Stores embeddings in a FAISS vector index for similarity search
- Retrieves relevant chunks and feeds them to a Groq LLM (Llama3-8b) via LangChain's retrieval chain

## Tech stack
Python, LangChain, Groq, HuggingFace Embeddings, FAISS, Streamlit

## Run locally
set GROQ_API_KEY and HUGGING_FACE in a .env file
streamlit run ChatBot.py
