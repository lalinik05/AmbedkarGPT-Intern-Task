# AmbedkarGPT – RAG Prototype (Intern Task)

This project is a simple Retrieval-Augmented Generation (RAG) prototype built using a small section of *Annihilation of Caste*.  
The aim of the assignment is to show how a local, lightweight RAG pipeline works without using any paid APIs.

## 📌 Project Overview

The system performs the following steps:

- Loads the raw text from **`speech.txt`**
- Breaks the text into manageable chunks with slight overlaps
- Converts each chunk into vector embeddings using **all-MiniLM-L6-v2**
- Stores those embeddings in a local **ChromaDB** instance
- Retrieves the most relevant chunks for a user query
- Uses **Mistral** (running locally via Ollama) to generate the final answer

Everything is processed on the local machine—no external API keys needed.

---
