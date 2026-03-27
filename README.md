#  RAG Document Chatbot — Local LLM Pipeline

A fully local **Retrieval-Augmented Generation (RAG)** system that lets you chat with your documents — no APIs, no cloud, no data leaving your machine. Built with Hugging Face Transformers, Sentence Transformers, Chroma, and Streamlit.

---

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How It Works](#how-it-works)
- [Design Decisions](#design-decisions)
- [Example](#example)
- [Potential Improvements](#potential-improvements)


---

## Overview

This project implements a **complete RAG pipeline** that enables semantic question-answering over a local document corpus. The system retrieves the most relevant document chunks using vector similarity and generates grounded, context-aware answers using a local language model — entirely offline.

**Key highlights:**
-  Fully local — no external API calls
-  Semantic search powered by sentence embeddings
-  Supports multi-document ingestion and chunking
-  Lightweight models optimized for CPU execution
-  Clean Streamlit interface for interactive querying

---

## Architecture
```
Input Documents
      │
      ▼
  Text Chunking (overlapping windows)
      │
      ▼
  Embeddings — all-MiniLM-L6-v2
      │
      ▼
  Chroma Vector DB (persisted locally)
      │
      ▼
  User Query → Query Embedding
      │
      ▼
  Cosine Similarity Search (Top-k + threshold filter)
      │
      ▼
  Context Construction → Structured Prompt
      │
      ▼
  FLAN-T5 Answer Generation
      │
      ▼
  Streamlit UI Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Generation | Hugging Face Transformers (FLAN-T5) |
| Orchestration | LangChain, LangChain-Community |
| UI | Streamlit |

---

## Project Structure
```
rag_llm/
│
├── create_database.py      # Document ingestion, chunking, and vector indexing
├── retrieval.py            # Semantic similarity retrieval logic
├── llm_handler.py          # FLAN-T5 generation layer
├── query_data.py           # CLI-based querying interface
├── app.py                  # Streamlit web application
│
├── chroma/                 # Persisted Chroma vector database
└── data/
    └── books/              # Place your input documents here (.txt, .pdf, etc.)
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag_llm.git
cd rag_llm
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install sentence-transformers
pip install transformers
pip install langchain langchain-community chromadb
pip install streamlit
```

>  All models are downloaded automatically on first run and cached locally.

### 4. Add Your Documents

Place your source documents (`.txt`, `.pdf`, etc.) inside the `data/books/` directory.

### 5. Build the Vector Database
```bash
python create_database.py
```

This step chunks your documents, generates embeddings, and persists them to the `chroma/` directory.

### 6. Launch the Streamlit App
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501` to start querying your documents.

---

## How It Works

### >> Document Processing
Documents in `data/books/` are split into overlapping text chunks. Overlapping windows preserve context across chunk boundaries, improving retrieval quality.

### >> Embedding Generation
Each chunk is encoded into a dense vector representation using `all-MiniLM-L6-v2` — a lightweight yet highly effective semantic embedding model.

### >> Vector Storage
Embeddings are stored and indexed in **ChromaDB**, a lightweight vector database that supports efficient similarity search and local persistence.

### >> Retrieval
Incoming user queries are embedded using the same model and matched against stored vectors via **cosine similarity**. Only results above a relevance threshold are returned (top-k filtering).

### >> Answer Generation
Retrieved chunks are assembled into a structured prompt and passed to **FLAN-T5** — an encoder-decoder language model — for grounded response generation. The model is constrained to answer from the provided context, reducing hallucination.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Fully local execution | Eliminates API dependency and data privacy concerns |
| Modular architecture | Separates ingestion, retrieval, and generation for maintainability |
| MiniLM for embeddings | Fast, accurate, and CPU-friendly |
| FLAN-T5 for generation | Lightweight encoder-decoder; solid instruction-following on CPU |
| Overlapping chunks | Preserves context at chunk boundaries for better retrieval |
| Score threshold filtering | Prevents low-relevance chunks from polluting the context window |

---

## Example

**Query:**
```
Who is Alice in the story?
```

**Generated Response:**
```
Alice is a young girl who appears in the story.
```

---

## Potential Improvements

- [ ] **Stronger LLM** — Replace FLAN-T5 with LLaMA 3, Mistral, or Phi-3 for richer reasoning
- [ ] **Hybrid Retrieval** — Combine BM25 keyword search with dense embeddings for better recall
- [ ] **Re-ranking** — Add a cross-encoder re-ranker to improve top-k precision
- [ ] **Conversational Memory** — Maintain dialogue history for multi-turn question answering
- [ ] **Document Metadata Filtering** — Filter retrieval by source, date, or topic tags
- [ ] **API Deployment** — Expose the pipeline as a REST API using FastAPI
- [ ] **Support More File Types** — Extend ingestion to handle `.pdf`, `.docx`, `.csv`

---


