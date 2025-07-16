# RahulGPTPlusPlus
# 🧠 RBI Circulars Q&A Chatbot (Fully Local RAG)

This project is a **fully offline Retrieval-Augmented Generation (RAG) chatbot** built to parse and understand **RBI circulars (PDFs)**. It allows users to upload circulars and ask domain-specific questions, returning answers based on actual content — with **page-level citation tracking**.

Designed for demonstration within **Risk & Compliance departments** (e.g. IARC), this solution runs **completely offline** without any cloud model calls or token-based usage.

---

## 🔧 Features

- ✅ **Fully Local LLM (No Cloud/Token Usage)** using `ctransformers` + Mistral-7B (GGUF)
- ✅ **HuggingFace MiniLM Embeddings** for compact yet accurate semantic search
- ✅ **FAISS Vector Store** for fast, persistent similarity search
- ✅ **PDF Parsing via PyMuPDF** with page-level metadata
- ✅ **Deduplication via SHA-256 Hashing** to prevent reprocessing files
- ✅ **Token-aware Prompt Construction** capped at 512 tokens (100 reserved for output)
- ✅ **Fallback LLM Answers** for vague queries or missing context
- ✅ **Source & Page Number Tracking** in chatbot responses

---

## 📂 Project Structure

<pre> RBI_RAG_DEMO/ ├── models/ │ └── mistral/ │ └── mistral-7b-instruct-v0.1.Q4_K_M.gguf # Local LLM model file │ ├── faiss_index/ # FAISS vector database files │ ├── index.faiss │ └── index.pkl │ ├── vectorstore/ # (Optional) Legacy folder for Chroma (if used) │ ├── hash_store.json # JSON to track hashes of processed PDFs ├── QA_bot.py # Main Gradio-based chatbot script ├── requirements.txt # Python dependencies └── README.md # Project documentation </pre>

---

## 🖥️ How It Works

1. **PDF Upload**
   - Users drag and drop RBI circulars in `.pdf` format.
   - Pages are parsed using PyMuPDF with page number metadata.

2. **Chunking & Compression**
   - Text is split using `RecursiveCharacterTextSplitter` and lightly compressed (remove blank lines, simplify legal terms).

3. **Vector Embedding & Storage**
   - Text chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
   - Stored locally in a FAISS vector index with metadata (filename + page).

4. **Question Answering (RAG)**
   - On user query:
     - Top 5 similar chunks are retrieved.
     - Token-aware prompt is created with source tagging.
     - Answer is generated via `ctransformers` running a local Mistral model.

5. **Fallback Handling**
   - If no good matches are found, LLM gives a generic but context-aware response.

---
