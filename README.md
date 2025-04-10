# 📺 RAG-powered Chatbot for YouTube Transcript Understanding

*A modular, Retrieval-Augmented Generation (RAG) system to query YouTube video transcripts using embeddings, vector search, and large language models.*

---

## 🧭 Overview

This project allows you to **chat with any public YouTube video** in natural language. It extracts the video transcript, indexes it intelligently, and returns responses using a **hybrid retrieval system** and a **language model** like Gemini.

The conversational tone of responses is **customizable through prompt engineering** — whether poetic, technical, or instructional.

---

## 🚀 Features

- 💬 Ask questions about any YouTube video
- 🔎 Hybrid retrieval pipeline (relevance + diversity)
- 🧠 Uses `sentence-transformers` for semantic embeddings
- 🧲 ChromaDB for fast, local vector storage
- 🔧 Sidebar to adjust:
  - Top-K relevant chunks
  - Score threshold
  - Chunk size & overlap
  - MMR lambda for diversity control
- 🔁 Reset and delete vector stores
- 🧵 Maintains session-based chat history
- 📦 Modular architecture for easy extension

---

## 📂Project Structure: RAG_CHAT

RAG_CHAT/
├── main/
│   ├── app.py                # Streamlit app
│   ├── rag.py                # RAG logic: embedding, retrieval, generation
│   └── transcription.py      # YouTube transcript fetcher
│
├── .streamlit/
│   └── secrets.toml          # Gemini API key (private, not pushed)
│
├── rag_test.ipynb            # Experimental notebook for testing logic
├── README.md                 # Full project documentation
├── requirements.txt          # Dependency list for pip installation
---

## ⚙ Getting Started

### 🛠 *1. Clone the Repository*

bash
git clone https://github.com/kumar2302github/RAG-powered-chatbot-for-YouTube-transcript-understanding_.git
cd RAG-powered-chatbot-for-YouTube-transcript-understanding


### ⚙ *2. Create a Virtual Environment*

bash
python -m venv venv
venv\Scripts\activate    # For Windows
# or
source venv/bin/activate # For Mac/Linux


### 📦 *3. Install Requirements*

bash
pip install -r requirements.txt


### 🔐 *4. Add API Key for Gemini*

Create a file .streamlit/secrets.toml and add:

toml
GEMINI_API_KEY = "your_google_gemini_api_key"


### 🚀 *5. Launch the App*

bash
streamlit run app/app.py


---

## 💡 How It Works

### 📼 *Transcript Extraction*
Retrieves video transcript via youtube-transcript-api.

### ✂ *Text Chunking & Embedding*
Splits the transcript into chunks and embeds them using sentence-transformers.

### 🧲 *Vector Storage*
Stores chunk embeddings in a local Chroma vector database per video.

### 🔍 *Hybrid Retrieval*
Uses:
- Score threshold filtering
- HDBSCAN clustering
- Maximal Marginal Relevance (MMR)

### ✨ *Response Generation*
Selected chunks and the user query are sent to Gemini (or another LLM) with a configurable prompt to generate a fluent, context-aware response.

---

## 🎛 Sidebar Settings

| Parameter            | Description                                                   |
|---------------------|---------------------------------------------------------------|
| *Top-K*           | Number of top chunks to retrieve                              |
| *Score Threshold* | Max distance between query and chunk (lower = better match)   |
| *MMR Lambda*      | Relevance vs. diversity balance for retrieval                 |
| *Chunk Size*      | Length of each embedded text segment                          |
| *Chunk Overlap*   | Number of overlapping tokens between chunks                   |
| *Delete Embeddings* | Remove vector DBs from previous video sessions                |

---
