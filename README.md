# 📺 RAG-powered Chatbot for YouTube Transcript Understanding

A modular RAG-powered chatbot for querying YouTube video transcripts using embeddings, vector search, and large language models.

---

## 🧭 Overview

This project allows you to *chat with any public YouTube video* by asking questions in natural language. It retrieves relevant parts of the transcript using a *Retrieval-Augmented Generation (RAG)* pipeline, then generates fluent answers using a *language model* like Gemini.

The conversational tone of the chatbot is fully *customizable via prompt engineering* — from poetic to technical, from mentor-style to casual assistant.

---

## 🚀 Features

- 💬 Ask questions about any YouTube video
- 🎯 Uses *RAG* with hybrid retrieval: score filtering + clustering + MMR
- 🧠 Embeds text with sentence-transformers
- 🧲 Stores embeddings locally with *ChromaDB*
- 🎛 Tune chunk size, retrieval diversity, and top-k relevance
- 🔐 Manages secrets via .streamlit/secrets.toml
- ♻ Vector database reset & deletion from the UI
- 🧵 Maintains chat history during the session
- ✅ Modular, extensible architecture

---

## ⚙ Getting Started

### 🛠 *1. Clone the Repository*

bash
git clone https://github.com/kumar2302github/RAG-powered-chatbot-for-YouTube-transcript-understanding.git
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