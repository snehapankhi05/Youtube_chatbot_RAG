# Production-Grade Hybrid YouTube Chatbot using RAG 🎬

An advanced, production-ready Retrieval-Augmented Generation (RAG) conversational pipeline built with Streamlit. The application reads any public English YouTube video URL, fetches its transcript, processes chunks, and implements a high-performance **Hybrid Search Engine** to handle granular user Q&A.

## 🚀 Key Features

* **Hybrid Search Pattern:** Combines dense semantic vector search (**FAISS** + `all-MiniLM-L6-v2` embeddings) with sparse keyword matching (**BM25 Search**) to provide highly relevant context window snapshots.
* **Reciprocal Rank Fusion (RRF):** Implements an algorithmic cross-ranking mechanism to interleave, score, and dedup vector and keyword candidate results seamlessly.
* **Production State Retention:** Leverages Streamlit's `st.session_state` API caching layer, preventing data loss, redundant API recalculations, or model instantiation loops during UI hot-reloads.
* **Ultrafast LLM Synthesis:** Backed by **ChatGroq** running `llama-3.1-8b-instant` for ultra-low latency contextual answers.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository & Navigate In
```bash
cd "Youtube Chatbot using RAG"