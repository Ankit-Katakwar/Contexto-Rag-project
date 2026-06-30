# 📚 Contexto

> An AI-powered PDF Intelligence application that enables natural conversations with documents using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-black)

---

## 🚀 Overview

Contexto transforms static PDF documents into an intelligent AI assistant.

Simply upload one or more PDFs, ask questions in natural language, and receive accurate answers grounded in the document content. Instead of relying solely on an LLM's memory, Contexto retrieves relevant document chunks using semantic search before generating responses, reducing hallucinations and improving answer quality.

---

## ✨ Features

- 📄 Upload and chat with multiple PDFs
- 🔍 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 💬 Conversational chat interface
- 📚 Source-aware responses
- ⚡ Fast document retrieval with ChromaDB
- 🧹 Clear Documents & Chat functionality
- 🎨 Clean Streamlit UI

---

## 🛠 Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & LLM
- LangChain
- OpenAI GPT
- HuggingFace Embeddings

### Vector Database
- ChromaDB

### Document Processing
- PyPDF
- Recursive Character Text Splitter

---

## 📂 Project Structure

```
Contexto/
│
├── app.py
├── requirements.txt
├── utils.py
├── chroma_db/
├── uploads/
├── assets/
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/contexto.git
```

Move into the project

```bash
cd contexto
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
OPENAI_API_KEY=your_api_key
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Upload one or more PDF documents.
2. PDFs are split into smaller chunks.
3. Each chunk is converted into embeddings.
4. Embeddings are stored in ChromaDB.
5. User queries are embedded.
6. Similar document chunks are retrieved.
7. The retrieved context is sent to the LLM.
8. The LLM generates an answer grounded in the document.

---

## 🔄 RAG Pipeline

```
PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Similarity Search
      │
      ▼
Retrieved Context
      │
      ▼
LLM
      │
      ▼
Final Answer
```

---

## 📸 Preview

- Upload PDFs
- Ask questions naturally
- Receive context-aware responses
- Clear documents and restart the session instantly

---

## 🎯 Future Improvements

- Multi-document citations
- Hybrid Search (BM25 + Vector Search)
- OCR support for scanned PDFs
- Conversation memory
- Local LLM support (Ollama)
- Multi-user authentication
- Document summarization
- Voice input

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ankit Katakwar**

Passionate about building production-ready AI applications with LangChain, RAG, Agentic AI, and LLMs.
