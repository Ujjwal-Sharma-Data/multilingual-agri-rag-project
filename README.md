# Multilingual Agricultural RAG Chatbot

A **production-oriented, multilingual Retrieval-Augmented Generation (RAG) system** that enables accurate question answering over agricultural PDF documents containing **dense text and complex tables**, with responses generated in the **same language as the user query**.

---

## Live Web Application

**Web Interface:**  
[https://multilingual-agri-rag-project-rsjbrqkh6vatatyeutcxnj.streamlit.app]

---

## Key Features

- PDF-based Question Answering over agricultural documents  
- Multilingual Support (Hindi, Bengali, Punjabi, Tamil, English)  
- Table-aware Retrieval for numeric and structured data  
- Retrieval-Augmented Generation (RAG) with vector search  
- MMR-based Retrieval to reduce redundant context  
- Conversational Clarification for ambiguous queries  
- Cloud-deployed Streamlit application  
- Persistent Vector Indexing to avoid repeated document processing  

---

## System Architecture
User Query -> LangChain Runnable Chains -> MMR-based Retriever (ChromaDB) -> Relevant Context (Tables + Text) -> Prompt with Grounding Rules -> Google Gemini 2.5 Flash -> Multilingual Answer


---

## Tech Stack

| Component | Technology |
|--------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | BGE-M3 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Parsing | LlamaParse |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

---

## How It Works

### Document Ingestion (One-Time)
- PDF is parsed using **LlamaParse**
- Tables and headings are preserved as Markdown
- Content is chunked using **Markdown header-based splitting**
- Chunks are embedded using **BGE-M3**
- Vector embeddings are stored persistently in **ChromaDB**

### Query Processing (Fast & Reusable)
- User query is embedded with a `query:` prefix
- Relevant chunks retrieved using **MMR**
- Context passed to Gemini 2.5 Flash with grounding constraints
- Response generated in the **user’s language**

---

## Test Examples & Expected Behavior

**The following test examples are based on the uploaded PDF _“Package of Practices for Cultivation of Vegetables”_, which was used during the development of this project.**

### Example 1 — Hindi (Table-based Query)


**User Query:**
आलू की औसत उपज कितनी है?

**Expected Output:**
आलू की औसत उपज लगभग 276.66 क्विंटल प्रति हेक्टेयर है।

---

### Example 2 — Ambiguous Query (Clarification)

**User Query:**
उपज कितनी है?

**Expected Output:**
आप किस फसल की उपज जानना चाहते हैं?

---

### Example 3 — Follow-up Query

**User Query:**
आलू की


**Expected Output:**
आलू की औसत उपज लगभग 276.66 क्विंटल प्रति हेक्टेयर है।

---

### Example 4 — Bengali

**User Query:**
আলুর গড় ফলন কত?


**Expected Output:**
আলুর গড় ফলন প্রায় 276.66 কুইন্টাল প্রতি হেক্টর।


---

### Example 5 — Out-of-Context Query

**User Query:**
धान की सिंचाई विधि क्या है?

**Expected Output:**
I don't know


---


---

## Design Decisions

### Why persistent vector storage?
- Avoids re-parsing and re-embedding documents
- Improves response time significantly
- Aligns with real-world RAG system design

### Why MMR retrieval?
- Prevents redundant chunk retrieval
- Improves answer relevance for long documents

### Why Gemini 2.5 Flash?
- Low-latency generation
- Suitable for retrieval-augmented pipelines

---

## Deployment Notes

- Python runtime locked to **3.10**
- Heavy ML components initialized lazily
- Vector indexes persisted across sessions
- Streamlit session state used for conversation handling

---


## Acknowledgements

- LangChain  
- LlamaIndex / LlamaParse  
- Hugging Face  
- Google Gemini  
- Streamlit  









