# ⬡ CodeSense AI — Code Explainer
> 🧠 AI-Powered Code Understanding System using LLM + RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

### LLM + RAG Powered Code Understanding System

> Academic Research Project · Streamlit + OpenAI + FAISS

---

## 📁 Project Structure

```
code_explainer/
├── app.py              # 🖥️  Main Streamlit UI (entry point)
├── llm_handler.py      # 🤖  LLM integration (OpenAI / demo mode)
├── rag_pipeline.py     # 🔍  RAG pipeline (chunking + FAISS + retrieval)
├── embeddings.py       # 🔢  Embedding models (SentenceTransformers / OpenAI)
├── utils.py            # 🛠️  Shared utilities (detection, formatting, etc.)
├── requirements.txt    # 📦  Python dependencies
└── README.md           # 📖  This file
```

---

## ⚡ Quick Start (5 minutes)

### 1 — Prerequisites
- Python 3.9 or later
- pip (comes with Python)
- An OpenAI API key (optional — Demo Mode works without one)

### 2 — Clone / Download
```bash
# If you have git:
git clone <your-repo-url>
cd code_explainer

# Or just unzip the downloaded folder and cd into it
```

### 3 — Create Virtual Environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

> ⏳ First install downloads sentence-transformers (~90 MB) — this only happens once.

**Minimal install (no local embeddings):**
```bash
pip install streamlit openai numpy faiss-cpu
```
The app will use hash-based embeddings as a fallback — still functional!

### 5 — Run the App
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** 🎉

---

## 🔑 API Key Setup

**Option A — Enter in the app:**
Open the left sidebar → paste your OpenAI API key in the `🔑 API Configuration` box.

**Option B — Environment variable (more secure):**
```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```
Then modify `llm_handler.py` line where `LLMHandler` is created to read `os.getenv("OPENAI_API_KEY")`.

**Option C — .env file:**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```
Install `python-dotenv` (already in requirements) and add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py`.

**No API key?** Enable **Demo Mode** in the sidebar — you'll get rule-based responses without any LLM calls.

---

## 🧩 How It Works — Architecture

```
User Code Input
      │
      ▼
┌─────────────┐    chunk_by_function()
│  RAG Pipeline│ ──────────────────────► Code Chunks
│  rag_pipeline│
│  .py         │    EmbeddingModel        FAISS Index
│             │ ──────────────────────► (vector store)
└─────────────┘
      │
      │  retrieve(query, top_k)
      ▼
Retrieved Context Snippets
      │
      ▼
┌─────────────┐
│ LLM Handler │  system prompt + code + RAG context
│ llm_handler │ ──────────────────────────────────► OpenAI API
│ .py         │
└─────────────┘
      │
      ▼
   Response → Streamlit UI
```

### RAG Pipeline Details
1. **Chunk** — Code is split at function/class boundaries (or by word count)
2. **Embed** — Each chunk → 384-dim vector via `all-MiniLM-L6-v2` (free, local)
3. **Index** — Vectors stored in FAISS `IndexFlatL2` (in-memory, no server needed)
4. **Retrieve** — Top-K nearest chunks fetched for the user query
5. **Augment** — Retrieved chunks + user code passed to the LLM as context

---

## 🎮 Features

| Feature | Description |
|---|---|
| 🔍 Explain | Detailed line-by-line explanation with overview and use cases |
| 📝 Summarize | Concise summary with inputs/outputs and complexity |
| 🐛 Debug | Bug detection, root cause analysis, fix suggestions |
| ⚡ Optimize | Performance tips, refactored code, complexity comparison |
| 💬 Chat | Multi-turn Q&A with the code as persistent context |
| 📂 Upload | Drag-and-drop .py, .java, .cpp, .js files |
| 🌍 Multi-language | Python, JavaScript, TypeScript, Java, C++, Go, Ruby, Rust |
| 🔍 Auto-detect | Heuristic language detection from code patterns |
| 🧠 RAG | Built-in knowledge base of 30+ programming best practices |
| 🎨 Dark UI | Terminal-aesthetic Streamlit interface with animations |

---

## 🛠️ Customisation

### Add More Knowledge Base Entries
Edit the `KNOWLEDGE_BASE` list in `rag_pipeline.py`:
```python
("python", "Your custom tip or documentation snippet here"),
```

### Change the LLM Prompts
Edit the `ACTION_PROMPTS` dictionary in `llm_handler.py`.

### Use a Different Embedding Model
Edit `embeddings.py`, `_try_sentence_transformers()` method:
```python
self._st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # better quality
```

### Swap OpenAI for a Local LLM (Ollama)
In `llm_handler.py`, replace the OpenAI client with:
```python
import openai
self._client = openai.OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key  = "ollama",
)
self.model = "codellama"  # or mistral, llama3, etc.
```
(Requires [Ollama](https://ollama.ai) installed and running locally)

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| `openai.AuthenticationError` | Check your API key in the sidebar |
| App is slow on first run | SentenceTransformers downloads the model once (~90 MB) |
| `torch` install fails | Try `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## 📚 Academic References

- **RAG**: Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **FAISS**: Johnson et al. (2017) — *Billion-scale similarity search with GPUs*
- **SentenceTransformers**: Reimers & Gurevych (2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT Networks*
- **GPT Models**: OpenAI (2023) — *GPT-4 Technical Report*

---

## 📄 License
MIT — Free to use, modify, and distribute for academic and personal projects.
