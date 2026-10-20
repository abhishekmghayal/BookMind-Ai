# 📚 BookMind AI — Chat With Your Books Using RAG

BookMind AI is a Retrieval-Augmented Generation (RAG) application that lets you upload a PDF book, retrieve relevant passages, and ask questions about its content through a Streamlit chat interface.

It combines **PDF text extraction, text chunking, Hugging Face embeddings, ChromaDB vector search, MMR retrieval, Multi-Query retrieval, and a Groq-hosted language model** to generate context-grounded answers.

> **Project status:** First RAG learning project. This README documents the current design and setup. Some details may need updating as the project evolves.

---

## ✨ Features

- **PDF ingestion:** Upload a text-based PDF and extract its pages.
- **Text chunking:** Split extracted text into overlapping chunks for retrieval.
- **Embeddings:** Convert text into vectors with `sentence-transformers/all-MiniLM-L6-v2`.
- **Persistent vector storage:** Store document vectors and metadata in a local `chroma_db/` directory.
- **MMR retrieval:** Retrieve relevant chunks while reducing redundant results.
- **Multi-Query retrieval:** Generate alternative versions of a question to broaden retrieval.
- **LLM-powered answers:** Use a Groq-hosted chat model to answer from retrieved context.
- **Source inspection:** Show retrieved passages and PDF page references in the UI.
- **Streamlit interface:** Upload a book and chat with it in a browser.
- **Environment-based secrets:** Load the Groq API key from a local `.env` file.

## 🧠 How RAG Works in BookMind AI

RAG stands for **Retrieval-Augmented Generation**. Rather than asking the language model to answer from its general knowledge alone, the application first retrieves relevant text from the uploaded document and then supplies that text as context for the answer.

```text
PDF book
   │
   ▼
PDF text extraction (PyPDFLoader)
   │
   ▼
Text splitting (RecursiveCharacterTextSplitter)
   │
   ▼
Embeddings (Hugging Face / Sentence Transformers)
   │
   ▼
ChromaDB vector store
   │
   ▼
User question
   │
   ▼
Multi-Query Retriever ── generates alternative search queries
   │
   ▼
MMR retrieval ── chooses relevant, less-redundant chunks
   │
   ▼
Retrieved context + question
   │
   ▼
Groq chat model
   │
   ▼
Answer + retrieved source passages
```

The current implementation uses a Multi-Query retriever wrapping a Chroma retriever configured for MMR. The exact retrieval results depend on the document, embedding model, retrieval settings, and generated alternative queries.

## 🧰 Technology Stack

| Technology / package | Role in this project |
|---|---|
| Python | Main programming language |
| Streamlit | Web UI, PDF uploader, chat interface, session state |
| LangChain Core | Prompt templates, message formats, and core interfaces |
| LangChain Community | PDF document loader and community integrations |
| LangChain Text Splitters | Split page text into chunks with overlap |
| LangChain Hugging Face | Connect LangChain to Hugging Face embedding models |
| Sentence Transformers | Produce vector embeddings from text |
| PyTorch / Transformers | Model runtime and supporting ML components for embeddings |
| ChromaDB | Store and search vector embeddings and metadata |
| LangChain Chroma | Chroma vector-store integration for LangChain |
| LangChain Classic | `MultiQueryRetriever` in the current app |
| LangChain Groq | Connect LangChain to a Groq-hosted chat model |
| pypdf | PDF text extraction used by `PyPDFLoader` |
| python-dotenv | Load local environment variables from `.env` |
| httpx | HTTP client dependency used by integrations |
| Pydantic | Data and configuration validation used by dependencies |

See [`BookMind_AI_First_RAG_Explained_Requirements.txt`](BookMind_AI_First_RAG_Explained_Requirements.txt) or [`requirements.txt`](requirements.txt) for the dependency list in this repository. The explained requirements file describes why each direct dependency is included.

## 🔎 Retrieval Concepts Used

### 1. Embeddings

The app uses:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

This model creates **384-dimensional embeddings**. Use the same embedding model when reopening a Chroma collection that was populated with it. Matching vector dimensions alone does not guarantee that two different embedding models are interchangeable.

### 2. ChromaDB

ChromaDB stores the vector representation of each text chunk along with metadata such as its source and page number. The app stores its local database in `chroma_db/`.

The local vector database is generated data and is excluded from Git by default. When another developer runs the app, they can upload and index their own PDF.

### 3. MMR (Maximal Marginal Relevance)

MMR balances query relevance with diversity among selected chunks. It helps reduce redundant results when several chunks contain similar information.

Current settings in the console RAG code:

```python
search_kwargs={
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

- `k=4`: Return up to four final chunks.
- `fetch_k=10`: Consider up to ten candidates before selecting the final results.
- `lambda_mult=0.5`: Balance relevance and diversity equally as a starting point.

### 4. Multi-Query Retrieval

`MultiQueryRetriever` asks the chat model to generate alternative phrasings of the question and retrieves documents for those queries. The results are combined to broaden the search.

Multi-Query can improve retrieval coverage but adds LLM calls, latency, and potentially API cost. MMR and Multi-Query solve different problems and can be used together.

### 5. Groq chat model

The app uses `ChatGroq` to generate answers from the retrieved context. The current code's default model identifier is `qwen/qwen3.8-27b`; model availability can change, so use a model ID supported by your Groq account.

The prompt instructs the model to answer using only the supplied context and to say it could not find the answer when the context does not support it. This instruction helps constrain answers, but does not guarantee that every answer or citation is correct.

## 📁 Project Structure

The following is a representative structure based on the files used in the project. Keep the names aligned with your actual repository.

```text
BookMind-Ai/
├── .gitignore
├── .env                         # Local only; never commit
├── README.md
├── requirements.txt
├── BookMind_AI_First_RAG_Explained_Requirements.txt
├── main.py                      # Console-based RAG chat
├── streamlit_app.py             # Streamlit PDF-upload/chat UI
├── create_database.py            # Build the Chroma index from a PDF
├── documentLoaders/
│   ├── GRU.pdf                  # Local sample document; check licensing before publishing
│   ├── notes.txt
│   ├── page.py
│   ├── pdf.py
│   └── test.py
├── retrivers/
│   ├── arxiv_retriever_demo.py
│   ├── mmr.py
│   └── multiquery.py
├── vector_store/
│   └── db.py
└── chroma_db/                   # Generated local database; do not commit
```

> Note: `retrivers/` preserves the spelling in the current project. You may choose to rename it to `retrievers/` later, updating imports and references as needed.

## ⚙️ Requirements

- Python **3.12 is recommended** for broad compatibility with the RAG and machine-learning packages.
- `uv` for dependency installation (or use pip if preferred).
- A Groq account and an API key for hosted LLM responses.
- Internet access for API requests and the initial model download.
- A text-based PDF. Scanned PDFs may need an OCR step before text retrieval will work.

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/abhishekmghayal/BookMind-Ai.git
cd BookMind-Ai
```

### 2. Create and activate a virtual environment

If Python 3.12 is installed:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

If the project already has a working `.venv`, activate that environment instead.

### 3. Install dependencies

If you use the explained requirements file:

```bash
uv pip install -r BookMind_AI_First_RAG_Explained_Requirements.txt
```

Or, if `requirements.txt` is your project's primary dependency file:

```bash
uv pip install -r requirements.txt
```

Use one authoritative dependency file for a project to avoid maintaining duplicate package lists that drift apart.

### 4. Configure the API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=replace_with_your_groq_api_key
```

Optionally set the model ID if your application reads `GROQ_MODEL`:

```dotenv
GROQ_MODEL=qwen/qwen3.8-27b
```

Replace the example with a model currently supported by your Groq account. Never put a real key in README, source code, screenshots, or Git commits.

### 5. Start the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### 6. Use the application

1. Select a PDF from the sidebar.
2. Click **Process book**.
3. Wait for text extraction, chunking, embedding, and indexing to finish.
4. Ask a question in the chat input.
5. Open the source expander to inspect retrieved passages and page numbers.

### Run the console version

If you want to use the command-line RAG program instead:

```bash
python main.py
```

Enter a question at the prompt. In the current console code, enter `0` to exit.

## 🔐 Security and Git Hygiene

Add a `.gitignore` file in the project root:

```gitignore
# Secrets
.env
.env.*

# Python environments and cache
.venv/
__pycache__/
*.py[cod]

# Local Chroma database
chroma_db/

# macOS
.DS_Store
```

Before pushing, inspect staged files:

```bash
git status
git diff --cached --stat
```

Do not commit API keys, virtual environments, local vector database files, or private documents. If a key is ever committed, revoke or rotate it even if the push is rejected by GitHub.

## 🧪 Troubleshooting

### Chroma says the embedding dimension does not match

Example:

```text
Collection expecting embedding with dimension of 384, got 768
```

Use the same embedding model that created the collection. This project uses `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors. If you deliberately change models, rebuild the collection by embedding the source documents again with the new model. Back up any data before deleting or rebuilding a collection.

### Groq API key or model errors

- Confirm `GROQ_API_KEY` is present in `.env`.
- Restart Streamlit after changing environment variables.
- Confirm the model ID is currently available to your Groq account.
- Do not print or share the secret key while debugging.

### PDF returns no useful text

The PDF may be scanned or image-based. `PyPDFLoader` extracts text but does not perform OCR. Add an OCR pipeline for scanned pages.

### Hugging Face authentication warning

Public embedding models often work without a Hugging Face token. Authentication may be useful for rate limits or gated models, but do not add a token unless needed.

### `compdef: command not found`

This message is associated with Zsh shell initialization and is separate from the RAG application's Python code.

## 🛣️ Future Improvements

- Add OCR support for scanned books.
- Improve bibliography/reference extraction with a dedicated workflow.
- Add document selection and book-library management.
- Add retrieval scores and more robust source citation handling.
- Add tests for ingestion, retrieval, and prompt behavior.
- Add a pinned lockfile after confirming a working dependency set.
- Add deployment instructions and automated checks.

## ⚠️ Limitations

- The quality of answers depends on PDF text quality, chunking, embeddings, retrieval, and the LLM.
- A context-only prompt is a useful guardrail, not a formal guarantee against hallucinations.
- Retrieved source passages are evidence for an answer, but page references should be checked against the original PDF for important work.
- Multi-Query retrieval may require additional model calls.
- Local Chroma data is not included in the repository; upload and index a PDF to create a local collection.

## 📄 License

No license has been specified yet. Add a `LICENSE` file before presenting this repository as open source. Also make sure you have permission to distribute any sample PDF included in the repository.

---

Built as a hands-on learning project to understand document ingestion, embeddings, vector databases, retrieval strategies, and grounded LLM responses.
'''
p = Path("/mnt/data/README_BookMind_AI.md")
p.write_text(readme, encoding="utf-8")
print(f"Created {p} ({len(readme.splitlines())} lines)")
