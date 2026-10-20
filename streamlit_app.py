import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

APP_TITLE = "BookMind AI"
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp { background:#0b1020; color:#eef2ff; }
[data-testid="stSidebar"] { background:#11182b; }
.hero { padding:1.5rem 1.7rem; border:1px solid #293653; border-radius:18px;
background:linear-gradient(120deg,#172440,#11182b); margin-bottom:1rem; }
.hero h1 { margin:0; color:#f8fafc; font-size:2.1rem; }
.hero p { margin:.45rem 0 0; color:#b7c4df; }
div[data-testid="stChatMessage"] { border:1px solid #25324c; border-radius:14px; }
.stButton > button { border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def collection_for(pdf_bytes):
    return "book_" + hashlib.sha256(pdf_bytes).hexdigest()[:16]


def process_pdf(pdf_bytes, filename):
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    collection_name = collection_for(pdf_bytes)
    embeddings = get_embeddings()

    # Reuse this PDF's collection if it was indexed before.
    db = Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    if db._collection.count() > 0:
        return db, db._collection.count(), True

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp.write(pdf_bytes)
        temp_path = temp.name
    try:
        pages = PyPDFLoader(temp_path).load()
        if not pages:
            raise ValueError("No readable text was found in this PDF.")
        for page in pages:
            page.metadata["source_file"] = filename

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(pages)
        if not chunks:
            raise ValueError("No text chunks were created. A scanned PDF may need OCR.")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=DB_DIR,
        )
        return db, len(chunks), False
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=GROQ_MODEL, temperature=0)


def format_context(docs):
    sections = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "unknown"
        source = doc.metadata.get("source_file", doc.metadata.get("source", "PDF"))
        sections.append(f"[Source: {source} | Page: {page_number}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are BookMind, a helpful book-question-answering assistant.
Use ONLY the provided context. Do not invent facts or references.
If the answer is not supported by the context, say:
"I could not find the answer in the document."
When possible, mention the page numbers supporting your answer."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "active_book" not in st.session_state:
    st.session_state.active_book = None
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.markdown("## 📚 BookMind AI")
    st.caption("Your personal PDF research assistant")
    st.divider()
    st.markdown("### Upload a book")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded:
        st.caption(f"Selected: **{uploaded.name}**")
        if st.button("⚡ Process book", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading PDF and creating embeddings..."):
                    pdf_bytes = uploaded.getvalue()
                    db, count, reused = process_pdf(pdf_bytes, uploaded.name)
                    collection = collection_for(pdf_bytes)
                    if st.session_state.active_collection != collection:
                        st.session_state.messages = []
                    st.session_state.vectorstore = db
                    st.session_state.active_book = uploaded.name
                    st.session_state.active_collection = collection
                    st.session_state.chunk_count = count
                if reused:
                    st.success(f"Loaded existing index ({count} chunks).")
                else:
                    st.success(f"Book processed ({count} chunks).")
            except Exception as exc:
                st.error(f"Could not process PDF: {exc}")

    st.divider()
    st.markdown("### Retrieval")
    st.write("**MMR:** enabled")
    st.write("**Multi-Query:** enabled")
    st.write("**Embeddings:** MiniLM-L6-v2 (384 dimensions)")
    st.write(f"**Chat model:** `{GROQ_MODEL}`")
    st.caption("PDF embeddings are stored locally in chroma_db.")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="hero">
<h1>📖 BookMind AI</h1>
<p>Upload a book. Ask questions. Explore answers with source references.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.active_book:
    c1, c2 = st.columns([3, 1])
    c1.success(f"Active book: {st.session_state.active_book}")
    c2.metric("Indexed chunks", st.session_state.chunk_count)
else:
    st.info("Upload a PDF in the sidebar and click **Process book** to get started.")

st.markdown("### Chat with your book")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 View retrieved sources"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}: {source['label']}**")
                    st.write(source["text"])

question = st.chat_input(
    "Ask a question about your book...",
    disabled=st.session_state.vectorstore is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = get_llm()
            base_retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, llm=llm
            )
            with st.spinner("Searching the book..."):
                docs = retriever.invoke(question)
                unique_docs, seen = [], set()
                for doc in docs:
                    key = (doc.metadata.get("source_file"), doc.metadata.get("page"), doc.page_content)
                    if key not in seen:
                        seen.add(key)
                        unique_docs.append(doc)
                context = format_context(unique_docs)
                answer = llm.invoke(prompt.invoke({
                    "context": context, "question": question
                })).content

            st.markdown(answer)
            sources = []
            for doc in unique_docs:
                page = doc.metadata.get("page")
                page_number = page + 1 if isinstance(page, int) else "unknown"
                filename = doc.metadata.get("source_file", doc.metadata.get("source", "PDF"))
                sources.append({
                    "label": f"{filename} — page {page_number}",
                    "text": doc.page_content,
                })
            if sources:
                with st.expander("📎 View retrieved sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}: {source['label']}**")
                        st.write(source["text"])
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })
        except Exception as exc:
            message = f"Request failed: {exc}\n\nCheck your Groq API key, model ID, and whether the PDF was processed."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})

st.caption("BookMind AI • Streamlit + LangChain + ChromaDB + Groq")
