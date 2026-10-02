import os
import re
import requests
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

OLLAMA_URL = "http://localhost:11434"
DATA_DIR = "data"
INDEX_ROOT = "faiss_index"
EMBED_HINTS = ("embed", "bge", "minilm", "mxbai", "arctic", "e5", "gte")

PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:""")


# ---------- Model discovery ----------
def list_ollama_models():
    """Return names of all models installed in Ollama."""
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))


def split_models(names):
    """Guess which installed models are embedding models by name."""
    embed = [n for n in names if any(h in n.lower() for h in EMBED_HINTS)]
    chat = [n for n in names if n not in embed]
    return chat, embed


# ---------- Index management (one index per embedding model) ----------
def index_path(embed_model):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", embed_model)
    return os.path.join(INDEX_ROOT, safe)


def index_exists(embed_model):
    return os.path.exists(os.path.join(index_path(embed_model), "index.faiss"))


def build_index(embed_model, chunk_size=800, chunk_overlap=150):
    docs = PyPDFDirectoryLoader(DATA_DIR).load()
    if not docs:
        raise ValueError(f"No PDFs found in '{DATA_DIR}'.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, OllamaEmbeddings(model=embed_model))
    db.save_local(index_path(embed_model))
    return len(docs), len(chunks)


def load_index(embed_model):
    return FAISS.load_local(
        index_path(embed_model),
        OllamaEmbeddings(model=embed_model),
        allow_dangerous_deserialization=True,
    )


# ---------- Question answering ----------
def ask(question, db, llm_model, k=3, temperature=0.0, num_ctx=8192):
    """Return (token_stream, source_docs)."""
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatOllama(model=llm_model, temperature=temperature, num_ctx=num_ctx)
    chain = PROMPT | llm
    stream = (c.content for c in chain.stream({"context": context, "question": question}))
    return stream, docs