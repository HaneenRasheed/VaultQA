import os
import re
import json
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

REWRITE_PROMPT = ChatPromptTemplate.from_template("""Given the chat history and a follow-up question, rewrite the follow-up as one standalone question that makes sense without the history. Replace pronouns like "it", "that" or "they" with what they refer to. If the question is already standalone, return it unchanged. Output ONLY the question, nothing else.

Chat history:
{history}

Follow-up question: {question}

Standalone question:""")


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


def _pdf_manifest():
    """Snapshot of the PDFs in data/: {filename: [size, modified_time]}."""
    out = {}
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.lower().endswith(".pdf"):
                s = os.stat(os.path.join(DATA_DIR, f))
                out[f] = [s.st_size, int(s.st_mtime)]
    return out


def _manifest_file(embed_model):
    return os.path.join(index_path(embed_model), "manifest.json")


def index_is_stale(embed_model):
    """True if the PDFs changed since this index was built."""
    if not index_exists(embed_model):
        return False
    try:
        with open(_manifest_file(embed_model)) as f:
            return json.load(f) != _pdf_manifest()
    except (FileNotFoundError, ValueError):
        return True   # old index with no manifest: rebuild once


def build_index(embed_model, chunk_size=800, chunk_overlap=150):
    manifest = _pdf_manifest()
    docs = PyPDFDirectoryLoader(DATA_DIR).load()
    if not docs:
        raise ValueError(f"No PDFs found in '{DATA_DIR}'.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(
        chunks, OllamaEmbeddings(model=embed_model), normalize_L2=True
    )
    db.save_local(index_path(embed_model))
    with open(_manifest_file(embed_model), "w") as f:
        json.dump(manifest, f)
    return len(docs), len(chunks)


def load_index(embed_model):
    return FAISS.load_local(
        index_path(embed_model),
        OllamaEmbeddings(model=embed_model),
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )

def rewrite_question(question, history, llm_model, max_turns=6):
    """Turn a follow-up into a standalone question using recent chat history."""
    if not history:
        return question
    recent = history[-max_turns:]
    text = "\n".join(f"{m['role'].title()}: {m['content'][:500]}" for m in recent)
    try:
        llm = ChatOllama(model=llm_model, temperature=0, num_ctx=4096)
        out = (REWRITE_PROMPT | llm).invoke(
            {"history": text, "question": question}
        ).content.strip()
    except Exception:
        return question                      # never block the answer
    out = out.strip('"').split("\n")[0].strip()
    if not out or len(out) > 3 * len(question) + 200:
        return question                      # model rambled, so fall back
    return out


# ---------- Question answering ----------
def ask(question, db, llm_model, k=3, temperature=0.0, num_ctx=8192, max_distance=1.0):
    """Return (token_stream, source_docs). Skips the LLM if nothing is relevant."""
    results = db.similarity_search_with_score(question, k=k)
    if not results:
        return iter(["I couldn't find that in the documents."]), []

    docs = []
    for d, score in results:
        d.metadata["score"] = float(score)      # shown in the Sources panel
        if score <= max_distance:
            docs.append(d)

    if not docs:
        best = min(score for _, score in results)
        msg = (
            "I couldn't find that in the documents.\n\n"
            f"*(Closest match distance: {best:.2f}, cutoff: {max_distance:.2f})*"
        )
        return iter([msg]), []

    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatOllama(model=llm_model, temperature=temperature, num_ctx=num_ctx)
    chain = PROMPT | llm
    stream = (c.content for c in chain.stream({"context": context, "question": question}))
    return stream, docs