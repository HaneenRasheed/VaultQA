# VaultQA

**Chat with your documents, 100% locally.**

VaultQA is a Retrieval-Augmented Generation (RAG) app that answers questions about your PDFs using local models. Nothing leaves your machine: embeddings, vector search, and generation all run through [Ollama](https://ollama.com). Answers are grounded in the retrieved text and shown with source citations.

## Features

- **Grounded answers** based only on retrieved document chunks, with a fallback when the answer isn't in the documents
- **Source citations** showing file name, page number, and the matching text
- **Choose models in the UI**: pick any installed Ollama chat model and embedding model from dropdowns
- **One index per embedding model**, built on demand from the sidebar, so switching embedders never breaks retrieval
- **Streaming responses** for a faster-feeling experience with small local models
- **Upload PDFs from the browser** and rebuild the index with one click
- **Tunable settings**: number of retrieved chunks (k), temperature, and context window
- **Fully offline** after the models are downloaded

## How it works

```
PDFs → Chunking → Embeddings → FAISS index
                                    │
User question → Query embedding → Similarity search → Top-k chunks
                                                          │
                              Prompt (context + question) → LLM → Answer + sources
```

1. **Load**: PDFs in `data/` are read with `PyPDFDirectoryLoader`.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits text into 800-character chunks with 150 characters of overlap.
3. **Embed**: each chunk is embedded with the selected Ollama embedding model.
4. **Store**: vectors are saved in a FAISS index at `faiss_index/<embedding-model>/`.
5. **Retrieve**: the question is embedded and the top-k most similar chunks are fetched.
6. **Generate**: the chunks and question are sent to the selected chat model with a strict "answer only from context" prompt.
7. **Respond**: the answer streams into the UI along with its sources.
8. **Chat memory**: follow-up questions are rewritten into standalone queries before retrieval.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.10 to 3.12 |
| Orchestration | LangChain |
| LLM and embeddings | Ollama (for example `phi4-mini`, `nomic-embed-text`) |
| Vector store | FAISS (CPU) |
| PDF parsing | pypdf |
| UI | Streamlit |

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.10 to 3.12
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone and install

```bash
git clone https://github.com/<HaneenRasheed>/VaultQA.git
cd vaultQA

python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Pull the models

```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

Any chat model works for generation. For the embedding model, use a dedicated one such as `nomic-embed-text`, `mxbai-embed-large`, or `bge-m3`.

### 3. Add your documents

Put PDFs in the `data/` folder, or upload them later from the app sidebar. Use PDFs with selectable text. Scanned image-only PDFs are not supported yet.

### 4. Run the app

```bash
streamlit run app.py
```

Then in the sidebar:

1. Choose a **chat model** and an **embedding model**.
2. Click **Build / rebuild index** (needed once per embedding model, and again whenever you add documents).
3. Ask questions in the chat box.

## Project structure

```
documind/
├── app.py              # Streamlit UI
├── rag.py              # Chunking, embeddings, FAISS, retrieval, generation
├── requirements.txt
├── data/               # Your PDFs (not committed)
├── faiss_index/        # Generated indexes, one folder per embedding model
└── README.md
```

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| Chat model | Sidebar | first installed | Any Ollama chat model |
| Embedding model | Sidebar | first embedding-like model | Each gets its own index |
| Chunks retrieved (k) | Sidebar | 3 | Lower is better for small models |
| Temperature | Sidebar | 0.0 | Keep low for factual answers |
| Context window | Sidebar | 8192 | Prevents silent truncation of context |
| Chunk size / overlap | `rag.py` (`build_index`) | 800 / 150 | Rebuild the index after changing |
| Ollama URL | `rag.py` (`OLLAMA_URL`) | `http://localhost:11434` | Change for a remote Ollama |

## Design decisions

- **One index per embedding model.** Different embedding models produce vectors with different dimensions and meaning, so an index can only be searched with the model that built it.
- **Strict grounding prompt.** The model is told to answer only from the context and to say so when the answer isn't there, which reduces hallucination in small models.
- **Small `k` and explicit context window.** Small local models degrade with too much context, and Ollama's default window can silently truncate retrieved chunks.
- **FAISS for simplicity.** It's fast, needs no server, and is easy to ship. Chroma would add metadata filtering and built-in persistence.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't reach Ollama` in the sidebar | Start the Ollama app or run `ollama serve` |
| `No index for <model> yet` | Click **Build / rebuild index** in the sidebar |
| `No PDFs found` | Add `.pdf` files to `data/` and rebuild |
| Pylance `Import could not be resolved` | In VS Code, run **Python: Select Interpreter** and choose the `venv` one |
| Embedding model missing from the dropdown | Tick **Show all models in embedding list** |
| Slow answers | Lower k, use a smaller model, or confirm Ollama is using your GPU |
| Weak answers | Check the Sources panel first. If the right passage isn't retrieved, adjust chunk size, k, or the embedding model |

## Limitations

- PDF only, and no OCR for scanned documents
- Answer quality depends on the local model size and hardware
- Rebuilding re-embeds all documents (no incremental indexing yet)

## Privacy

All processing happens locally. Documents, embeddings, and questions are never sent to external services. Do not commit private documents or their indexes to a public repository (see `.gitignore`).

## Suggested `.gitignore`

```
venv/
__pycache__/
data/
faiss_index/
.env
```

## License

MIT. See `LICENSE` for details.

## Acknowledgements

[LangChain](https://www.langchain.com/) · [Ollama](https://ollama.com) · [FAISS](https://github.com/facebookresearch/faiss) · [Streamlit](https://streamlit.io)
