import os
import time
import streamlit as st
from rag import (
    DATA_DIR, list_ollama_models, split_models,
    index_exists, build_index, load_index, ask,
)

st.set_page_config(page_title="VaultQA", page_icon="", layout="wide")
st.title("VaultQA")


@st.cache_resource(show_spinner="Loading index...")
def get_db(embed_model):
    return load_index(embed_model)


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("⚙️ Settings")

    try:
        installed = list_ollama_models()
    except Exception:
        st.error("Can't reach Ollama. Start the Ollama app and refresh.")
        st.stop()

    chat_models, embed_models = split_models(installed)
    show_all = st.checkbox("Show all models in embedding list", value=False)
    if show_all or not embed_models:
        embed_models = installed
    if not chat_models:
        chat_models = installed

    llm_model = st.selectbox("Chat model", chat_models)
    embed_model = st.selectbox("Embedding model", embed_models)

    k = st.slider("Chunks to retrieve (k)", 1, 8, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    num_ctx = st.select_slider("Context window", [2048, 4096, 8192, 16384], value=8192)

    st.divider()
    st.subheader("📄 Documents")

    # this key resets the uploader after each save
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploads = st.file_uploader(
        "Add PDFs", type="pdf", accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploads:
            with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.session_state.uploader_key += 1
        st.session_state.upload_msg = (
            f"Saved {len(uploads)} file(s). Rebuild the index to include them."
        )
        st.rerun()

    if "upload_msg" in st.session_state:
        st.success(st.session_state.pop("upload_msg"))

    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    st.caption(f"{len(pdf_files)} PDF(s) in `{DATA_DIR}/`")

    with st.expander("Manage files"):
        if not pdf_files:
            st.caption("No PDFs yet.")
        for f in pdf_files:
            c1, c2 = st.columns([4, 1])
            c1.caption(f)
            if c2.button("🗑️", key=f"del_{f}"):
                os.remove(os.path.join(DATA_DIR, f))
                st.rerun()

    has_index = index_exists(embed_model)
    st.caption("Index: Ready ✅" if has_index else "Index: ❌ not built for this embedding model")

    if st.button("🔄 Build / rebuild index", use_container_width=True):
        try:
            with st.spinner(f"Embedding with {embed_model}..."):
                pages, chunks = build_index(embed_model)
            get_db.clear()
            st.success(f"Indexed {pages} pages into {chunks} chunks.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Main: chat ----------
if not index_exists(embed_model):
    st.info(f"No index for **{embed_model}** yet. Click **Build / rebuild index** in the sidebar.")
    st.stop()

db = get_db(embed_model)
st.caption(f"Chat: `{llm_model}` · Embeddings: `{embed_model}` · k={k}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            t0 = time.time()
            stream, sources = ask(question, db, llm_model, k, temperature, num_ctx)
            answer = st.write_stream(stream)
            elapsed = time.time() - t0
            st.caption(f"⏱ {elapsed:.1f}s · {llm_model}")
            with st.expander("Sources"):
                for s in sources:
                    page = s.metadata.get("page", 0) + 1
                    st.caption(f"{s.metadata.get('source')} (page {page})")
                    st.text(s.page_content[:300] + "...")
        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})