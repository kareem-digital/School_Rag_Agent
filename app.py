"""
School Data Chat Agent
======================
Upload your school Excel and PDF files and chat with them in natural language.
Powered by Azure OpenAI + LangChain + FAISS + Gradio
"""

import os
import shutil
from dotenv import load_dotenv

# Monkeypatch gradio_client schema parsing to prevent Pydantic v2 "TypeError: argument of type 'bool' is not iterable"
try:
    import gradio_client.utils as client_utils
    original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    def safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "any" if schema else "None"
        return original_json_schema_to_python_type(schema, defs)
    client_utils._json_schema_to_python_type = safe_json_schema_to_python_type
    print("[INFO] Successfully applied Gradio API schema parser monkeypatch.")
except Exception as e:
    print(f"[WARN] Failed to apply Gradio API schema patch: {e}")

import gradio as gr

from utils.loader import ingest_files
from utils.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from utils.rag_chain import build_rag_chain

load_dotenv()

# ── Global state ──────────────────────────────────────────────────────────────
VECTORSTORE_PATH = "vectorstore/index"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

_chain = None


# ── Core functions ────────────────────────────────────────────────────────────


def index_files(files) -> str:
    global _chain

    if not files:
        return "⚠️ Please upload at least one Excel or PDF file."

    saved_paths = []
    for f in files:
        dest = os.path.join(DATA_DIR, os.path.basename(f.name))
        shutil.copy(f.name, dest)
        saved_paths.append(dest)

    missing = [
        k
        for k in [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        ]
        if not os.environ.get(k)
    ]
    if missing:
        return f"❌ Missing environment variables: {', '.join(missing)}"

    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)

        docs = ingest_files(saved_paths)
        if not docs:
            return "❌ No content could be extracted from the uploaded files."

        vs = build_vectorstore(docs, save_path=VECTORSTORE_PATH)
        _chain = build_rag_chain(vs)

        file_names = ", ".join(os.path.basename(p) for p in saved_paths)
        return (
            f"✅ **Indexed successfully!**\n\n"
            f"📁 Files: `{file_names}`\n"
            f"📄 Chunks created: `{len(docs)}`\n\n"
            f"You can now ask questions in the chat below."
        )

    except Exception as e:
        return f"❌ Error during indexing: {str(e)}"


def load_existing_index() -> str:
    global _chain
    if not vectorstore_exists(VECTORSTORE_PATH):
        return "⚠️ No existing index found. Please upload files first."
    try:
        vs = load_vectorstore(VECTORSTORE_PATH)
        _chain = build_rag_chain(vs)
        return "✅ Existing index loaded. Ready to chat!"
    except Exception as e:
        return f"❌ Failed to load index: {str(e)}"


def chat(message: str, history: list) -> tuple:
    global _chain

    if not message.strip():
        return "", history

    # Always use dict format for messages-type Chatbot
    if _chain is None:
        history = history + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": "⚠️ Please index your files first using the panel on the left.",
            },
        ]
        return "", history

    try:
        result = _chain.invoke(message)
        answer = result.get("answer", "")

        source_docs = result.get("source_documents", [])
        if source_docs:
            sources = set()
            for doc in source_docs:
                meta = doc.metadata
                src = meta.get("source", "unknown")
                if meta.get("type") == "excel":
                    sources.add(
                        f"📊 {src} (Sheet: {meta.get('sheet', '?')}, Row: {meta.get('row', '?')})"
                    )
                elif meta.get("type") == "pdf":
                    sources.add(f"📄 {src} (Page {meta.get('page', '?')})")
                else:
                    sources.add(f"📁 {src}")
            answer += "\n\n---\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]

    except Exception as e:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Error: {str(e)}"},
        ]

    return "", history


def clear_chat() -> tuple:
    global _chain
    if _chain and hasattr(_chain, "clear_memory"):
        _chain.clear_memory()
    return [], "🔄 Chat cleared. Memory reset."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
#status_box { min-height: 80px; }
"""

with gr.Blocks(title="School Data Chat Agent", theme=gr.themes.Soft(primary_hue="blue"), css=CSS) as demo:

    gr.Markdown("# 🏫 School Data Chat Agent", elem_id="title")
    gr.Markdown(
        "Upload your school **Excel** and **PDF** files and ask anything in plain English.",
        elem_id="subtitle",
    )

    with gr.Row():

        # ── Left panel ─────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📂 File Management")

            file_input = gr.File(
                label="Upload Excel / PDF files",
                file_types=[".xlsx", ".xls", ".xlsm", ".pdf"],
                file_count="multiple",
            )

            with gr.Row():
                index_btn = gr.Button("🔄 Index Files", variant="primary")
                load_btn = gr.Button("📥 Load Existing Index")

            status_box = gr.Markdown(
                value="Upload files and click **Index Files** to get started.",
                elem_id="status_box",
            )

            gr.Markdown("---")
            gr.Markdown("### ℹ️ Tips")
            gr.Markdown("""
- Ask questions in **plain English**
- Works with **multiple sheets** in Excel
- Supports **tables inside PDFs**
- Conversation **memory** is maintained
- Click **Clear Chat** to reset memory
""")

        # ── Right panel: Chat ───────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chatbot = gr.Chatbot(
                label="",
                height=480,
                show_label=False,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask anything about your school data...",
                    label="",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # ── Event wiring ──────────────────────────────────────────────────────
    index_btn.click(fn=index_files, inputs=[file_input], outputs=[status_box])
    load_btn.click(fn=load_existing_index, inputs=[], outputs=[status_box])

    send_btn.click(fn=chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
    msg_box.submit(fn=chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, status_box])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
