---
title: School Rag Agent
emoji: 🏫
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
python_version: 3.10
---

# 🏫 School Data Chat Agent

A conversational AI agent that lets you talk to your school Excel and PDF files in plain English — powered by **Azure OpenAI**, **LangChain**, **FAISS**, and **Gradio**.

---

## 📁 Project Structure

```
school_agent/
├── app.py                  # Main Gradio app (entry point)
├── requirements.txt
├── .gitignore              # Git ignore configuration
├── .env                    # Your Azure credentials (never commit this)
├── .env.example
├── data/                   # Uploaded files are stored here
├── vectorstore/            # FAISS index stored here (auto-created)
└── utils/
    ├── loader.py           # Excel + PDF ingestion & chunking
    ├── vectorstore.py      # FAISS build/load with Azure Embeddings
    └── rag_chain.py        # Conversational Hybrid Agent with tool-calling
```

---

## 💬 How It Works (Hybrid Architecture)

Instead of relying solely on similarity search, the agent uses a **Tool-Calling Agent Loop** to dynamically choose the best way to resolve your query:

```
                  User Question
                        │
                        ▼
            Does it query Excel/numbers
            or search descriptive text?
           /                           \
          /                             \
         ▼                               ▼
[query_excel_data Tool]         [search_documents Tool]
Executes Python/Pandas          Queries FAISS Vector Database
on DataFrames in memory          to search PDF/text documents
         │                               │
         ▼                               ▼
     DataFrame Output               Text Snippets
         \                               /
          \                             /
           ▼                           ▼
                Azure OpenAI (GPT-4o)
                + Conversation Memory
                         │
                         ▼
                Answer + Citations
```

1. **Python Pandas Execution (`query_excel_data`)**: Used for numerical or aggregate queries (e.g., *“how many students are in Class 10?”*, *“what is the average Maths score?”*, *“show the top 10 students by attendance”*). It writes and runs python code directly on the dataframes to return 100% accurate, non-hallucinated results.
2. **Vector Search (`search_documents`)**: Used for unstructured queries (e.g., school rules, policies, or specific student descriptive lookups in PDFs).
3. **Automatic Schema Discovery**: The agent scans all Excel files on startup, reads sheet layouts, shapes, and columns, and injects them directly into its system prompt so it knows exactly what fields exist.

---

## ⚙️ Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 3. Run locally
```bash
python app.py
# Open: http://localhost:7860
```

---

## 🔗 Sharing & Deployment

### 1. Easiest: Share a Live Temporary Link (Gradio Share)
If you want to share a live link with anyone to test the agent from your computer:
1. Open `app.py`.
2. Change `share=False` to `share=True` in the launch block at the bottom of the file.
3. Run `python app.py`.
4. Gradio will output a public URL (e.g., `https://xxxxxxxx.gradio.live`) which is shareable and stays active for 72 hours.

### 2. Permanent: Hugging Face Spaces (Free Cloud Hosting)
1. Create a new **Gradio** Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Upload the project files (ignoring files in `.gitignore`).
3. Add your Azure OpenAI credentials under Space Settings -> **Repository Secrets**.
4. The Space will build and deploy the app with a permanent public link.

---

## 📊 Supported File Types

| Format | Execution Method | Perfect For |
|--------|------------------|-------------|
| `.xlsx`, `.xls`, `.xlsm` | Pandas/Python Code Execution | Aggregate calculations, math, sorting, counting |
| `.pdf` | FAISS Vector Search | Descriptive text queries, school rules & policies |
