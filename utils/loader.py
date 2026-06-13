"""
Document Loader — handles Excel (.xlsx) and PDF ingestion into text chunks.
"""

import os
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_excel(file_path: str) -> list[Document]:
    """Load all sheets from an Excel file into LangChain Documents."""
    docs = []
    xl = pd.read_excel(file_path, sheet_name=None)  # all sheets

    for sheet_name, df in xl.items():
        df = df.fillna("").astype(str)
        # Convert each row to a readable key:value string
        for i, row in df.iterrows():
            row_text = f"[Sheet: {sheet_name} | Row {i + 1}]\n"
            row_text += "\n".join(
                f"  {col}: {val}" for col, val in row.items() if val.strip()
            )
            if row_text.strip():
                docs.append(
                    Document(
                        page_content=row_text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "sheet": sheet_name,
                            "row": i + 1,
                            "type": "excel",
                        },
                    )
                )

    # Also add a sheet-level summary with column headers
    for sheet_name, df in xl.items():
        header_doc = (
            f"[Sheet: {sheet_name}] — Columns: {', '.join(df.columns.tolist())}\n"
            f"Total rows: {len(df)}"
        )
        docs.append(
            Document(
                page_content=header_doc,
                metadata={
                    "source": os.path.basename(file_path),
                    "sheet": sheet_name,
                    "type": "excel_schema",
                },
            )
        )

    return docs


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file page-by-page into LangChain Documents."""
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Also try to extract tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_text = "\n".join(
                        " | ".join(str(cell or "") for cell in row)
                        for row in table
                    )
                    text += f"\n\n[TABLE on page {i+1}]\n{table_text}"

            if text.strip():
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": os.path.basename(file_path),
                            "page": i + 1,
                            "type": "pdf",
                        },
                    )
                )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_files(file_paths: list[str]) -> list[Document]:
    """Main entry: load and chunk all provided files."""
    all_docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls", ".xlsm"):
            raw = load_excel(path)
        elif ext == ".pdf":
            raw = load_pdf(path)
        else:
            print(f"[WARN] Unsupported file type: {path}")
            continue
        all_docs.extend(raw)
        print(f"[INFO] Loaded {len(raw)} chunks from: {os.path.basename(path)}")

    chunks = split_documents(all_docs)
    print(f"[INFO] Total chunks after splitting: {len(chunks)}")
    return chunks
