"""
VectorStore — builds and loads a FAISS index using Azure OpenAI Embeddings.
"""

import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


import time


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        chunk_size=16,
        max_retries=10,
    )


def build_vectorstore(docs: list[Document], save_path: str = "vectorstore/index") -> FAISS:
    """Embed documents in batches with a delay to avoid rate limits, and save FAISS index to disk."""
    embeddings = get_embeddings()
    
    batch_size = 50
    print(f"[INFO] Starting indexing of {len(docs)} documents in batches of {batch_size}...")
    
    # Initialize FAISS with the first batch
    first_batch = docs[:batch_size]
    vs = FAISS.from_documents(first_batch, embeddings)
    
    # Process remaining batches
    for i in range(batch_size, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"[INFO] Embedding batch {i // batch_size + 1} / {((len(docs) - 1) // batch_size) + 1}...")
        vs.add_documents(batch)
        time.sleep(2)  # Sleep 2 seconds between batches to avoid Azure 429 rate limit
        
    vs.save_local(save_path)
    print(f"[INFO] VectorStore saved to: {save_path}")
    return vs


def load_vectorstore(save_path: str = "vectorstore/index") -> FAISS:
    """Load existing FAISS index from disk."""
    embeddings = get_embeddings()
    vs = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[INFO] VectorStore loaded from: {save_path}")
    return vs


def vectorstore_exists(save_path: str = "vectorstore/index") -> bool:
    return os.path.exists(os.path.join(save_path, "index.faiss"))
