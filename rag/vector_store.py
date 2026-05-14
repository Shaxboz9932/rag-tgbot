"""
LangChain-FAISS orqali vektor storage.
FAISS lokal faylga saqlaydi, ChromaDB ga bog'liqlik yo'q.
"""
import os
import shutil
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.config import FAISS_INDEX_DIR


def build_vector_store(
    docs: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    """
    Hujjatlarni embedding qilib FAISS indeksini yaratadi va diskka saqlaydi.
    """
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def load_vector_store(
    embeddings: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    """
    Oldindan saqlangan FAISS indeksini diskdan yuklaydi.
    """
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def index_exists() -> bool:
    """FAISS indeks fayllari mavjudligini tekshiradi."""
    faiss_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    pkl_file = os.path.join(FAISS_INDEX_DIR, "index.pkl")
    return os.path.exists(faiss_file) and os.path.exists(pkl_file)


def reset_index() -> None:
    """FAISS indeks papkasini o'chiradi (qayta indekslash uchun)."""
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
