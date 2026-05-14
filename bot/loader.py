"""
RAG tizimini bir marta yuklaydi va global o'zgaruvchilarda saqlaydi.
Bot ishga tushganda chaqiriladi (startup hook).
"""
import logging

from rag.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from rag.pdf_loader import load_and_chunk_pdf
from rag.embeddings import get_embeddings
from rag.vector_store import (
    build_vector_store,
    load_vector_store,
    index_exists,
    reset_index,
)
from rag.qa_chain import build_rag_chain

logger = logging.getLogger(__name__)

# Global RAG obyektlari — bot ishlayotgan davomida xotirada turadi
rag_chain = None
retriever = None


def init_rag(force_reindex: bool = False) -> None:
    """
    RAG tizimini ishga tushiradi:
      - FAISS indeksni yuklaydi yoki qaytadan quradi
      - rag_chain va retriever ni global ga yozadi
    """
    global rag_chain, retriever

    embeddings = get_embeddings()

    if force_reindex:
        logger.info("Eski indeks o'chirilmoqda...")
        reset_index()

    if index_exists() and not force_reindex:
        logger.info("FAISS indeks mavjud — yuklanmoqda...")
        vector_store = load_vector_store(embeddings)
    else:
        logger.info("PDF o'qilmoqda: %s", PDF_PATH)
        chunks = load_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info("%d ta chunk hosil qilindi. Embedding yaratilmoqda...", len(chunks))
        vector_store = build_vector_store(chunks, embeddings)
        logger.info("FAISS indeks tayyor va saqlandi.")

    rag_chain, retriever = build_rag_chain(vector_store)
    logger.info("RAG chain tayyor — bot savollarni qabul qilishga tayyor!")


def get_rag_chain():
    """Joriy rag_chain ni qaytaradi."""
    return rag_chain


def get_retriever():
    """Joriy retriever ni qaytaradi."""
    return retriever
