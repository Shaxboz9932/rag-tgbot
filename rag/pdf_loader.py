"""
PyMuPDF orqali PDF dan matn o'qish va LangChain Document formatida chunklash.
"""
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    PyMuPDF bilan PDF ni sahifama-sahifa o'qiydi,
    RecursiveCharacterTextSplitter bilan chunklaydi,
    LangChain Document ro'yxatini qaytaradi.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF fayl topilmadi: {pdf_path}")

    # 1) PyMuPDF bilan sahifalarni o'qiymiz
    page_docs: List[Document] = []
    doc = pymupdf.open(pdf_path)
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text and text.strip():
                page_docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                        },
                    )
                )
    finally:
        doc.close()

    # 2) LangChain text splitter bilan chunklash
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_docs)

    return chunks
