"""Loyiha sozlamalari."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Loyiha papkasi ──
BASE_DIR = Path(__file__).resolve().parent.parent

# ── PDF fayl yo'li ──
PDF_PATH = BASE_DIR / "Python.pdf"

# ── FAISS indeks saqlash yo'li ──
FAISS_INDEX_DIR = str(BASE_DIR / "faiss_index")

# ── Google API kalit ──
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY topilmadi! .env faylga GOOGLE_API_KEY=... qo'shing."
    )

# LangChain uchun env ga ham yozamiz
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Modellar ──
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-pro"

# ── Chunking sozlamalari ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Qidiruv sozlamalari ──
TOP_K = 5
