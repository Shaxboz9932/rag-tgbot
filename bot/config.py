"""Telegram bot sozlamalari."""
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN topilmadi! .env faylga BOT_TOKEN=... qo'shing.")

# Admin ID (ixtiyoriy) — faqat admin /reindex qila oladi
ADMIN_IDS: list[int] = []
_admin_raw = os.getenv("ADMIN_IDS", "")
if _admin_raw.strip():
    ADMIN_IDS = [int(x.strip()) for x in _admin_raw.split(",") if x.strip().isdigit()]
