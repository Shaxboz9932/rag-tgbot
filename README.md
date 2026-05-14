# Python Qo'llanma RAG Bot

PDF qo'llanma (`Python.pdf`) asosida **RAG** (Retrieval Augmented Generation) tizimi.
**Telegram bot** (aiogram 3, long polling) + **CLI** rejimi.

## Texnologiyalar

| Komponent | Texnologiya |
|---|---|
| PDF o'qish / chunking | PyMuPDF + LangChain TextSplitter |
| Embedding | Google `gemini-embedding-001` (LangChain) |
| Vektor DB | FAISS (lokal, `faiss_index/`) |
| LLM | Google `gemini-2.5-pro` (LangChain) |
| Telegram Bot | aiogram 3 (long polling) |
| CLI | argparse + rich |

## O'rnatish

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## `.env` fayl

```env
GOOGLE_API_KEY=sizning_google_api_kalitingiz
BOT_TOKEN=telegram_bot_tokeningiz
ADMIN_IDS=123456789   # ixtiyoriy, vergul bilan
```

## Ishga tushirish

### Telegram Bot (production — long polling)
```bash
python bot_main.py
```

### CLI rejim
```bash
python main.py                      # Interaktiv
python main.py -q "Pythonda list nima?"  # Bir martalik
python main.py --reindex             # Qayta indekslash
```

## Loyiha tuzilishi

```
.
├── bot_main.py            # Telegram bot entry point
├── main.py                # CLI entry point
├── bot/
│   ├── config.py          # Bot sozlamalari (BOT_TOKEN, ADMIN_IDS)
│   ├── loader.py          # RAG tizimini yuklash (global state)
│   └── handlers.py        # /start, /help, /reindex, savol-javob
├── rag/
│   ├── config.py          # Umumiy sozlamalar
│   ├── pdf_loader.py      # PyMuPDF + LangChain chunking
│   ├── embeddings.py      # Gemini embedding (LangChain)
│   ├── vector_store.py    # FAISS (LangChain)
│   └── qa_chain.py        # LCEL RAG chain
├── Python.pdf             # Manba hujjat
├── faiss_index/           # Lokal vektor indeks (avto-yaratiladi)
├── requirements.txt
└── .env
```

## RAG ish jarayoni

1. **PDF -> chunklar**: PyMuPDF har sahifani o'qiydi, LangChain `RecursiveCharacterTextSplitter` 1000 belgilik chunklarga bo'ladi.
2. **Embedding**: `gemini-embedding-001` orqali vektorga aylantiriladi.
3. **Saqlash**: FAISS lokal faylga yoziladi.
4. **So'rov**: Savol embed qilinadi, top-5 yaqin chunk topiladi.
5. **Generatsiya**: Kontekst + savol `gemini-2.5-pro` ga yuboriladi, o'zbek tilida javob qaytadi.

## Telegram Bot buyruqlari

| Buyruq | Tavsif |
|---|---|
| `/start` | Botni boshlash |
| `/help` | Yordam |
| `/reindex` | Indeksni qayta qurish (admin) |
| _oddiy matn_ | Savolga javob berish |
