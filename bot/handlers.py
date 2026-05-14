"""
Aiogram 3 handler'lar — /start, /help, /reindex va savol-javob.
"""
import asyncio
import logging

from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.enums import ParseMode

from bot.config import ADMIN_IDS
from bot import loader

logger = logging.getLogger(__name__)

router = Router(name="main_router")

# ── Telegram Markdown V2 uchun maxsus belgilarni escape qilish ──
_ESCAPE_CHARS = r"_[]()~`>#+-=|{}.!"


def _escape_md(text: str) -> str:
    """MarkdownV2 uchun maxsus belgilarni escape qiladi."""
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        # Kod bloklarini saqlab qolish: ```...```
        if text[i:i+3] == "```":
            end = text.find("```", i + 3)
            if end != -1:
                result.append(text[i:end+3])
                i = end + 3
                continue
        # Inline kod: `...`
        if ch == "`":
            end = text.find("`", i + 1)
            if end != -1:
                result.append(text[i:end+1])
                i = end + 1
                continue
        # Bold: **text**
        if text[i:i+2] == "**":
            end = text.find("**", i + 2)
            if end != -1:
                inner = _escape_md(text[i+2:end])
                result.append(f"*{inner}*")
                i = end + 2
                continue
        # Oddiy belgilarni escape
        if ch in _ESCAPE_CHARS:
            result.append(f"\\{ch}")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


# ── /start ──
@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Assalomu alaykum! Men *Python qo'llanma* bo'yicha savol-javob botiman.\n\n"
        "Menga istalgan savolingizni yuboring — qo'llanma asosida javob beraman.\n\n"
        "Buyruqlar:\n"
        "/help — Yordam\n"
        "/reindex — Indeksni qayta qurish (faqat admin)",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── /help ──
@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Menga Python haqida savol yuboring.\n\n"
        "Masalan:\n"
        "- Pythonda list nima?\n"
        "- Dictionary va set farqi?\n"
        "- For loop qanday ishlaydi?\n\n"
        "Bot qo'llanma asosida javob beradi.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── /reindex (faqat admin) ──
@router.message(Command("reindex"))
async def cmd_reindex(message: Message) -> None:
    if ADMIN_IDS and message.from_user.id not in ADMIN_IDS:
        await message.answer("Bu buyruq faqat admin uchun.")
        return

    await message.answer("Indeks qayta qurilmoqda... Biroz kuting.")

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: loader.init_rag(force_reindex=True))
        await message.answer("Indeks muvaffaqiyatli qayta qurildi!")
    except Exception as e:
        logger.exception("Reindex xatolik")
        await message.answer(f"Xatolik yuz berdi: {e}")


# ── Savol-javob (oddiy matnli xabar) ──
@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = message.text.strip()
    if not question:
        return

    chain = loader.get_rag_chain()
    if chain is None:
        await message.answer("RAG tizimi hali yuklanmagan. Biroz kuting...")
        return

    # Foydalanuvchiga kuttiruvchi xabar yuboramiz
    loading_msg = await message.answer("⏳ Javob tayyorlanmoqda, biroz kuting...")
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    loop = asyncio.get_running_loop()
    try:
        # LangChain sinxron — threadpool da ishlatamiz
        answer = await loop.run_in_executor(None, chain.invoke, question)

        # Retriever dan sahifa raqamlarini olish
        ret = loader.get_retriever()
        if ret:
            sources = await loop.run_in_executor(None, ret.invoke, question)
            pages = sorted({doc.metadata.get("page", "?") for doc in sources})
            answer += f"\n\n_(Manba: {pages}-sahifalar)_"

    except Exception as e:
        logger.exception("RAG xatolik: %s", e)
        answer = f"Xatolik yuz berdi: {e}"

    # Kuttiruvchi xabarni o'chiramiz va asl javobni yuboramiz
    await loading_msg.delete()
    
    # Javobni yuborish (uzun bo'lsa bo'laklarga bo'lib)
    await _send_long_message(message, answer)


async def _send_long_message(message: Message, text: str, chunk_size: int = 4000) -> None:
    """
    Telegram 4096 belgi limitiga mos ravishda xabarni bo'laklarga bo'lib yuboradi.
    """
    if len(text) <= chunk_size:
        try:
            await message.answer(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # Markdown parse xatolik bo'lsa oddiy text yuboramiz
            await message.answer(text)
        return

    # Uzun javobni bo'laklarga bo'lish
    parts = []
    while text:
        if len(text) <= chunk_size:
            parts.append(text)
            break
        # Eng yaqin yangi qatordan kesish
        cut = text.rfind("\n", 0, chunk_size)
        if cut == -1:
            cut = chunk_size
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")

    for part in parts:
        try:
            await message.answer(part, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await message.answer(part)
