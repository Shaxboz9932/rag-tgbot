"""
Telegram Bot — RAG asosida Python qo'llanma bo'yicha savol-javob.

Ishga tushirish:
    python bot_main.py
"""
import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import BOT_TOKEN
from bot.handlers import router
from bot import loader


def setup_logging() -> None:
    """Logging sozlamasi."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Ortiqcha loglarni kamaytirish
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


async def on_startup(bot: Bot) -> None:
    """Bot ishga tushganda RAG tizimini yuklaydi."""
    logger = logging.getLogger(__name__)
    logger.info("RAG tizimi yuklanmoqda...")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, loader.init_rag)

    me = await bot.me()
    logger.info("Bot ishga tushdi: @%s", me.username)


async def main() -> None:
    setup_logging()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )

    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)

    logging.getLogger(__name__).info("Long polling ishga tushmoqda...")
    await dp.start_polling(
        bot,
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
