"""
RAG CLI - Python o'zbekcha qo'llanma bo'yicha savol-javob.

Foydalanish:
    python main.py            # Interaktiv rejim
    python main.py --reindex  # Indeksni qaytadan qurish
    python main.py -q "savol" # Bir martalik savol
"""
import sys
import os
import argparse

os.environ["PYTHONIOENCODING"] = "utf-8"

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

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


console = Console(force_terminal=True)


def ensure_index(force=False):
    embeddings = get_embeddings()

    if force:
        console.print("[yellow]Eski indeks o'chirilmoqda...[/yellow]")
        reset_index()

    if index_exists() and not force:
        console.print("[green]Indeks allaqachon mavjud. Yuklanmoqda...[/green]")
        vector_store = load_vector_store(embeddings)
        rag_chain, retriever = build_rag_chain(vector_store)
        return rag_chain, retriever

    console.print(f"[cyan]PDF o'qilmoqda:[/cyan] {PDF_PATH}")
    chunks = load_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    console.print(f"[cyan]Jami {len(chunks)} ta chunk hosil qilindi.[/cyan]")

    console.print("[bold green]Embeddinglar yaratilmoqda va FAISS ga saqlanmoqda...[/bold green]")
    vector_store = build_vector_store(chunks, embeddings)
    console.print("[bold green]Indeks tayyor![/bold green]")

    rag_chain, retriever = build_rag_chain(vector_store)
    return rag_chain, retriever


def interactive_loop(rag_chain, retriever):
    console.print(
        Panel.fit(
            "[bold cyan]Python o'zbekcha qo'llanma - RAG CLI[/bold cyan]\n"
            + "Savolingizni yozing. Chiqish uchun: exit yoki quit",
            border_style="cyan",
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold yellow]Savol[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Chiqildi.[/red]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "chiqish", "q"}:
            console.print("[green]Xayr![/green]")
            break

        try:
            console.print("[dim]Javob tayyorlanmoqda...[/dim]")
            answer = rag_chain.invoke(question)
            sources = retriever.invoke(question)

            console.print()
            console.print(
                Panel(
                    Markdown(answer),
                    title="[bold green]Javob[/bold green]",
                    border_style="green",
                )
            )

            pages = sorted({doc.metadata.get("page", "?") for doc in sources})
            console.print(f"[dim]Foydalanilgan sahifalar: {pages}[/dim]")

        except Exception as e:
            console.print(f"[bold red]Xatolik:[/bold red] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF qo'llanma bo'yicha RAG asosida savol-javob CLI."
    )
    parser.add_argument("--reindex", action="store_true", help="Indeksni qayta qurish.")
    parser.add_argument("--question", "-q", type=str, default=None, help="Bir martalik savol.")
    args = parser.parse_args()

    rag_chain, retriever = ensure_index(force=args.reindex)

    if args.question:
        answer = rag_chain.invoke(args.question)
        console.print(Panel(Markdown(answer), title="Javob", border_style="green"))
        return 0

    interactive_loop(rag_chain, retriever)
    return 0


if __name__ == "__main__":
    sys.exit(main())
