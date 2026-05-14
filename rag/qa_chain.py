"""
LangChain + Gemini 2.5 Pro orqali RAG savol-javob zanjiri.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from rag.config import LLM_MODEL, GOOGLE_API_KEY, TOP_K


SYSTEM_PROMPT = """Sen Python dasturlash tili bo'yicha o'zbek tilida javob beruvchi yordamchisan.
Sening vazifang — foydalanuvchining savoliga FAQAT berilgan kontekst (PDF qo'llanmadan olingan parchalar) asosida javob berish.

QOIDALAR:
1. Javob faqat o'zbek tilida bo'lsin.
2. Agar kontekstda javob bo'lmasa, "Bu savolga qo'llanmada javob topilmadi." deb yoz.
3. Javobni aniq, tushunarli va tartibli yoz (kerak bo'lsa ro'yxat yoki kod blokidan foydalan).
4. Kod misollarini ```python ... ``` formatida ber.
5. O'zingdan ma'lumot to'qima — faqat kontekstga tayan.
6. Javob oxirida qaysi sahifalardan foydalanganingni qavs ichida ko'rsat. Masalan: (Manba: 12, 15-sahifalar)
"""

USER_TEMPLATE = """KONTEKST (PDF qo'llanmadan):
{context}

SAVOL: {question}

Yuqoridagi kontekst asosida o'zbek tilida to'liq javob ber."""


def _format_docs(docs) -> str:
    """Retriever natijalarini matn formatiga o'giradi."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Parcha {i} | Sahifa {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(vector_store: FAISS):
    """
    LangChain RAG zanjirini quradi:
    Savol → Retriever → Kontekst + Savol → Gemini LLM → Javob
    """
    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    # RAG chain (LCEL)
    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
