import os
from langchain_core.tools import tool
from ddgs import DDGS

# Імпортуємо наші налаштування
from config import settings


@tool
def web_search(query: str) -> str:
    """
    Виконує пошук в інтернеті за запитом.
    Повертає список знайдених сторінок із заголовком (title), посиланням (href) та коротким описом (body).
    Використовуй для пошуку трендів, фактів, статистики, fact-checking.
    """
    try:
        results = list(DDGS().text(query, max_results=settings.max_search_results))
        res_str = str(results)

        if len(res_str) > settings.max_url_content_length:
            return res_str[:settings.max_url_content_length] + "\n... [ТЕКСТ ОБРІЗАНО ЧЕРЕЗ ЛІМІТ]"

        return res_str
    except Exception as e:
        return f"Помилка пошуку: {str(e)}"


# Глобальний кеш для рітрівера, щоб не завантажувати модель та базу щоразу
_GLOBAL_RETRIEVER = None

def _get_cached_retriever():
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        from retriever import get_retriever
        _GLOBAL_RETRIEVER = get_retriever()
    return _GLOBAL_RETRIEVER

@tool
def knowledge_search(query: str) -> str:
    """
    Search the local knowledge base (style guide, brand description, example posts).
    Use for questions about brand voice, tone, audience, content examples.
    """
    try:
        retriever = _get_cached_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return "Не знайдено релевантної інформації у базі знань."

        result = f"[{len(docs)} documents found]\n"
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if '/' in source or '\\' in source else source
            result += f"- [{filename}] {doc.page_content}\n\n"

        if len(result) > settings.max_url_content_length:
            return result[:settings.max_url_content_length] + "\n... [ТЕКСТ ОБРІЗАНО ЧЕРЕЗ ЛІМІТ]"

        return result
    except Exception as e:
        return f"Помилка пошуку в базі знань: {str(e)}"


@tool
def save_content(filename: str, content: str) -> str:
    """
    Зберігає фінальний затверджений контент як .md-файл у директорії example_output/.
    Приймає назву файлу (наприклад, ai_productivity_post.md) та повний текст контенту.
    """
    try:
        os.makedirs(settings.output_dir, exist_ok=True)
        filepath = os.path.join(settings.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ Контент успішно збережено: {filepath}"
    except Exception as e:
        return f"Помилка збереження файлу: {str(e)}"