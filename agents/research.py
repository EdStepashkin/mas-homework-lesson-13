"""
Research Agent — виконує глибоке дослідження за планом.

Перевикористовує інструменти з hw5: web_search, read_url, knowledge_search.
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler

from config import settings, llm, get_prompt
from tools import web_search, read_url, knowledge_search


# Створюємо Research Agent
_research_agent = create_agent(
    model=llm,
    tools=[web_search, read_url, knowledge_search],
    system_prompt=get_prompt("researcher-agent"),
)


@tool
def research(plan_or_feedback: str) -> str:
    """
    Виконує глибоке дослідження за планом або уточнює інфу за фідбеком критика.
    Використовує інструменти web_search, read_url, та knowledge_search.
    Повертає рядок із текстовими знахідками (перелік фактів, описів).
    """
    langfuse_handler = CallbackHandler()
    result = _research_agent.invoke(
        {"messages": [{"role": "user", "content": f"Твоє завдання:\n{plan_or_feedback}"}]},
        config={"callbacks": [langfuse_handler]},
    )

    findings_text = ""
    out_msgs = result.get("messages", [])
    if out_msgs:
        content = out_msgs[-1].content
        if isinstance(content, list):
            findings_text = " ".join([str(c.get("text", "")) for c in content if isinstance(c, dict) and "text" in c])
        else:
            findings_text = str(content)
    else:
        findings_text = "Researcher не зміг виконати дослідження."

    return f"🔬 Знахідки Researcher (СИРІ ДАНІ, НЕ фінальний звіт):\n{findings_text}\n\n⚠️ УВАГА SUPERVISOR: Ці знахідки потрібно ОБОВ'ЯЗКОВО перевірити — виклич critique(findings) ЗАРАЗ."
