"""
Writer Agent — пише статтю/пост за контент-планом.

Інструменти: DuckDuckGo Search (факти, статистика), save_content (збереження .md).
Structured output: DraftContent.
"""

from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

from config import llm, get_prompt
from schemas import DraftContent
from tools import web_search, save_content


# Створюємо Writer Agent з structured output
_writer_agent = create_agent(
    model=llm,
    tools=[web_search, save_content],
    system_prompt=get_prompt("content-writer"),
    response_format=DraftContent,
)


def run_writer(plan_and_context: str, callbacks: list = None) -> DraftContent:
    """
    Запускає Writer Agent для написання контенту за планом.
    
    Args:
        plan_and_context: Затверджений ContentPlan + можливий feedback від Editor.
        callbacks: Langfuse callback handlers для tracing.
    
    Returns:
        DraftContent structured output.
    """
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    result = _writer_agent.invoke(
        {"messages": [{"role": "user", "content": plan_and_context}]},
        config=config,
    )

    structured = result.get("structured_response")
    if structured and isinstance(structured, DraftContent):
        return structured

    # Fallback: витягуємо текст з повідомлень
    out_msgs = result.get("messages", [])
    if out_msgs:
        content = out_msgs[-1].content
        if isinstance(content, list):
            text = " ".join([str(c.get("text", "")) for c in content if isinstance(c, dict) and "text" in c])
        else:
            text = str(content)
    else:
        text = "Writer не зміг створити контент."

    return DraftContent(
        content=text,
        word_count=len(text.split()),
        keywords_used=[],
    )
