"""
Content Strategist Agent — отримує бриф, досліджує тему, створює контент-план.

Інструменти: DuckDuckGo Search (тренди, конкуренти), RAG (style guide та приклади контенту).
Structured output: ContentPlan.
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler

from config import llm, get_prompt
from schemas import ContentPlan
from tools import web_search, knowledge_search


# Створюємо Content Strategist Agent з structured output
_strategist_agent = create_agent(
    model=llm,
    tools=[web_search, knowledge_search],
    system_prompt=get_prompt("content-strategist"),
    response_format=ContentPlan,
)


def run_strategist(brief: str, callbacks: list = None) -> ContentPlan:
    """
    Запускає Content Strategist Agent для створення контент-плану.
    
    Args:
        brief: Бриф від користувача (topic, audience, channel, tone, word_count).
        callbacks: Langfuse callback handlers для tracing.
    
    Returns:
        ContentPlan structured output.
    """
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    result = _strategist_agent.invoke(
        {"messages": [{"role": "user", "content": brief}]},
        config=config,
    )

    structured = result.get("structured_response")
    if structured and isinstance(structured, ContentPlan):
        return structured

    # Fallback: якщо structured output не спрацював
    out_msgs = result.get("messages", [])
    if out_msgs:
        content = out_msgs[-1].content
        if isinstance(content, list):
            text = " ".join([str(c.get("text", "")) for c in content if isinstance(c, dict) and "text" in c])
        else:
            text = str(content)
    else:
        text = "Strategist не зміг створити план."

    # Return a minimal ContentPlan as fallback
    return ContentPlan(
        outline=[text],
        keywords=[],
        key_messages=[],
        target_audience="unknown",
        tone="neutral",
    )
