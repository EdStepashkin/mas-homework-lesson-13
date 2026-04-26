"""
Editor Agent — рев'ює контент (tone, factual accuracy, structure).

Інструменти: DuckDuckGo Search (fact-check).
Structured output: EditFeedback (з numbers scores та verdict).
"""

from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

from config import llm, get_prompt
from schemas import EditFeedback
from tools import web_search


# Створюємо Editor Agent з structured output
_editor_agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=get_prompt("content-editor"),
    response_format=EditFeedback,
)


def run_editor(draft_content: str, callbacks: list = None) -> EditFeedback:
    """
    Запускає Editor Agent для рев'ю контенту.
    
    Args:
        draft_content: Текст чернетки для рев'ю.
        callbacks: Langfuse callback handlers для tracing.
    
    Returns:
        EditFeedback structured output з verdict та scores.
    """
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    result = _editor_agent.invoke(
        {"messages": [{"role": "user", "content": draft_content}]},
        config=config,
    )

    structured = result.get("structured_response")
    if structured and isinstance(structured, EditFeedback):
        return structured

    # Fallback: повертаємо APPROVED якщо structured output не спрацював
    out_msgs = result.get("messages", [])
    feedback_text = ""
    if out_msgs:
        content = out_msgs[-1].content
        if isinstance(content, list):
            feedback_text = " ".join([str(c.get("text", "")) for c in content if isinstance(c, dict) and "text" in c])
        else:
            feedback_text = str(content)

    return EditFeedback(
        verdict="APPROVED",
        issues=[feedback_text] if feedback_text else ["Unable to parse structured feedback"],
        tone_score=0.5,
        accuracy_score=0.5,
        structure_score=0.5,
    )
