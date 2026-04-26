"""
Content Creation Pipeline — LangGraph StateGraph.

Патерн: Prompt Chaining (Strategist → HITL gate → Writer)
       + Evaluator-Optimizer (Writer ↔ Editor, max 5 iterations).

Nodes:
  strategist_node — створює ContentPlan
  hitl_gate       — interrupt() для затвердження плану користувачем
  writer_node     — пише DraftContent за планом
  editor_node     — рев'ює контент, повертає EditFeedback
  save_node       — зберігає фінальний контент

Edges:
  START → strategist → hitl_gate → writer → editor → conditional:
    REVISION_NEEDED & iteration < 5 → writer (Command)
    APPROVED or iteration >= 5       → save → END
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langfuse.langchain import CallbackHandler

from config import settings
from schemas import PipelineState, ContentPlan, DraftContent, EditFeedback
from agents.strategist import run_strategist
from agents.writer import run_writer
from agents.editor import run_editor
from tools import save_content


# ─────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────

def strategist_node(state: PipelineState) -> dict:
    """Content Strategist: досліджує тему, створює ContentPlan."""
    # Формуємо бриф з state
    brief = (
        f"Створи контент-план для наступного запиту:\n"
        f"Тема: {state.get('topic', 'не вказано')}\n"
        f"Цільова аудиторія: {state.get('target_audience', 'не вказано')}\n"
        f"Канал: {state.get('channel', 'blog')}\n"
        f"Тон: {state.get('tone', 'не вказано')}\n"
        f"Кількість слів: {state.get('word_count', 800)}\n"
    )

    langfuse_handler = CallbackHandler()
    plan: ContentPlan = run_strategist(brief, callbacks=[langfuse_handler])

    plan_text = (
        f"📋 ContentPlan:\n"
        f"  outline: {plan.outline}\n"
        f"  keywords: {plan.keywords}\n"
        f"  key_messages: {plan.key_messages}\n"
        f"  target_audience: {plan.target_audience}\n"
        f"  tone: {plan.tone}"
    )

    return {"content_plan": plan_text, "iteration": 0}


def hitl_gate(state: PipelineState) -> Command[Literal["strategist", "writer"]]:
    """HITL gate: користувач затверджує ContentPlan або повертає з feedback."""
    plan = state.get("content_plan", "")

    # interrupt() зупиняє граф і чекає відповідь від користувача
    user_response = interrupt({
        "question": "Затвердити контент-план?",
        "plan": plan,
        "instruction": "Введіть 'approve' для затвердження або напишіть feedback для доопрацювання."
    })

    # user_response — це те, що передає користувач через Command(resume=...)
    if isinstance(user_response, dict):
        action = user_response.get("action", "approve")
        feedback = user_response.get("feedback", "")
    elif isinstance(user_response, str):
        if user_response.lower().strip() == "approve":
            action = "approve"
            feedback = ""
        else:
            action = "revise"
            feedback = user_response
    elif user_response is True:
        action = "approve"
        feedback = ""
    else:
        action = "reject"
        feedback = str(user_response)

    if action == "approve":
        return Command(goto="writer")
    else:
        # Повертаємо до Strategist з feedback — оновлюємо topic
        original_topic = state.get("topic", "")
        updated_topic = f"{original_topic}\n\nFeedback від користувача: {feedback}"
        return Command(goto="strategist", update={"topic": updated_topic})


def writer_node(state: PipelineState) -> dict:
    """Writer: пише статтю/пост за затвердженим планом."""
    plan = state.get("content_plan", "")
    feedback = state.get("edit_feedback", "")
    iteration = state.get("iteration", 0)

    # Формуємо контекст для Writer
    if feedback and iteration > 0:
        writer_input = (
            f"Перепиши контент з урахуванням зауважень Editor.\n\n"
            f"Контент-план:\n{plan}\n\n"
            f"Попередній контент:\n{state.get('draft', '')}\n\n"
            f"Зауваження Editor (ітерація {iteration}):\n{feedback}\n\n"
            f"Канал: {state.get('channel', 'blog')}\n"
            f"Кількість слів: {state.get('word_count', 800)}"
        )
    else:
        writer_input = (
            f"Напиши контент за наступним планом.\n\n"
            f"Контент-план:\n{plan}\n\n"
            f"Канал: {state.get('channel', 'blog')}\n"
            f"Кількість слів: {state.get('word_count', 800)}"
        )

    langfuse_handler = CallbackHandler()
    draft: DraftContent = run_writer(writer_input, callbacks=[langfuse_handler])

    return {
        "draft": draft.content,
        "iteration": iteration + 1,
    }


def editor_node(state: PipelineState) -> Command[Literal["writer", "save"]]:
    """Editor: рев'ює контент, повертає EditFeedback з verdict."""
    draft = state.get("draft", "")
    plan = state.get("content_plan", "")
    iteration = state.get("iteration", 0)

    editor_input = (
        f"Оціни наступний контент:\n\n"
        f"Контент-план:\n{plan}\n\n"
        f"Контент (ітерація {iteration}):\n{draft}\n\n"
        f"Канал: {state.get('channel', 'blog')}\n"
        f"Тон: {state.get('tone', 'не вказано')}\n"
    )

    langfuse_handler = CallbackHandler()
    feedback: EditFeedback = run_editor(editor_input, callbacks=[langfuse_handler])

    feedback_text = (
        f"📝 EditFeedback:\n"
        f"  verdict: {feedback.verdict}\n"
        f"  issues: {feedback.issues}\n"
        f"  tone_score: {feedback.tone_score}\n"
        f"  accuracy_score: {feedback.accuracy_score}\n"
        f"  structure_score: {feedback.structure_score}"
    )

    # Conditional routing: Command API
    if feedback.verdict == "REVISION_NEEDED" and iteration < settings.max_revisions:
        # Повертаємо до Writer з feedback (Evaluator-Optimizer cycle)
        return Command(
            goto="writer",
            update={"edit_feedback": feedback_text},
        )
    else:
        # APPROVED або досягнуто ліміт ітерацій → зберігаємо
        return Command(
            goto="save",
            update={
                "edit_feedback": feedback_text,
                "final_content": draft,
            },
        )


def save_node(state: PipelineState) -> dict:
    """Save: зберігає фінальний затверджений контент як .md файл."""
    content = state.get("final_content", state.get("draft", ""))
    topic = state.get("topic", "content")

    # Генеруємо filename з topic
    filename = topic.lower().replace(" ", "_").replace(".", "")[:50] + ".md"
    # Очищаємо від спец символів
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    if not filename.endswith(".md"):
        filename += ".md"

    result = save_content.invoke({"filename": filename, "content": content})

    return {"final_content": content}


# ─────────────────────────────────────────────
# Build the StateGraph
# ─────────────────────────────────────────────

builder = StateGraph(PipelineState)

# Add nodes
builder.add_node("strategist", strategist_node)
builder.add_node("hitl_gate", hitl_gate)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)
builder.add_node("save", save_node)

# Add edges — Prompt Chaining: START → strategist → hitl_gate → writer → editor
builder.add_edge(START, "strategist")
builder.add_edge("strategist", "hitl_gate")
# hitl_gate uses Command(goto=...) for routing: → writer or → strategist
# writer → editor is a linear edge
builder.add_edge("writer", "editor")
# editor uses Command(goto=...) for routing: → writer or → save
builder.add_edge("save", END)

# Compile with checkpointer for HITL support
memory = InMemorySaver()
pipeline_graph = builder.compile(checkpointer=memory)
