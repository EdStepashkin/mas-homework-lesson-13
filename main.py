"""
Main REPL for the Content Creation Pipeline.

Workflow:
  1. Юзер вводить бриф (topic, audience, channel, tone, word_count)
  2. Strategist створює ContentPlan
  3. HITL gate: юзер затверджує план або дає feedback
  4. Writer пише контент
  5. Editor рев'ює → Writer переробляє (до 5 разів)
  6. Фінальний контент зберігається у файл

Langfuse observability:
  - Кожен run створює trace з повним деревом суб-агентів
  - Traces згруповані за session_id та tagged user_id
"""

import uuid
from langgraph.types import Command
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from supervisor import pipeline_graph
from config import settings

# Pre-warm retriever: завантажуємо FAISS/BM25/CrossEncoder ОДИН РАЗ до старту
from tools import _get_cached_retriever
_get_cached_retriever()

# ─────────────────────────────────────────────
# Langfuse session & user
# ─────────────────────────────────────────────
SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"
USER_ID = "Dmytro"


def get_brief_from_user() -> dict:
    """Збирає бриф від користувача."""
    print("\n📝 Введіть бриф для створення контенту:")

    topic = input("  📌 Тема: ").strip()
    if not topic:
        return None

    if topic.lower() in ("exit", "quit"):
        return "exit"

    target_audience = input("  👥 Цільова аудиторія (Enter = Tech Leads): ").strip() or "Tech Leads"
    channel = input("  📢 Канал [blog/linkedin/twitter] (Enter = blog): ").strip() or "blog"
    tone = input("  🎯 Тон (Enter = впевнений та експертний): ").strip() or "впевнений та експертний"

    word_count_str = input("  📏 К-ть слів (Enter = 800): ").strip() or "800"
    try:
        word_count = int(word_count_str)
    except ValueError:
        word_count = 800

    return {
        "topic": topic,
        "target_audience": target_audience,
        "channel": channel,
        "tone": tone,
        "word_count": word_count,
    }


def handle_hitl_interrupt(config: dict):
    """Обробляє HITL interrupt від hitl_gate (затвердження ContentPlan)."""
    state = pipeline_graph.get_state(config)

    # Витягуємо interrupt payload
    interrupts = state.tasks
    plan_text = ""
    if interrupts:
        for task in interrupts:
            if hasattr(task, 'interrupts') and task.interrupts:
                for intr in task.interrupts:
                    if isinstance(intr.value, dict):
                        plan_text = intr.value.get("plan", "")
                    else:
                        plan_text = str(intr.value)
                    break
            break

    print("\n" + "=" * 60)
    print("📋 КОНТЕНТ-ПЛАН НА ЗАТВЕРДЖЕННЯ")
    print("=" * 60)
    print(plan_text)
    print("=" * 60)

    while True:
        try:
            choice = input("\n👉 approve (затвердити) / feedback (доопрацювати) / reject (відхилити): ").strip().lower()
        except (UnicodeDecodeError, EOFError, KeyboardInterrupt):
            choice = "reject"

        if choice == "approve":
            print("\n✅ План затверджено! Writer починає писати...")
            result = pipeline_graph.invoke(
                Command(resume={"action": "approve"}),
                config=config,
            )
            # Перевіряємо чи є ще interrupts (не має бути)
            new_state = pipeline_graph.get_state(config)
            if new_state.next:
                handle_hitl_interrupt(config)
            return result

        elif choice in ("feedback", "edit", "revise"):
            try:
                feedback = input("✏️  Ваш feedback: ").strip()
            except (UnicodeDecodeError, EOFError):
                print("❌ Помилка введення")
                continue

            if not feedback:
                print("Feedback не може бути порожнім.")
                continue

            print(f"\n🔄 Відправляємо feedback Strategist...")
            result = pipeline_graph.invoke(
                Command(resume={"action": "revise", "feedback": feedback}),
                config=config,
            )
            # Після доопрацювання Strategist знову зупиниться на hitl_gate
            new_state = pipeline_graph.get_state(config)
            if new_state.next:
                handle_hitl_interrupt(config)
            return result

        elif choice == "reject":
            print("\n❌ Відхилено.")
            return None

        else:
            print("Введіть 'approve', 'feedback' або 'reject'.")


def main():
    print("📝 Content Creation Pipeline (Prompt Chaining + Evaluator-Optimizer)")
    print("   Strategist → HITL gate → Writer ↔ Editor → Save")
    print(f"   📊 Langfuse tracing enabled | session: {SESSION_ID} | user: {USER_ID}")
    print("-" * 60)

    langfuse = get_client()

    while True:
        brief = get_brief_from_user()

        if brief is None:
            continue
        if brief == "exit":
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())[:8]

        # Langfuse: propagate session/user/tags
        with propagate_attributes(
            session_id=SESSION_ID,
            user_id=USER_ID,
            tags=["homework-13", "content-pipeline"],
        ):
            langfuse_handler = CallbackHandler()

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.max_iterations,
                "callbacks": [langfuse_handler],
            }

            print(f"\n🚀 Запускаємо pipeline для: \"{brief['topic']}\"")
            print(f"   Канал: {brief['channel']} | Аудиторія: {brief['target_audience']}")
            print(f"   Тон: {brief['tone']} | Слів: {brief['word_count']}")

            try:
                # Початковий state
                initial_state = {
                    "messages": [],
                    "topic": brief["topic"],
                    "target_audience": brief["target_audience"],
                    "channel": brief["channel"],
                    "tone": brief["tone"],
                    "word_count": brief["word_count"],
                    "content_plan": "",
                    "draft": "",
                    "edit_feedback": "",
                    "iteration": 0,
                    "final_content": "",
                }

                # Запуск — граф зупиниться на hitl_gate (interrupt)
                result = pipeline_graph.invoke(initial_state, config=config)

                # Перевіряємо чи граф на паузі (interrupt)
                state = pipeline_graph.get_state(config)
                if state.next:
                    result = handle_hitl_interrupt(config)

                # Показуємо результат
                if result and isinstance(result, dict):
                    final = result.get("final_content", "")
                    iteration = result.get("iteration", 0)
                    feedback = result.get("edit_feedback", "")

                    print("\n" + "=" * 60)
                    print("✅ КОНТЕНТ СТВОРЕНО!")
                    print("=" * 60)
                    print(f"  📊 Ітерацій Writer↔Editor: {iteration}")
                    if feedback:
                        print(f"  📝 Останній feedback Editor:")
                        for line in feedback.split("\n"):
                            print(f"     {line}")
                    if final:
                        preview = final[:300] + "..." if len(final) > 300 else final
                        print(f"\n  📄 Превʼю контенту:\n{preview}")
                    print("=" * 60)

            except Exception as e:
                print(f"\n❌ Помилка: {type(e).__name__}: {e}")

        langfuse.flush()
        print()

    langfuse.shutdown()


if __name__ == "__main__":
    main()