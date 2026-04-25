"""
Main REPL for the Multi-Agent Research System.

Streams Agent responses, handles HITL interrupts for save_report
(approve / edit / reject) using the HumanInTheLoopMiddleware and Command.

Langfuse observability:
- Every run creates a trace in Langfuse with full sub-agent tree
- Traces are grouped by session_id and tagged with user_id
- CallbackHandler is passed via config to capture all LLM calls
"""

import uuid
from langgraph.types import Command
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from supervisor import supervisor_graph as supervisor
from config import settings

# Pre-warm retriever: завантажуємо FAISS/BM25/CrossEncoder ОДИН РАЗ до старту агентів.
# Це запобігає malloc crash на macOS ARM, коли Planner і Researcher намагаються
# ініціалізувати ретрівер одночасно (race condition → FAISS + fork crash).
from tools import _get_cached_retriever
_get_cached_retriever()

# ─────────────────────────────────────────────
# Langfuse session & user for this REPL session
# ─────────────────────────────────────────────
SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"
USER_ID = "Dmytro"


def _extract_text(content):
    """Витягує текст з контенту."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(str(c["text"]))
            elif isinstance(c, str):
                parts.append(c)
        return " ".join(parts) if parts else str(content)
    return str(content)


def handle_interrupt(config, thread_id):
    """Обробляє interrupt від HumanInTheLoopMiddleware перед збереженням звіту."""
    state_obj = supervisor.get_state(config)
    messages = state_obj.values.get("messages", [])
    
    # Знаходимо аргументи для save_report в останньому повідомленні від AI
    content = ""
    filename = ""
    if messages:
        # Шукаємо tool_call з save_report в останніх повідомленнях
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if call["name"] == "save_report":
                        content = call["args"].get("content", "")
                        filename = call["args"].get("filename", "")
                        break
            if content:
                break

    print("\n" + "=" * 60)
    print("⏸️  ACTION REQUIRES APPROVAL")
    print("=" * 60)
    print(f"  Tool:  save_report")
    
    # Показуємо аргументи як у ASSIGNMENT.md
    content_preview = content[:200] + "..." if len(content) > 200 else content
    print(f'  Args:  {{"filename": "{filename}", "content": "{content_preview}"}}')
    print()

    while True:
        try:
            choice = input("👉 approve / edit / reject: ").strip().lower()
        except UnicodeDecodeError:
            print("❌ Помилка кодування. Введіть ще раз.")
            continue
        except (EOFError, KeyboardInterrupt):
            choice = "reject"

        if choice == "approve":
            print("\n✅ Approved! Зберігаємо...")
            for chunk in supervisor.stream(
                Command(resume={"decisions": [{"type": "approve"}]}), 
                config=config, 
                stream_mode="updates"
            ):
                _process_chunk(chunk)
            break

        elif choice == "edit":
            try:
                feedback = input("✏️  Your feedback: ").strip()
            except UnicodeDecodeError:
                print("❌ Помилка кодування. Спробуйте ще раз.")
                continue

            if not feedback:
                print("Feedback cannot be empty.")
                continue

            print(f"\n🔄 Sending feedback to Supervisor...")
            for chunk in supervisor.stream(
                Command(resume={"decisions": [{"type": "edit", "edited_action": {"feedback": feedback}}]}), 
                config=config, 
                stream_mode="updates"
            ):
                _process_chunk(chunk)
                
            # Перевіряємо чи він знову зупинився
            new_state = supervisor.get_state(config)
            if new_state.next:
                handle_interrupt(config, thread_id)
            break

        elif choice == "reject":
            print(f"\n❌ Rejected. Завершуємо задання.")
            for chunk in supervisor.stream(
                Command(resume={"decisions": [{"type": "reject", "message": "Rejected by user"}]}), 
                config=config, 
                stream_mode="updates"
            ):
                _process_chunk(chunk)
            break

        else:
            print("Please enter 'approve', 'edit', or 'reject'.")


def _process_chunk(chunk):
    """Обробляє один chunk формату updates."""
    if isinstance(chunk, dict):
        for node_name, state_update in chunk.items():
            if isinstance(state_update, dict):
                messages = state_update.get("messages", [])
                if not messages:
                    continue
                
                msg = messages[-1]
                
                # AI вибрав tool call
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        args_preview = str(call.get("args", {}))
                        if len(args_preview) > 100:
                            args_preview = args_preview[:100] + "..."
                        print(f"\n🔧 {call['name']}({args_preview})")
                # Tool result повернувся
                elif getattr(msg, "type", "") == "tool" or getattr(msg, "name", ""):
                    snippet = _extract_text(msg.content)
                    name = getattr(msg, "name", "tool")
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    print(f"  📎 [{name}]: {snippet}")
                # Фінальна текстова відповідь агента
                elif getattr(msg, "content", ""):
                    text = _extract_text(msg.content)
                    if text.strip():
                        print(f"\n🤖 Agent: {text}")
    

def main():
    print("🔬 Multi-Agent Research System (Agent-as-a-Tool / Orchestrator pattern)")
    print("   Supervisor coordinates plan, research, critique, and save_report tools.")
    print(f"   📊 Langfuse tracing enabled | session: {SESSION_ID} | user: {USER_ID}")
    print("-" * 60)

    langfuse = get_client()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except UnicodeDecodeError:
            print("\n❌ Помилка кодування тексту в терміналі. Будь ласка, спробуйте ще раз.")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())[:8]

        # ── Langfuse: propagate session/user/tags to all child observations ──
        with propagate_attributes(
            session_id=SESSION_ID,
            user_id=USER_ID,
            tags=["homework-12", "multi-agent"],
        ):
            langfuse_handler = CallbackHandler()
            
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.max_iterations,
                "callbacks": [langfuse_handler],
            }

            print()

            try:
                # Запуск графу з Langfuse callback
                for chunk in supervisor.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                    stream_mode="updates",
                ):
                    _process_chunk(chunk)
                    
                # Перевірка чи граф на паузі
                state = supervisor.get_state(config)
                if state.next:
                    handle_interrupt(config, thread_id)
                    
            except Exception as e:
                print(f"\n\n❌ Помилка: {type(e).__name__}: {e}")

        # Flush Langfuse events after each query
        langfuse.flush()
        print()

    # Final shutdown
    langfuse.shutdown()


if __name__ == "__main__":
    main()