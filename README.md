# Конвеєр створення контенту 📝✨📊

Мультиагентна система для створення контенту для блогу та соцмереж. Планує, пише та перевіряє якість перед публікацією.

Побудована на базі **LangChain** та **LangGraph**, використовує комбінацію двох патернів Anthropic:
- **Prompt Chaining** (Strategist → HITL gate → Writer) — лінійний pipeline з Human-in-the-Loop
- **Evaluator-Optimizer** (Writer ↔ Editor) — цикл генерації та оцінки якості (до 5 ітерацій)

> Homework-13: Content Creation Pipeline з Langfuse Observability та LLM-as-a-Judge тестами.

---

## 🏗 Архітектура

```
User (REPL)
  │ brief: topic, audience, channel, tone, word_count
  ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                │
│                                                      │
│  START                                               │
│    │                                                 │
│    ▼                                                 │
│  Content Strategist  ─── DuckDuckGo + RAG ──→ ContentPlan
│    │                                                 │
│    ▼                                                 │
│  HITL Gate ← interrupt() ─── User approves/revises plan
│    │                                                 │
│    ▼                                                 │
│  Writer  ─── DuckDuckGo ──→ DraftContent             │
│    │                                                 │
│    ▼                                                 │
│  Editor  ─── DuckDuckGo (fact-check) ──→ EditFeedback│
│    │                                                 │
│    ├── verdict: REVISION_NEEDED & iter < 5           │
│    │   └── Command(goto="writer") ── back to Writer  │
│    │                                                 │
│    └── verdict: APPROVED                             │
│        └── Save ──→ .md file ──→ END                 │
│                                                      │
└─────────────────────────────────────────────────────┘

All system prompts loaded from Langfuse Prompt Management (label: production)
All traces sent to Langfuse with session_id + user_id + tags
```

### Агенти

| Агент | Роль | Інструменти | Structured Output |
|-------|------|-------------|-------------------|
| **Content Strategist** | Досліджує тему, створює контент-план | `web_search`, `knowledge_search` (RAG) | `ContentPlan` |
| **Writer** | Пише статтю/пост за планом | `web_search`, `save_content` | `DraftContent` |
| **Editor** | Рев'ює контент з числовими оцінками | `web_search` (fact-check) | `EditFeedback` |

### Structured Output контракти (Pydantic)

| Модель | Поля |
|--------|------|
| `ContentPlan` | `outline: list[str], keywords: list[str], key_messages: list[str], target_audience: str, tone: str` |
| `DraftContent` | `content: str, word_count: int, keywords_used: list[str]` |
| `EditFeedback` | `verdict: Literal["APPROVED","REVISION_NEEDED"], issues: list[str], tone_score: float, accuracy_score: float, structure_score: float` |

---

## 🌟 Ключові можливості

- **Prompt Chaining + Evaluator-Optimizer**: два патерни Anthropic у одному pipeline
- **LangGraph StateGraph**: explicit nodes + conditional edges + Command API
- **HITL gate**: `interrupt()` для затвердження ContentPlan перед написанням
- **Evaluator-Optimizer цикл**: Writer ↔ Editor до 5 ітерацій через `Command(goto=...)`
- **RAG для style guide**: FAISS + BM25 + CrossEncoder по brand documents
- **Structured Output**: всі три агенти повертають Pydantic-моделі
- **Langfuse Tracing**: повне дерево LLM-викликів, tool calls у кожному trace
- **Prompt Management**: промпти з Langfuse за label `production`
- **LLM-as-a-Judge тести**: DeepEval з GEval метриками

---

## 🛠 Технологічний стек

- **LLM**: Google Gemini (`gemini-3-flash-preview`) через `ChatGoogleGenerativeAI`
- **Pipeline**: `langgraph.graph.StateGraph` з `interrupt()` та `Command`
- **Observability**: `langfuse` — tracing, prompt management, evaluators
- **RAG**: `FAISS`, `OpenAIEmbeddings`, `BM25Retriever`, `HuggingFaceCrossEncoder`, `EnsembleRetriever`
- **Тестування**: `DeepEval` (GEval, AnswerRelevancy), `pytest`
- **Конфігурація**: Pydantic `BaseSettings` + `.env`

---

## 📁 Структура проєкту

```
homework-lesson-13/
├── main.py                  # REPL з Langfuse tracing + HITL для ContentPlan
├── supervisor.py            # LangGraph StateGraph (5 nodes, conditional edges)
├── agents/
│   ├── __init__.py
│   ├── strategist.py        # Content Strategist (response_format=ContentPlan)
│   ├── writer.py            # Writer (response_format=DraftContent)
│   └── editor.py            # Editor (response_format=EditFeedback)
├── config.py                # Settings + Langfuse init + get_prompt()
├── setup_prompts.py         # Одноразовий скрипт: створення промптів у Langfuse
├── schemas.py               # Pydantic: ContentPlan, DraftContent, EditFeedback, PipelineState
├── tools.py                 # web_search, knowledge_search, save_content
├── retriever.py             # Hybrid search: FAISS + BM25 + CrossEncoder reranking
├── ingest.py                # Ingestion pipeline: Markdown → chunks → FAISS index
├── data/
│   ├── style_guide.md       # Style guide бренду NovaTech Solutions
│   ├── brand_description.md # Опис бренду: місія, продукт, конкурентні переваги
│   └── examples/            # 7 прикладів контенту (blog posts, LinkedIn posts)
├── tests/
│   ├── golden_dataset.json  # 5 golden scenarios для E2E
│   ├── test_strategist.py   # GEval: план відповідає брифу
│   ├── test_writer.py       # GEval: контент покриває outline + keywords
│   ├── test_editor.py       # GEval: feedback actionable, scores адекватні
│   └── test_e2e.py          # E2E: фінальний контент відповідає брифу
├── screenshots/             # Скріншоти Langfuse UI
├── requirements.txt         # Залежності
├── example_output/          # Згенерований контент (.md файли)
├── index/                   # FAISS індекс (не в Git)
└── .env                     # API-ключі (не в Git)
```

---

## 🚀 Встановлення та запуск

### 1. Клонування репозиторію
```bash
git clone https://github.com/EdStepashkin/mas-homework-lesson-13.git
cd homework-lesson-13
```

### 2. Створення віртуального середовища
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 4. Налаштування змінних середовища (.env)

Створіть файл `.env`:

```env
GEMINI_API_KEY=AIzaSyYourApiKeyHere...
OPENAI_API_KEY=sk-proj-YourOpenAiKey...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

### 5. Індексація brand документів (Ingestion)

```bash
python ingest.py
```

Очікуваний вивід:
```
Loading Markdown documents from data/ ...
Found 9 Markdown files:
  - data/style_guide.md
  - data/brand_description.md
  - data/examples/blog_cycle_time.md
  ...
Generated 85 chunks.
✅ Ingestion complete!
```

### 6. Створення промптів у Langfuse (одноразово)

```bash
python setup_prompts.py
```

### 7. Запуск системи

```bash
python main.py
```

---

## 📊 Langfuse Observability

### Як працює tracing

При кожному запуску `main.py`:
1. Створюється `CallbackHandler`, який передається agents через `config`
2. Через `propagate_attributes()` додаються `session_id`, `user_id`, `tags`
3. Після кожного запиту — `langfuse.flush()`

### Prompt Management

Промпти завантажуються з Langfuse через `get_prompt()`:

| Prompt Name | Агент | Template Var |
|-------------|-------|-------------|
| `content-strategist` | Strategist | `{{current_date}}` |
| `content-writer` | Writer | `{{current_date}}` |
| `content-editor` | Editor | `{{current_date}}` |

### LLM-as-a-Judge Evaluators

У Langfuse UI налаштовано evaluators:
1. **Relevance** (numeric 1-5) — відповідність output до input
2. **Completeness** (boolean) — повнота контенту

---

## 💬 Приклад роботи

```
📝 Content Creation Pipeline (Prompt Chaining + Evaluator-Optimizer)
   Strategist → HITL gate → Writer ↔ Editor → Save
   📊 Langfuse tracing enabled | session: session-a1b2c3d4 | user: Dmytro
------------------------------------------------------------

📝 Введіть бриф для створення контенту:
  📌 Тема: AI-асистенти в розробці: хайп чи продуктивність
  👥 Цільова аудиторія (Enter = Tech Leads):
  📢 Канал [blog/linkedin/twitter] (Enter = blog):
  🎯 Тон (Enter = впевнений та експертний):
  📏 К-ть слів (Enter = 800):

🚀 Запускаємо pipeline для: "AI-асистенти в розробці: хайп чи продуктивність"
   Канал: blog | Аудиторія: Tech Leads
   Тон: впевнений та експертний | Слів: 800

============================================================
📋 КОНТЕНТ-ПЛАН НА ЗАТВЕРДЖЕННЯ
============================================================
📋 ContentPlan:
  outline: ['Вступ: AI-хайп vs реальність', 'Де AI працює найкраще', ...]
  keywords: ['AI assistants', 'developer productivity', ...]
  key_messages: ['AI прискорює рутину, але не замінює архітектурні рішення']
  target_audience: Tech Leads
  tone: впевнений та експертний
============================================================

👉 approve (затвердити) / feedback (доопрацювати) / reject: approve

✅ План затверджено! Writer починає писати...

============================================================
✅ КОНТЕНТ СТВОРЕНО!
============================================================
  📊 Ітерацій Writer↔Editor: 2
  📝 Останній feedback Editor:
     verdict: APPROVED
     tone_score: 0.9
     accuracy_score: 0.85
     structure_score: 0.9

  📄 Превʼю контенту:
# AI-асистенти в розробці: хайп чи реальна продуктивність?

Кожна друга стаття в tech-медіа обіцяє: "AI замінить розробників"...
============================================================
```

---

## 🧪 Тестування (LLM-as-a-Judge)

### Запуск тестів

```bash
# Усі тести
deepeval test run tests/

# Окремі файли
deepeval test run tests/test_strategist.py -v
deepeval test run tests/test_writer.py -v
deepeval test run tests/test_editor.py -v
deepeval test run tests/test_e2e.py -v
```

### Що тестується

| Файл | Що тестується | Критерій | Поріг |
|------|--------------|----------|-------|
| `test_strategist.py` | Strategist | План відповідає брифу: audience, tone, channel | 0.7 |
| `test_writer.py` | Writer | Контент покриває outline та keywords з плану | 0.7 |
| `test_editor.py` | Editor | Feedback actionable, scores адекватні якості | 0.7 |
| `test_e2e.py` | End-to-end | Фінальний контент відповідає початковому брифу | 0.6 / 0.7 |

### RAG Data для бренду

Бренд **NovaTech Solutions** (вигаданий):
- Style guide: tone of voice, аудиторія, заборонені/рекомендовані формулювання
- Brand description: місія, продукт, конкурентні переваги
- 7 прикладів контенту: 4 blog posts + 3 LinkedIn posts

---

## 📸 Скріншоти Langfuse UI

Скріншоти знаходяться в папці `screenshots/`:
1. **Trace tree** — повне дерево суб-агентів та tool calls
2. **Session** — сесія з кількома трейсами
3. **Evaluator scores** — автоматичні оцінки від LLM-as-a-Judge
4. **Prompt management** — промпти 3 агентів

---

*Оригінальне завдання доступне у файлі `ASSIGNMENT.md`.*