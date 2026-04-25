# Мультиагентна дослідницька система з Langfuse Observability 🤖🔬📊

Мультиагентна AI-система, побудована на базі **LangChain** та **LangGraph**, яка координує трьох спеціалізованих суб-агентів за патерном **Plan → Research → Critique**.

Supervisor оркеструє ітеративний цикл дослідження: Planner декомпозує запит, Researcher виконує глибокий аналіз, а Critic верифікує результати і може повернути на доопрацювання. Збереження звіту захищене через **Human-in-the-Loop (HITL)** — користувач затверджує, редагує або відхиляє фінальний документ.

**Нове в цій версії (homework-12):**
- 📊 **Langfuse Tracing** — кожен запуск створює trace з повним деревом суб-агентів та tool calls
- 🗂 **Session & User tracking** — traces згруповані в sessions, мають user_id
- 📝 **Prompt Management** — усі system prompts завантажуються з Langfuse (жодних захардкоджених)
- ⚖️ **LLM-as-a-Judge** — автоматична оцінка якості через Langfuse Evaluators

> Розширення мультиагентної системи (hw-8) + тести DeepEval (hw-10) + Langfuse observability (hw-12).

---

## 🏗 Архітектура

```
User (REPL)
  │
  ▼
Supervisor Agent  ◄── Langfuse CallbackHandler (tracing)
  │
  ├── 1. plan(request)       → Planner Agent      → structured ResearchPlan
  │
  ├── 2. research(plan)      → Research Agent      → findings (web + knowledge base)
  │
  ├── 3. critique(findings)  → Critic Agent        → structured CritiqueResult
  │       │
  │       ├── verdict: "APPROVE"  → step 4
  │       └── verdict: "REVISE"   → back to step 2 with feedback (max 2 rounds)
  │
  └── 4. save_report(...)    → HITL gated          → approve / edit / reject

All system prompts loaded from Langfuse Prompt Management (label: production)
All traces sent to Langfuse with session_id + user_id + tags
```

### Суб-агенти

| Агент | Роль | Інструменти | Structured Output |
|-------|------|-------------|-------------------|
| **Planner** | Декомпозиція запиту у план дослідження | `web_search`, `knowledge_search` | `ResearchPlan` |
| **Researcher** | Глибоке дослідження за планом | `web_search`, `read_url`, `knowledge_search` | — |
| **Critic** | Верифікація якості (freshness, completeness, structure) | `web_search`, `read_url`, `knowledge_search` | `CritiqueResult` |
| **Supervisor** | Оркестрація циклу Plan→Research→Critique→Save | `plan`, `research`, `critique`, `save_report` | — |

---

## 🌟 Ключові можливості

- **Мультиагентна оркестрація**: Supervisor координує 3 спеціалізованих агенти через `create_agent`
- **Structured Output**: Planner і Critic повертають Pydantic-моделі через `response_format`
- **Ітеративне дослідження**: Critic може повернути Researcher на доопрацювання (evaluator-optimizer)
- **HITL (Human-in-the-Loop)**: `HumanInTheLoopMiddleware` перехоплює `save_report`
- **RAG з гібридним пошуком**: FAISS (семантичний) + BM25 (лексичний) + CrossEncoder реранкінг
- **Langfuse Tracing**: повне дерево LLM-викликів, tool calls, суб-агентів у кожному trace
- **Prompt Management**: усі system prompts завантажуються з Langfuse за іменем та label `production`
- **Online Evaluation**: LLM-as-a-Judge автоматично оцінює нові traces
- **Автоматизовані тести**: DeepEval із GEval метриками та e2e evaluation

---

## 🛠 Технологічний стек

- **LLM**: Google Gemini (`gemini-3-flash-preview`) через `ChatGoogleGenerativeAI`
- **Агентний фреймворк**: `langchain.agents.create_agent` + `HumanInTheLoopMiddleware`
- **Observability**: `langfuse` (v4+) — tracing, prompt management, evaluators
- **Персистентність**: `langgraph.checkpoint.memory.InMemorySaver`
- **RAG-пайплайн**: `FAISS`, `OpenAIEmbeddings`, `BM25Retriever`, `HuggingFaceCrossEncoder`, `EnsembleRetriever`
- **Тестування**: `DeepEval` (GEval, AnswerRelevancy, ToolCorrectness), `pytest`
- **Конфігурація**: Pydantic `BaseSettings` + `.env`

---

## 📁 Структура проєкту

```
homework-lesson-12/
├── main.py                  # REPL з Langfuse tracing + HITL interrupt/resume
├── supervisor.py            # Supervisor Agent + HITL middleware
├── agents/
│   ├── __init__.py
│   ├── planner.py           # Planner Agent (response_format=ResearchPlan)
│   ├── research.py          # Research Agent
│   └── critic.py            # Critic Agent (response_format=CritiqueResult)
├── config.py                # Settings + Langfuse init + get_prompt()
├── setup_prompts.py         # Одноразовий скрипт: створення промптів у Langfuse
├── schemas.py               # Pydantic-моделі: ResearchPlan, CritiqueResult
├── tools.py                 # web_search, read_url, knowledge_search, save_report
├── retriever.py             # Hybrid search: FAISS + BM25 + CrossEncoder reranking
├── ingest.py                # Ingestion pipeline: PDF → chunks → FAISS index
├── tests/
│   ├── golden_dataset.json  # 15 golden examples
│   ├── test_planner.py      # GEval Plan Quality
│   ├── test_researcher.py   # GEval Groundedness
│   ├── test_critic.py       # GEval Critique Quality
│   ├── test_tools.py        # ToolCorrectnessMetric
│   └── test_e2e.py          # E2E evaluation на golden dataset
├── screenshots/             # Скріншоти Langfuse UI (trace tree, session, evaluators, prompts)
├── requirements.txt         # Залежності
├── data/                    # Вхідні PDF-документи для RAG
├── index/                   # Згенеровані індекси (не в Git)
├── example_output/          # Приклади згенерованих звітів
└── .env                     # API-ключі (не в Git)
```

---

## 🚀 Встановлення та запуск

### 1. Клонування репозиторію
```bash
git clone https://github.com/EdStepashkin/mas-homework-lesson-12.git
cd homework-lesson-12
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

Створіть файл `.env` у кореневій директорії:

```env
# LLM та Embeddings
GEMINI_API_KEY=AIzaSyYourApiKeyHere...
OPENAI_API_KEY=sk-proj-YourOpenAiKey...

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

**Де взяти ключі Langfuse:**
1. Зареєструйтесь на [us.cloud.langfuse.com](https://us.cloud.langfuse.com) (безкоштовно, без кредитної карти)
2. Створіть Organization → Project (наприклад, `homework-12`)
3. **Settings → API Keys → + Create new API keys** — скопіюйте `Public Key` та `Secret Key`

### 5. Індексація документів (Ingestion)

Помістіть PDF-документи у папку `data/` і запустіть:
```bash
python ingest.py
```

### 6. Створення промптів у Langfuse (одноразово)

Цей скрипт завантажить system prompts усіх 4 агентів у Langfuse Prompt Management з label `production`:

```bash
python setup_prompts.py
```

Очікуваний вивід:
```
Creating prompt: planner-agent ...
  ✅ planner-agent created with label 'production'
Creating prompt: researcher-agent ...
  ✅ researcher-agent created with label 'production'
Creating prompt: critic-agent ...
  ✅ critic-agent created with label 'production'
Creating prompt: supervisor-agent ...
  ✅ supervisor-agent created with label 'production'

🎉 All 4 prompts created in Langfuse!
```

### 7. Запуск системи

```bash
python main.py
```

---

## 📊 Langfuse Observability

### Як працює tracing

При кожному запуску `main.py`:

1. Створюється **Langfuse `CallbackHandler`**, який передається в `config={"callbacks": [langfuse_handler]}`
2. Через `propagate_attributes()` до кожного trace додаються:
   - `session_id` — унікальний для кожного запуску REPL (групує traces в одну сесію)
   - `user_id` — `"Dmytro"`
   - `tags` — `["homework-12", "multi-agent"]`
3. Після кожного запиту виконується `langfuse.flush()` — дані відправляються в Langfuse Cloud

### Prompt Management

Усі system prompts завантажуються з Langfuse через `get_prompt()`:

```python
from config import get_prompt

# Завантажує prompt за іменем з label="production"
# Компілює {{current_date}} у поточну дату
prompt = get_prompt("supervisor-agent")
```

| Prompt Name | Агент | Template Variables |
|-------------|-------|--------------------|
| `planner-agent` | Planner | `{{current_date}}` |
| `researcher-agent` | Researcher | `{{current_date}}` |
| `critic-agent` | Critic | `{{current_date}}` |
| `supervisor-agent` | Supervisor | `{{current_date}}` |

### LLM-as-a-Judge Evaluators

У Langfuse UI (\*\*LLM-as-a-Judge → Evaluators\*\*) налаштовано 2 evaluators:

1. **Relevance** (numeric 1-5) — оцінює відповідність output до input запиту
2. **Completeness** (boolean) — оцінює повноту дослідження

Evaluators автоматично оцінюють кожен новий trace.

### Де дивитися результати

Перейдіть на [us.cloud.langfuse.com](https://us.cloud.langfuse.com) → ваш проєкт:

| Розділ | Що перевірити                                                                         |
|--------|---------------------------------------------------------------------------------------|
| **Tracing → Traces** | Список усіх запусків. Кожен розгортається у повне дерево з суб-агентами та tool calls |
| **Sessions** | Ваша сесія з кількома трейсами всередині                                              |
| **Users** | Ваш user (`Dmytro`)                                                                   |
| **Prompts** | 4 промпти з label `production`                                                        |
| **LLM-as-a-Judge → Evaluators** | 2 evaluators та їх статус                                                             |
| **Trace → Scores** | Автоматично проставлені scores на кожному trace                                       |

---

## 🧪 Тестування та валідація

### Тестові запити для генерації traces

Після запуску `python main.py` введіть 3-5 таких запитів (один за одним):

```
You: Порівняй підходи RAG: naive, sentence-window та parent-child retrieval
```

```
You: Які основні переваги та недоліки використання LLM агентів у продакшн-системах?
```

```
You: Розкажи про сучасні методи fine-tuning великих мовних моделей у 2025-2026 роках
```

```
You: Порівняй FAISS та ChromaDB для векторного пошуку
```

```
You: Що таке Retrieval Augmented Generation і як воно працює?
```

Кожен запит пройде цикл Plan → Research → Critique → (можливо REVISE) → Save Report.

На етапі `save_report` система зупиниться і попросить вас:
```
⏸️  ACTION REQUIRES APPROVAL
  Tool:  save_report
  Args:  {"filename": "...", "content": "..."}

👉 approve / edit / reject:
```

Введіть `approve` щоб зберегти звіт, `edit` щоб дати фідбек, або `reject` щоб відхилити.

Для виходу з системи введіть:
```
You: exit
```

### Перевірка результатів у Langfuse

Після 3-5 запусків:

1. Зайдіть на [us.cloud.langfuse.com](https://us.cloud.langfuse.com)
2. **Tracing → Traces** — має бути 3-5 рядків, кожен розгортається у повне дерево
3. **Sessions** — сесія з кількома трейсами
4. **Prompts** — 4 промпти

### DeepEval тести

```bash
# Усі тести
deepeval test run tests/

# Окремі файли
deepeval test run tests/test_planner.py -v
deepeval test run tests/test_researcher.py -v
deepeval test run tests/test_critic.py -v
deepeval test run tests/test_tools.py -v
deepeval test run tests/test_e2e.py -v
```

| Файл | Що тестує | Метрика | Поріг |
|------|-----------|---------|-------|
| `test_planner.py` | Якість плану | `GEval("Plan Quality")` | 0.7 |
| `test_researcher.py` | Обґрунтованість відповіді | `GEval("Groundedness")` | 0.7 |
| `test_critic.py` | Конкретність критики | `GEval("Critique Quality")` | 0.7 |
| `test_tools.py` | Правильність tool calls | `ToolCorrectnessMetric` | 0.5 |
| `test_e2e.py` | Повний pipeline | `GEval` + `AnswerRelevancy` | 0.6 / 0.7 |

---

## 💬 Приклад роботи

```
🔬 Multi-Agent Research System (Agent-as-a-Tool / Orchestrator pattern)
   Supervisor coordinates plan, research, critique, and save_report tools.
   📊 Langfuse tracing enabled | session: session-a1b2c3d4 | user: Dmytro
------------------------------------------------------------

You: Порівняй підходи RAG: naive, sentence-window та parent-child retrieval

🔧 plan("Порівняй підходи RAG: naive, sentence-window та parent-child retrieval")
  📎 [plan]: 📎 ResearchPlan:
    goal: Порівняти три підходи до RAG...
    search_queries: ["naive RAG approach", "sentence-window retrieval", ...]
    sources_to_check: ["knowledge_base", "web"]

🔧 research("📎 ResearchPlan: goal: Порівняти три підходи...")
  📎 [research]: 🔬 Знахідки Researcher:
    - Факт 1: Naive RAG використовує фіксовані чанки... (джерело: knowledge_base)
    - Факт 2: Sentence-window повертає вікно навколо... (джерело: web)
    ...

🔧 critique("🔬 Знахідки Researcher...")
  📎 [critique]: 📎 CritiqueResult:
    verdict: APPROVE
    is_fresh: True
    is_complete: True
    strengths: ["Актуальні дані", "Покриті всі три підходи"]

🔧 save_report({"filename": "rag_comparison.md", "content": "# Порівняння підходів RAG..."})

============================================================
⏸️  ACTION REQUIRES APPROVAL
============================================================
  Tool:  save_report
  Args:  {"filename": "rag_comparison.md", "content": "# Порівняння підходів RAG..."}

👉 approve / edit / reject: approve

✅ Approved! Зберігаємо...

🤖 Agent: Звіт збережено у example_output/rag_comparison.md
```

---

## 📸 Скріншоти Langfuse UI

Скріншоти знаходяться в папці `screenshots/`:

1. **Trace tree** — повне дерево суб-агентів та tool calls
2. **Session** — сесія з кількома трейсами
3. **Evaluator scores** — автоматичні оцінки від LLM-as-a-Judge
4. **Prompt management** — промпти всіх 4 агентів

---

*Оригінальне завдання доступне у файлі `ASSIGNMENT.md`.*