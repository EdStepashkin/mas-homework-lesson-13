"""
LLM-as-a-Judge тест для Writer.

Критерій: Контент відповідає плану — покриває всі пункти outline, використовує keywords.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.writer import run_writer


writer_quality = GEval(
    name="Writer Content Quality",
    evaluation_steps=[
        "Extract the outline points from the input plan",
        "Check that EACH outline point is covered in the actual output content",
        "Extract the keywords from the input plan",
        "Check that most keywords are naturally used in the content",
        "Check that key_messages from the plan are reflected in the content",
        "Penalize if any outline section is completely missing",
        "Penalize if the content has no structure (headers, lists, formatting)",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)


def test_writer_covers_outline():
    """План з 5 пунктами outline → Writer покриває всі 5 у тексті."""
    plan_input = (
        "Напиши контент за наступним планом.\n\n"
        "Контент-план:\n"
        "  outline: ['Вступ: проблема cycle time', 'Менші PR = швидший review', "
        "'Автоматизація code review', 'TDD як інвестиція', 'WIP ліміти', 'Висновок та CTA']\n"
        "  keywords: ['cycle time', 'code review', 'TDD', 'WIP ліміти', 'продуктивність']\n"
        "  key_messages: ['Швидкість і якість не суперечать одне одному', "
        "'Правильні процеси скорочують cycle time']\n"
        "  target_audience: Tech Leads\n"
        "  tone: впевнений та експертний\n\n"
        "Канал: blog\n"
        "Кількість слів: 800"
    )

    draft = run_writer(plan_input)
    actual_output = draft.content

    test_case = LLMTestCase(
        input=plan_input,
        actual_output=actual_output,
    )

    assert_test(test_case, [writer_quality])


def test_writer_linkedin_format():
    """LinkedIn-пост має бути коротким з хештегами."""
    plan_input = (
        "Напиши контент за наступним планом.\n\n"
        "Контент-план:\n"
        "  outline: ['Hook: статистика про WIP', 'Проблема: жонглювання задачами', "
        "'Рішення: WIP ліміт = 2', 'Результат з цифрами', 'CTA']\n"
        "  keywords: ['WIP limits', 'productivity', 'agile']\n"
        "  key_messages: ['Менше задач одночасно = швидше завершення']\n"
        "  target_audience: Tech Leads\n"
        "  tone: професійний але дружній\n\n"
        "Канал: linkedin\n"
        "Кількість слів: 250"
    )

    draft = run_writer(plan_input)
    actual_output = draft.content

    test_case = LLMTestCase(
        input=plan_input,
        actual_output=actual_output,
    )

    assert_test(test_case, [writer_quality])
