"""
LLM-as-a-Judge тест для Content Strategist.

Критерій: План відповідає брифу — враховує target audience, tone, channel.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.strategist import run_strategist


strategist_quality = GEval(
    name="Strategist Plan Quality",
    evaluation_steps=[
        "Check that the plan's target_audience matches the brief's target audience",
        "Check that the plan's tone matches the brief's requested tone",
        "Check that outline contains specific, actionable sections (not vague placeholders)",
        "Check that keywords are relevant to the topic",
        "Check that key_messages reflect the core topic of the brief",
        "Penalize if the plan contains casual tone when professional was requested, or vice versa",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


def test_strategist_blog_professional():
    """Бриф: пост для блогу про AI у медицині, професійний тон."""
    brief = (
        "Створи контент-план:\n"
        "Тема: AI у медицині — діагностика та лікування\n"
        "Цільова аудиторія: CTOs медичних стартапів\n"
        "Канал: blog\n"
        "Тон: професійний та експертний\n"
        "Кількість слів: 1000"
    )

    plan = run_strategist(brief)
    actual_output = (
        f"outline: {plan.outline}\n"
        f"keywords: {plan.keywords}\n"
        f"key_messages: {plan.key_messages}\n"
        f"target_audience: {plan.target_audience}\n"
        f"tone: {plan.tone}"
    )

    test_case = LLMTestCase(
        input=brief,
        actual_output=actual_output,
    )

    assert_test(test_case, [strategist_quality])


def test_strategist_linkedin_casual():
    """Бриф: пост для LinkedIn про продуктивність, дружній тон."""
    brief = (
        "Створи контент-план:\n"
        "Тема: 5 звичок продуктивних розробників\n"
        "Цільова аудиторія: Junior та Middle розробники\n"
        "Канал: linkedin\n"
        "Тон: дружній та мотиваційний\n"
        "Кількість слів: 300"
    )

    plan = run_strategist(brief)
    actual_output = (
        f"outline: {plan.outline}\n"
        f"keywords: {plan.keywords}\n"
        f"key_messages: {plan.key_messages}\n"
        f"target_audience: {plan.target_audience}\n"
        f"tone: {plan.tone}"
    )

    test_case = LLMTestCase(
        input=brief,
        actual_output=actual_output,
    )

    assert_test(test_case, [strategist_quality])
