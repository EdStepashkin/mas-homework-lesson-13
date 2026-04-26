"""
LLM-as-a-Judge тест для Editor.

Критерій: Feedback конкретний і actionable, scores відповідають реальній якості.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.editor import run_editor


editor_quality = GEval(
    name="Editor Feedback Quality",
    evaluation_steps=[
        "Evaluate ONLY the quality of the Editor's feedback in ACTUAL_OUTPUT.",
        "Check that the feedback identifies specific, actionable issues",
        "If verdict is APPROVED, issues list should be empty or contain only minor items",
        "If verdict is REVISION_NEEDED, there must be at least one concrete issue",
        "Check that scores make sense given the input quality. Bad input should get low scores, good input higher scores.",
        "IMPORTANT: Do NOT penalize the feedback if the input content is terrible. The Editor is doing its job by giving REVISION_NEEDED.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


def test_editor_catches_bad_content():
    """Свідомо поганий текст (off-topic, wrong tone) → Editor має виявити проблеми та поставити низькі scores."""
    bad_content = (
        "Оціни наступний контент:\n\n"
        "Контент-план:\n"
        "  outline: ['Вступ', 'Переваги AI', 'Кейси використання', 'Висновок']\n"
        "  tone: професійний та серйозний\n\n"
        "Контент (ітерація 1):\n"
        "ЙОООО! 🎉🎉🎉 AI це просто РЕВОЛЮЦІЯ!!! Всім треба юзати AI прямо зараз!!!\n"
        "Це легко і просто! Будь-хто може! Довіряйте мені, я знаю!!!\n"
        "AI замінить всіх програмістів до 2025 року, це точно!!!\n"
        "Підписуйтесь на мій канал!!!\n\n"
        "Канал: blog\n"
        "Тон: професійний та серйозний"
    )

    feedback = run_editor(bad_content)
    actual_output = (
        f"verdict: {feedback.verdict}\n"
        f"issues: {feedback.issues}\n"
        f"tone_score: {feedback.tone_score}\n"
        f"accuracy_score: {feedback.accuracy_score}\n"
        f"structure_score: {feedback.structure_score}"
    )

    test_case = LLMTestCase(
        input=bad_content,
        actual_output=actual_output,
    )

    assert_test(test_case, [editor_quality])


def test_editor_approves_good_content():
    """Якісний контент відповідно до плану → Editor має затвердити."""
    good_content = (
        "Оціни наступний контент:\n\n"
        "Контент-план:\n"
        "  outline: ['Вступ: проблема', 'Рішення 1', 'Рішення 2', 'Висновок']\n"
        "  keywords: ['cycle time', 'code review', 'productivity']\n"
        "  tone: впевнений та експертний\n\n"
        "Контент (ітерація 1):\n"
        "# Як скоротити cycle time без втрати якості\n\n"
        "Кожен Tech Lead стикається з дилемою: менеджмент хоче швидше, команда хоче якісніше. "
        "За нашим досвідом, це не протиріччя.\n\n"
        "## Рішення 1: Менші PR\n"
        "Дослідження Microsoft показує: PR з менше ніж 200 рядками отримують review у 3 рази швидше. "
        "Встановіть ліміт у 300 рядків і побачите результат за 2 спринти.\n\n"
        "## Рішення 2: Автоматизація code review\n"
        "80% коментарів — стилістичні. Налаштуйте ESLint/Prettier з pre-commit hooks "
        "і звільніть час для змістовного review.\n\n"
        "## Висновок\n"
        "Швидкість і якість — результат правильних процесів. Почніть з одного пункту "
        "і виміряйте результат через 2 спринти.\n\n"
        "Канал: blog\n"
        "Тон: впевнений та експертний"
    )

    feedback = run_editor(good_content)
    actual_output = (
        f"verdict: {feedback.verdict}\n"
        f"issues: {feedback.issues}\n"
        f"tone_score: {feedback.tone_score}\n"
        f"accuracy_score: {feedback.accuracy_score}\n"
        f"structure_score: {feedback.structure_score}"
    )

    test_case = LLMTestCase(
        input=good_content,
        actual_output=actual_output,
    )

    assert_test(test_case, [editor_quality])
