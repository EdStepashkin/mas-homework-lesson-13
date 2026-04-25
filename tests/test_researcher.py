import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.research import research

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)

def test_research_grounded():
    # Симулюємо ResearchPlan
    plan_text = "Goal: Find who won the 2022 World Cup. Search queries: '2022 World Cup winner'. Sources: web"
    
    # Викликаємо агента
    actual_output = research.invoke({"plan_or_feedback": plan_text})
    
    # Для GEval groundedness, retrieval context це те що знайшов веб пошук
    # Оскільки research під капотом сам робить пошук, ми симулюємо retrieval context
    # як загальновідомий факт щоб тест міг оцінити. 
    # (В реальності треба було б перехопити результати web_search, але для спрощення HW
    # ми передаємо контекст вручну).
    mock_retrieval_context = ["Argentina won the 2022 FIFA World Cup, defeating France in the final."]
    
    test_case = LLMTestCase(
        input=plan_text,
        actual_output=actual_output,
        retrieval_context=mock_retrieval_context
    )
    
    assert_test(test_case, [groundedness])

def test_research_edge_case():
    plan_text = "Goal: Find the biography of a fictional character named 'Zygarflax'. Search queries: 'Zygarflax biography'. Sources: web"
    actual_output = research.invoke({"plan_or_feedback": plan_text})
    
    # Якщо він вигадає факти, groundedness впаде.
    mock_retrieval_context = ["There are no search results for Zygarflax. Zygarflax does not exist."]
    
    test_case = LLMTestCase(
        input=plan_text,
        actual_output=actual_output,
        retrieval_context=mock_retrieval_context
    )
    
    assert_test(test_case, [groundedness])
