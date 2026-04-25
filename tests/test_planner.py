import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.planner import plan

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

def test_plan_quality():
    input_request = "I need a detailed report comparing RAG memory mechanisms."
    
    # Викликаємо функцию plan безпосередньо через invoke.
    # plan.invoke() повертає рядок із планом.
    actual_output = plan.invoke({"request": input_request})
    
    test_case = LLMTestCase(
        input=input_request,
        actual_output=actual_output
    )
    
    assert_test(test_case, [plan_quality])

def test_plan_has_queries():
    input_request = "What are the latest AI models in 2026?"
    actual_output = plan.invoke({"request": input_request})
    
    test_case = LLMTestCase(
        input=input_request,
        actual_output=actual_output
    )
    
    assert_test(test_case, [plan_quality])
