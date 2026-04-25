import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agents.critic import critique

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

def test_critique_approve():
    input_findings = "The capital of France is Paris. The population is approximately 2.1 million. The Eiffel Tower is a major landmark."
    
    # Викликаємо критика
    actual_output = critique.invoke({"findings": input_findings})
    
    test_case = LLMTestCase(
        input=input_findings,
        actual_output=actual_output
    )
    
    assert_test(test_case, [critique_quality])

def test_critique_revise():
    input_findings = "I searched but could not find anything useful about quantum computing. I guess it uses computers."
    
    actual_output = critique.invoke({"findings": input_findings})
    
    test_case = LLMTestCase(
        input=input_findings,
        actual_output=actual_output
    )
    
    assert_test(test_case, [critique_quality])
