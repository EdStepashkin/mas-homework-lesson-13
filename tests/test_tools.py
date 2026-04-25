import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

tool_metric = ToolCorrectnessMetric(threshold=0.5, model="gpt-4o-mini")

def test_planner_tools():
    # Симулюємо використання tool-ів
    plan_text = "Here is the query."
    tool_call = ToolCall(
        name="web_search",
        description="Search internet for context",
        input_parameters={"query": "latest AI news 2026"},
        output="News article text..."
    )
    
    test_case = LLMTestCase(
        input=plan_text,
        actual_output="Research Plan created.",
        tools_called=[tool_call]
    )
    
    # Tool correctness перевіряє, чи виклики інструментів мали сенс
    assert_test(test_case, [tool_metric])

def test_researcher_tools():
    # Симулюємо Research Agent tools
    input_plan = "Goal: find details on quantum algorithms. Sources: web, knowledge_base"
    
    tool1 = ToolCall(
        name="knowledge_search",
        description="Search local PDF KB",
        input_parameters={"query": "Shor's algorithm"},
        output="Shor's algorithm factors integers in polynomial time."
    )
    
    test_case = LLMTestCase(
        input=input_plan,
        actual_output="Found details about Shor's and Grover's algorithms.",
        tools_called=[tool1]
    )
    
    assert_test(test_case, [tool_metric])

def test_supervisor_save():
    input_critic_approve = "Verdict: APPROVE"
    
    tool_save = ToolCall(
        name="save_report",
        description="Saves final markdown to file",
        input_parameters={"filename": "report.md", "content": "# Report Header..."},
        output="Report saved successfully."
    )
    
    test_case = LLMTestCase(
        input=input_critic_approve,
        actual_output="Report saved.",
        tools_called=[tool_save]
    )
    
    assert_test(test_case, [tool_metric])
